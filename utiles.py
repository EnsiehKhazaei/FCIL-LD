import torch
import random
import argparse
import numpy as np
from copy import deepcopy

from constant import *
from clients.helper import Teacher

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import numpy as np
from models.ResNet import ResNet18, ResNets

import math
import torchvision
from torchvision.utils import make_grid, save_image
from PIL import Image



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.init as init
import torchvision.transforms as transforms
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    # torch.cuda.empty_cache()


def fedavg_aggregation(weights):
    w_avg = deepcopy(weights[0])
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights))
    return w_avg


def evaluate_accuracy(model, test_loader, method=None):
    model.to('cuda')
    model.eval()
    correct, total = 0, 0
    features, labels = [], []
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to('cuda'), y.to('cuda')
        with torch.no_grad():
            outputs = model(x)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == y.cpu()).sum()
        total += len(y)
        features.extend(model.feature(x).cpu().detach().numpy())
        labels.extend(y.cpu().numpy())  # Ensure labels are on the CPU and in numpy format
        x.cpu()
        y.cpu()
        del x
        del y
    
    model.to('cpu')
    return correct, total, features, labels


def evaluate_accuracy_forgetting(model, test_loaders, method=None):
    c, t = 0, 0
    accuracies = []
    features, labels = [], []
    for task_id, test_loader in enumerate(test_loaders):
        ci, ti, fi, labi = evaluate_accuracy(model, test_loader, method)
        accuracies.append(100 * ci / ti)
        c += ci
        t += ti
        features.extend(fi)
        labels.extend(labi)
    features = np.array(features)
    labels = np.array(labels)
    
    # Calculate centroids for each class
    unique_labels = np.unique(labels)
    centroids = np.array([features[labels == label].mean(axis=0) for label in unique_labels])
    
    combined_data = np.vstack((features, centroids))

    # Apply t-SNE on the combined data
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_data)
    
    # Separate t-SNE results for features and centroids
    num_features = features.shape[0]
    tsne_features = tsne_results[:num_features]
    tsne_centroids = tsne_results[num_features:]
    
    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = labels == label
        plt.scatter(tsne_features[indices, 0], tsne_features[indices, 1], label=f'Class {label}', alpha=0.5)
    plt.scatter(tsne_centroids[:, 0], tsne_centroids[:, 1], c='black', marker='x', s=100, label='Centroids')


    plt.legend()
    plt.title('t-SNE of Image Representations by Class with Centroids')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('tsne_with_centroids.jpg')

    return c, t, accuracies

def evaluate_local_accuracy(model, test_loaders, local_weights):
    accuracies = []
    for weight in local_weights:
        model.load_state_dict(weight)
        client_acc = []
        for task_id, test_loader in enumerate(test_loaders):
            ci, ti, fi, labi = evaluate_accuracy(model, test_loader)
            client_acc.append(100 * ci / ti)
        accuracies.append(client_acc)
    return accuracies

def create_protos(model, train_loaders):
    model.to("cuda").eval()
    features, outputs, labels = [], [], []

    for i, (x, y) in enumerate(train_loaders):
        x, y = x.to('cuda'), y.to('cuda')
        
        with torch.no_grad():
            output = model(x)
            probabilities = torch.softmax(output, dim=1)
        
        # Filter samples with high probability
        max_probs, preds = torch.max(probabilities, dim=1)
        high_prob_indices = max_probs > 0.5
        
        # Only keep features and labels for samples with high probability
        high_prob_features = model.proj(model.feature(x[high_prob_indices]))
        high_prob_outputs = model(x[high_prob_indices])
        high_prob_labels = y[high_prob_indices]
        if high_prob_features.ndim == 1:
            high_prob_features = high_prob_features.unsqueeze(0)
            high_prob_outputs = high_prob_outputs.unsqueeze(0)
            high_prob_labels = high_prob_labels.unsqueeze(0)
        # print(high_prob_labels.shape)
        features.extend(high_prob_features.cpu().detach().numpy())
        outputs.extend(high_prob_outputs.cpu().detach().numpy())
        labels.extend(high_prob_labels.cpu().numpy())
    features = np.array(features)
    outputs = np.array(outputs)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    centroids = np.array([features[labels == label].mean(axis=0) for label in unique_labels])
    del features
    final_outputs = np.array([outputs[labels == label].mean(axis=0) for label in unique_labels])
    del outputs
    del labels
    model.to("cpu")
    return centroids, final_outputs

    
def train_gen(model, valid_out_dim, generator, args):
    dataset_size = (-1, 3, args.img_size, args.img_size)
    model.to('cuda')
    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=args.generator_lr)
    teacher = Teacher(solver=model, generator=generator, gen_opt=generator_optimizer,
                      img_shape=dataset_size, iters=args.pi, deep_inv_params=[1e-3, args.w_bn, args.w_noise, 1e3, 1],
                      class_idx=np.arange(valid_out_dim), train=True, args=args)
    # teacher.sample(args.server_ss, return_scores=False)
    return teacher, deepcopy(model.fc)
class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer, y_input=None, diversity_loss_type=None):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        if diversity_loss_type == 'div2':
            y_input_dist = self.pairwise_distance(y_input, how='l1')
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        if diversity_loss_type == 'div2':
            return torch.exp(-torch.mean(noise_dist * layer_dist * torch.exp(y_input_dist)))
        else:
            return torch.exp(-torch.mean(noise_dist * layer_dist))
        

def augment_and_concatenate(x, y):
    """
    Applies a random augmentation to each sample in the input tensor x,
    then concatenates the original inputs with the augmented ones.

    Args:
    - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
    - y (torch.Tensor): Corresponding labels tensor of shape (batch_size, ...).

    Returns:
    - torch.Tensor: Concatenated tensor of original and augmented inputs.
    - torch.Tensor: Concatenated tensor of original and duplicated labels.
    """
    # Define possible augmentations
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        # transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(degrees=20),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomResizedCrop(size=(x.shape[2], x.shape[3]), scale=(0.8, 1.0))
    ])

    augmented_x = []
    for i in range(x.size(0)):
        # Choose a random augmentation
        aug_fn = random.choice(augmentations.transforms)
        # Apply the chosen augmentation to the i-th sample
        augmented_sample = aug_fn(x[i])
        augmented_x.append(augmented_sample)

    augmented_x = torch.stack(augmented_x)
    
    # Concatenate original and augmented inputs and labels
    x_concat = torch.cat([x, augmented_x], dim=0)
    y_concat = torch.cat([y, y], dim=0)
    return x_concat, y_concat
    
class DeepInversionFeatureHook():
    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        x = module.running_var.data.type(var.type())
        y = module.running_mean.data.type(var.type())
        m1 = torch.log(var**(0.5) / (x + 1e-8)**(0.5)).mean()
        r_feature = m1 - 0.5 * (1.0 - (x + 1e-8 + (y - mean)**2) / var).mean()
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()
        
class Gaussiansmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).to('cuda')
        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)       
        
def update_gen(global_model, valid_out_dim, generator, args, protos):#cls_counts, selected_clients_idx):
    centroids = protos[0]
    final_outs = protos[1]
    generator.to('cuda')
    global_model.eval()
    global_model.to("cuda")
    loss_r_feature_layers = []
    for module in global_model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.GroupNorm):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, args.w_bn))

    
    CE = nn.CrossEntropyLoss().to("cuda")
    # KL_batchmean = nn.KLDivLoss(reduction="batchmean").to("cuda")
    diversity_loss = DiversityLoss(metric='l2').to("cuda")
    smoothing = Gaussiansmoothing(3, 5, 1)
    mse_loss = nn.MSELoss(reduction="none").to('cuda')
    kd_criterion = nn.MSELoss(reduction="none").to("cuda")
    L_FID, L_TRAN, L_DIV, L_EY = 0, 0, 0, 0
    optimizer_G = optim.Adam(generator.parameters(), lr=args.generator_lr)

    for e in range(args.gen_epochs):
        bn_loss = 0
        generator.zero_grad()
        optimizer_G.zero_grad()
        
        z = torch.FloatTensor(np.random.normal(0, 1, (args.server_ss, args.z_dim)))#.to('cuda')
        y_replay = torch.argmax(z[:,:valid_out_dim], dim=1)#torch.randint(0, valid_out_dim, (args.server_ss,)).to('cuda')
        x_replay = generator(z)
        global_logits = global_model(x_replay)

        global_logits_softmax = F.softmax(global_logits / args.temp, dim=1).clone().detach()
        _, global_pre = torch.max(global_logits, -1)
        L_CE = CE(global_logits_softmax, y_replay.to("cuda"))

        
        softmax_o_T = F.softmax(global_logits, dim=1).mean(dim=0)
        ie_loss = (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(valid_out_dim)).sum()) * args.w_ie
        
        inputs_smooth = smoothing(F.pad(x_replay, (2, 2, 2, 2), mode='reflect'))
        loss_var = mse_loss(x_replay, inputs_smooth).mean()
        noise_loss = args.w_noise * loss_var
        
        
        x_replay.to("cpu")
        k = 0
        for mod in loss_r_feature_layers:
            if k == 0:
                # w_bn = 10*args.w_bn
                w_bn = args.w_bn
            else:
                w_bn = args.w_bn
            loss_distr = mod.r_feature * w_bn / len(loss_r_feature_layers)
            bn_loss = bn_loss + loss_distr
            k += 1
        
        
        loss_gen = L_CE + noise_loss + bn_loss + ie_loss  
        if (e+1) % 1000 == 0:
            # print(f'Epoch {e}: L_gen: {loss_gen}, L_CE: {L_CE}, L_fid: {L_fid}, L_tran: {args.beta_tran * L_tran}, L_div: {args.beta_div * L_div}, L_ey: {args.beta_ey * L_ey}, L_bn: {bn_loss}')
            print(f'Epoch {e+1}: L_gen: {loss_gen}, L_CE: {L_CE}, L_noise: {noise_loss}, L_bn: {bn_loss}, L_div: {ie_loss}')

            first_40_images = x_replay[:40]
            del x_replay

            # Create a grid of the first 40 images
            grid = make_grid(first_40_images, nrow=8)  # 8 images per row
            
            # Save the grid as an image using torchvision
            save_image(grid, f'generated_images_{e+1}.png')
        loss_gen.backward()
        optimizer_G.step()
        
    generator.to('cpu')
    global_model.to('cpu')
    del CE
    # del KL_batchmean
    del diversity_loss
    del mse_loss
    del global_model
    torch.cuda.empty_cache()
    return generator

def combine_data(data):
    x, y = [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    x, y = torch.cat(x), torch.cat(y)
    return x, y


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuID', type=str, default='0', help="GPU ID")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--method', type=str, default=FedAVG, help="name of method", choices=[FedAVG, FedProx, MFCL, FCILLD])
    parser.add_argument('--dataset', type=str, default=CIFAR10)#CIFAR100, help="name of dataset")
    parser.add_argument('--num_clients', type=int, default=5)#50, help='#clients')
    parser.add_argument('--epochs', type=int, default=5)#10, help='Local Epoch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Local Learning Rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Local Bachsize')
    parser.add_argument('--eval_int', type=int, default=10)#100)#100, help='Evaluation intervals')
    parser.add_argument('--global_round', type=int, default=10)#100)#100, help='#global rounds per task')
    parser.add_argument('--frac', type=float, default=0.6)#0.1, help='#selected clients in each round')
    parser.add_argument('--alpha', type=float, default=1.0, help='LDA parameter for data distribution')
    parser.add_argument('--n_tasks', type=int, default=5)#55)#4)#10, help='#tasks')
    parser.add_argument('--syn_size', type=int, default=32)#64, help='size of mini-batch')
    parser.add_argument('--server_ss', type=int, default=64)#128, help='batch size for genrative training')
    parser.add_argument('--pi', type=int, default=100, help='local epochs of each global round')
    parser.add_argument('--generator_lr', type=float, default=0.005)#02)
    parser.add_argument('--z_dim', type=int, default=200)
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--ie_loss', type=int, default=1)
    parser.add_argument('--act_loss', type=int, default=0)
    parser.add_argument('--bn_loss', type=int, default=1)
    parser.add_argument('--noise', type=int, default=1)
    parser.add_argument('--w_ie', type=float, default=1.)
    parser.add_argument('--w_kd', type=float, default=1e-1)
    parser.add_argument('--w_ft', type=float, default=1)
    parser.add_argument('--w_act', type=float, default=0.1)
    parser.add_argument('--w_noise', type=float, default=1e-3)
    parser.add_argument('--w_bn', type=float, default=75)#5e1)#5e1)
    parser.add_argument('--generator_model', type=str, default='CIFAR_GEN', help='name of the generative model')
    parser.add_argument('--path', type=str, help='path to dataset')
    parser.add_argument('--version', type=str, default='L')
    parser.add_argument('--ipc', type=int, default=3, help='#selected image per class')
    parser.add_argument('--gen_epochs', type=int, default=5)#000)#5000, help='#epochs in generator')
    parser.add_argument('--temp', type=int, default=1e3, help='Distillation temperature')
    parser.add_argument('--beta_tran', type=float, default=0.0, help='hyper-parameter of L_tran loss')
    parser.add_argument('--beta_div', type=float, default=1.0, help='hyper-parameter of L_div loss')
    parser.add_argument('--beta_ey', type=float, default=0.0, help='hyper-parameter of L_ey loss')
    parser.add_argument('--model', type=str, default='resnet18', help='clients model')
    args = parser.parse_args()
    args.lr_end = 0.01
    return args
