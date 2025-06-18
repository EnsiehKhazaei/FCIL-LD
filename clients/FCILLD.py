import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from copy import deepcopy

from clients.simple import AVG
from utiles import combine_data

import torch.nn.functional as F
import copy

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.init as init
import torchvision.transforms as transforms
import random
import itertools

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

class FCILLD_client(AVG):
    def __init__(self, batch_size, epochs, train_dataset, groups, kd_weight, ft_weight, syn_size, dataset_name):
        super(FCILLD_client, self).__init__(batch_size, epochs, train_dataset, groups, dataset_name)
        self.kd_criterion = nn.MSELoss(reduction="none")
        self.last_valid_dim = 0
        self.valid_dim = 0
        self.mu = kd_weight
        self.ft_weight = ft_weight
        self.syn_size = syn_size

    def train(self, args, model, lr, teacher, generator_server, gen_classes, protos):
        
        model.to('cuda')
        model.train()
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        kd_criterion = nn.MSELoss(reduction="none").to("cuda")
        # if teacher is None:
        if generator_server is None or teacher is None:
            for epoch in range(self.local_epoch):
                for i, (x, y) in enumerate(self.train_loader):
                    x, y = augment_and_concatenate(x, y)
                    x, y = x.to('cuda'), y.to('cuda')
                    
                    logits_pen = model.feature(x)
                    logits_pen = model.proj(logits_pen)
                    # Compute cosine similarities for all combinations in a batch
                    similarities = F.cosine_similarity(logits_pen.unsqueeze(1), logits_pen.unsqueeze(0), dim=2)
                    del logits_pen
                    positive_mask = y.unsqueeze(0) == y.unsqueeze(1)
                    positive_mask.fill_diagonal_(False)
                    negative_mask = ~positive_mask
                    

                    exp_similarities = torch.exp(similarities / 0.1)
                    numerators = (exp_similarities * positive_mask)
                    exp_similarities_ = exp_similarities * (1 - torch.eye(exp_similarities.size(0)).to("cuda"))
                    denominators = exp_similarities_.sum(dim=1)     
                    
                    loss_cl = 0
                    for j in range(numerators.shape[0]):
                        non_zero_numerator = numerators[j][numerators[j] != 0]
                        log_values = -torch.log(non_zero_numerator / denominators[j])
                        loss_cl += log_values.sum() / len(non_zero_numerator)
                    

                    with torch.no_grad():
                        feat_class = model.feature(x).detach()
                    loss = self.criterion_fn(model.fc(feat_class), y)
                    loss += 0.1*loss_cl
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            
            x.cpu()
            y.cpu()
            del x
            del y
            model.to('cpu')
            torch.cuda.empty_cache()
       
        else:
            self.train_cl(args, model, teacher, generator_server, opt, gen_classes, protos)
            model.to('cpu')

    def train_cl(self, args, model, teacher, generator_server, opt, gen_classes, protos):
        self.dw_k = torch.ones((self.valid_dim + 1), dtype=torch.float32)
        # previous_teacher, previous_linear = deepcopy(teacher[0]), deepcopy(teacher[1])
        centroids = protos[0]
        final_outs = protos[1]
        teacher.to("cuda")
        generator_server.to("cuda")
        previous_teacher, previous_linear = copy.deepcopy(teacher.feature), copy.deepcopy(teacher.fc)
        n_classes = gen_classes #model.fc.out_features
        local_weights = copy.deepcopy(model.state_dict())
        kd_criterion = nn.MSELoss(reduction="none").to("cuda")
        for epoch in range(self.local_epoch):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to('cuda'), y.to('cuda')
                            
                z = torch.FloatTensor(np.random.normal(0, 1, (self.syn_size, args.z_dim)))#.to('cuda')
                x_replay = generator_server(z)#proto_replay_BN) #z, proto_replay)
                
                global_logits = teacher(x_replay)
                global_logits_softmax = F.softmax(global_logits / args.temp, dim=1).clone().detach()
                _, y_replay = torch.max(global_logits, dim=1)
                x_com, y_com = combine_data(((x, y), (x_replay, y_replay)))
                idx1 = torch.where(y_com >= self.last_valid_dim)[0]
                
                logits_pen = model.feature(x_com)
                logits = model.fc(logits_pen)              
    
                mappings = torch.ones(self.valid_dim, dtype=torch.float32, device='cuda') 
                dw_cls = mappings[y_com.long()]
                loss_class = self.criterion(logits[idx1, self.last_valid_dim:self.valid_dim], (y_com[idx1] - self.last_valid_dim), dw_cls[idx1])

                feat_class = teacher.fc(model.feature(x_replay))

                hkd_Loss = (kd_criterion(feat_class, global_logits).sum(dim=1)).mean() / (feat_class.shape[1])
                del feat_class
                
                model_proj = model.proj(model.feature(x))
                teacher_proj = teacher.proj(teacher.feature(x)).clone().detach()
                
                similarities = F.cosine_similarity(model_proj.unsqueeze(1), teacher_proj.unsqueeze(0), dim=2)
                
                exp_similarities = torch.exp(similarities / 0.5)
                numerators = torch.eye(model_proj.shape[0]).to("cuda") * exp_similarities
                numerators = numerators.sum(dim=1)
                
                denumerators = exp_similarities.sum(dim=1)
                rkd_loss = -torch.log(numerators / denumerators).mean()
                logits_pen = model.feature(x)
                logits_pen = model.proj(logits_pen)
                # Compute cosine similarities for all combinations in a batch
                similarities = F.cosine_similarity(logits_pen.unsqueeze(1), logits_pen.unsqueeze(0), dim=2)#torch.matmul(logits_pen, logits_pen.T)#F.cosine_similarity(logits_pen.unsqueeze(1), logits_pen.unsqueeze(0), dim=2)#F.cosine_similarity(logits_pen.unsqueeze(1), logits_pen.unsqueeze(0), dim=2)

                del logits_pen
              
                with torch.no_grad():
                    feat_class = model.feature(x_com).detach()
                loss = self.criterion_fn(model.fc(feat_class), y_com)
 
                del feat_class

                total_loss = loss_class + 2.0 * hkd_Loss + 5.0 * rkd_loss + 0.5 * loss
                
                opt.zero_grad()
                total_loss.backward()
                opt.step()
                
        x.cpu()
        y.cpu()

        del x
        del y
        # del mappings
        teacher.to('cpu')
        generator_server.to("cpu")
        torch.cuda.empty_cache()
    def kd(self, x_com, previous_linear, logits_pen, previous_teacher):
        kd_index = np.arange(x_com.size(0))
        dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()].to('cuda')
        logits_KD = previous_linear(logits_pen[kd_index])[:, :self.last_valid_dim]
        logits_KD_past = previous_linear(previous_teacher(x_com[kd_index]))[:, :self.last_valid_dim]
        loss_kd = self.mu * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
        del dw_KD
        torch.cuda.empty_cache()
        return loss_kd

    def sample(self, teacher, dim, return_scores=True):
        return teacher.sample(dim, return_scores=return_scores)

    def criterion(self, logits, targets, data_weights):
        return (self.criterion_fn(logits, targets) * data_weights).mean()
