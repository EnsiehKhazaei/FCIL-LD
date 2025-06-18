import torch
import torch.nn as nn
from constant import *
"""
@article{fang2019datafree,
    title={Data-Free Adversarial Distillation},
    author={Gongfan Fang and Jie Song and Chengchao Shen and Xinchao Wang and Da Chen and Mingli Song},
    journal={arXiv preprint arXiv:1912.11006}
    year={2019}
}
"""

import torch.nn.init as init

class SmallGen(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super(SmallGen, self).__init__()
        self.z_dim = zdim
        self.out_channel = 32

        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, self.out_channel * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(self.out_channel),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(self.out_channel, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.out_channel, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size, device):
        # sample z
        z = torch.randn(size, self.z_dim).to(device)
        z = z.to(device)
        X = self.forward(z)
        return X


class Generator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz, out_channel):
        super(Generator, self).__init__()
        self.z_dim = zdim
        self.out_channel = out_channel

        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, self.out_channel * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(self.out_channel),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(self.out_channel, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

    def forward(self, z):
        out = self.l1(z.to("cuda"))
        out = out.view(out.shape[0], self.out_channel, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size, device):
        # sample z
        z = torch.randn(size, self.z_dim).to(device)
        z = z.to(device)
        X = self.forward(z)
        return X


class GeneratorMed(nn.Module):
    def __init__(self, zdim, in_channel, img_sz):
        super(GeneratorMed, self).__init__()
        self.z_dim = zdim

        self.init_size = img_sz // 8
        self.l1 = nn.Sequential(nn.Linear(zdim, 128 * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks3(img)
        return img

    def sample(self, size, device):
        # sample z
        z = torch.randn(size, self.z_dim).to(device)
        z = z.to(device)
        X = self.forward(z)
        return X


class GeneratorBig(nn.Module):
    def __init__(self, zdim, in_channel, img_sz, convdim):
        super(GeneratorBig, self).__init__()
        self.z_dim = zdim
        self.init_size = img_sz // (2**5)
        self.dim = convdim
        self.l1 = nn.Sequential(nn.Linear(zdim, self.dim//2 * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(self.dim//2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(self.dim//2, self.dim//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(self.dim//2, self.dim//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(self.dim//2, self.dim//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks4 = nn.Sequential(
            nn.Conv2d(self.dim//2, self.dim//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks5 = nn.Sequential(
            nn.Conv2d(self.dim//2, self.dim//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks6 = nn.Sequential(
            nn.Conv2d(self.dim//2, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )
        
        # self.l_nayer = nn.Linear(512, zdim)
        
        # self.embedding = nn.Embedding(512, zdim)

    def forward(self, z):#z, protos):
        # label_embedding = self.embedding(torch.tensor(protos).to("cuda").long())
        # print(label_embedding.shape)
        # z = torch.mul(z.to("cuda"), label_embedding)
        # init.kaiming_uniform_(self.l_nayer.weight)
        
        # protos = self.l_nayer(protos.to("cuda"))
        out = self.l1(z.to("cuda"))
        out = out.view(out.shape[0], self.dim//2, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks3(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks4(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks5(img)
        img = self.conv_blocks6(img)
        return img

    # def sample(self, size, device):
    #     # sample z
    #     z = torch.randn(size, self.z_dim).to(device)
    #     z = z.to(device)
    #     X = self.forward(z)
    #     return X


def CIFAR_GEN(bn=False, zdim=1000, in_channel=3, img_sz=32, out_channel=128, convdim=64):
    return Generator(zdim, in_channel, img_sz, out_channel)


def TINYIMNET_GEN(bn=False, zdim=1000, convdim=64):
    return GeneratorMed(zdim, in_channel=3, img_sz=64)


def IMNET_GEN(zdim=1000, convdim=64):
    return GeneratorBig(zdim, in_channel=3, img_sz=224, convdim=convdim)

def generator_model(args):
    if args.dataset == CIFAR100 or args.dataset == CIFAR10:
        return CIFAR_GEN(zdim=args.z_dim, convdim=args.conv_dim)
    elif args.dataset == tinyImageNet:
        return TINYIMNET_GEN(zdim=args.z_dim, convdim=args.conv_dim)
    else:
        return IMNET_GEN(zdim=args.z_dim, convdim=args.conv_dim)
    

class Generator_ACGan(nn.Module):
    
    def __init__(self, args):
        super(Generator_ACGan,self).__init__()
        
        self.z_dim = args.z_dim
        self.out_channel = 128
        in_channel = 3
        conv_dim = 64

        self.init_size = 8#img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(self.z_dim, self.out_channel * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(self.out_channel),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(self.out_channel, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

        # self.data_set = args.dataset #['data_set']
        # self.latent_dim = args.z_dim #['latent_dim']
        # self.noise_label_combine = 'mul' #args['noise_label_combine']
        # # if self.data_set in ['CIFAR-10', 'FMNIST', 'MNIST', 'SVHN']:
        # #     self.n_classes = 10
        # # elif self.data_set in ['CIFAR-100']:
        # #     self.n_classes = 100
        # # elif self.data_set in ['Tiny-Imagenet']:
        # #     self.n_classes = 200
        # # elif self.data_set in ['FOOD101']:
        # #     self.n_classes = 101
        
        # # if self.data_set == CIFAR100 :
        # #     self.n_classes = 100
        # # elif self.data_set == tinyImageNet:
        # #     self.n_classes = 200
        # # elif self.data_set in ['FOOD101']:
        # #     self.n_classes = 101        

        
        # if self.noise_label_combine in ['cat']:
        #     input_dim = 2 * self.latent_dim
        # elif self.noise_label_combine in ['cat_naive']:
        #     input_dim = self.latent_dim + self.n_classes
        # else:
        #     input_dim = self.latent_dim


        # self.layer1 = nn.Sequential(nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias = False),
        #                             nn.ReLU(True))
        
        # # self.layer1 = nn.Sequential(nn.ConvTranspose2d(input_dim, 128, 4, 1, 0, bias = False),
        # #                            nn.ReLU(True))


        # self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
        #                             nn.BatchNorm2d(256),
        #                             nn.ReLU(True))
        
        # # self.layer2 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1, bias = False),
        # #                            nn.BatchNorm2d(128),
        # #                            nn.ReLU(True))

        # self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU(True))
        # # self.layer3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
        # #                            nn.BatchNorm2d(64),
        # #                            nn.ReLU(True))
        
        # self.layer4_1 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1, bias = False),
        #                             nn.Tanh())
        # # self.layer4_1 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
        # #                            nn.Tanh())

        # self.layer4_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
        #                            nn.BatchNorm2d(64),
        #                            nn.ReLU(True)) 

        # self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
        #                            nn.Tanh()) 

        # self.layer4_3 = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1, bias = False),
        #                            nn.Tanh()) 

        # self.layer4_4 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1, bias = False),
        #                            nn.BatchNorm2d(128),
        #                            nn.ReLU(True))
        
        # self.layer4_5 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
        #                            nn.BatchNorm2d(64),
        #                            nn.ReLU(True))
        
        # self.layer4_6 = nn.Sequential(nn.ConvTranspose2d(64, 3, 3, 1, 1, bias = False),
        #                            nn.Tanh())
        # # self.embedding = nn.Embedding(self.n_classes, self.latent_dim)
        
        
    def forward(self, z): #, label, n_classes):
        out = self.l1(z)
        out = out.view(out.shape[0], self.out_channel, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        # # noise = noise.unsqueeze(1)
        # self.n_classes = n_classes
        # self.embedding = nn.Embedding(self.n_classes, self.latent_dim).to("cuda")
        # if self.noise_label_combine == 'mul':
        #     label_embedding = self.embedding(label)
        #     # print(noise.shape)
        #     # print(label_embedding.shape)
        #     h = torch.mul(noise, label_embedding)
        # elif self.noise_label_combine == 'add':
        #     label_embedding = self.embedding(label)
        #     h = torch.add(noise, label_embedding)
        # elif self.noise_label_combine == 'cat':
        #     label_embedding = self.embedding(label)
        #     h = torch.cat((noise, label_embedding), dim=1)
        # elif self.noise_label_combine == 'cat_naive':
        #     label_embedding = Variable(torch.cuda.FloatTensor(len(label), self.n_classes))
        #     label_embedding.zero_()
        #     label_embedding.scatter_(1, label.view(-1,1), 1)
        #     h = torch.cat((noise, label_embedding), dim=1)
        # else:
        #     label_embedding = noise
        #     h = noise
            
        # x = h.view(-1, h.shape[1], 1, 1)
        
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # # if self.data_set in ['Tiny-Imagenet', 'FOOD101']:
        # #     x = self.layer4_2(x)
        # #     x = self.layer5(x)
        # # elif self.data_set in ['FMNIST', 'MNIST']:
        # #     x = self.layer4_3(x)
        # if self.data_set == tinyImageNet: # or self.data_set == CIFAR100:
        #     x = self.layer4_2(x)
        #     x = self.layer5(x)
        # # elif self.data_set == CIFAR100:
        # #     x = self.layer4_3(x)
        # elif self.data_set == CIFAR100:
        #     # x = self.layer4_4(x)
        #     # x = self.layer4_5(x)
        #     # x = self.layer4_6(x)
        #     x = self.layer4_1(x)
        # else:
        #     x = self.layer4_1(x)

        return img#, h, label_embedding
