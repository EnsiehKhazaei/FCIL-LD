import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader, Subset

from constant import *

import random
import torch
import numpy as np

class AVG:
    def __init__(self, batch_size, epochs, train_dataset, groups, dataset_name):
        self.criterion_fn = F.cross_entropy
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.groups = groups
        self.current_t = -1
        self.local_epoch = epochs
        self.dataset_name = dataset_name
        # self.args = args
        
        

    def set_dataloader(self, samples):
        if self.dataset_name in [CIFAR100, tinyImageNet, CIFAR10, 'ppmi', 'voc2012', 'stanford']:
            self.train_loader = DataLoader(Subset(self.train_dataset, samples), batch_size=self.batch_size, shuffle=True)
        if self.dataset_name == SuperImageNet:
            self.train_loader = self.train_dataset.get_dl(samples, train=True)

    def set_next_t(self):
        self.current_t += 1
        # print('self.current_t')
        # print(self.current_t)
        samples = self.groups[self.current_t]
        
        # Subset(self.train_dataset, samples)
        # data = [d for d in dataset if d[1] in classes]
        # memory_samples = []
        # selected_memory_samples = None
        # if self.current_t != 0:
        #     for t in range(self.current_t):
        #         memory_samples.extend(self.groups[t])
        #     selected_memory_samples = random.sample(memory_samples, self.args.ipc * (self.current_t))
    
        # print(samples)
        # print('len(samples)')
        # print(len(samples))
        self.set_dataloader(samples)

    def train(self, model, lr, teacher, generator_server, glob_iter_):
        model.to('cuda')
        model.train()
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        for epoch in range(self.local_epoch):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to('cuda'), y.to('cuda')
                logits = model(x)
                loss = self.criterion_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.to('cpu')

    # def create_protos(self, model):
    #     model.to("cuda").eval()
    #     features, labels = [], []
    #     # for task_id, test_loader in enumerate(train_loaders):
    #     for i, (x, y) in enumerate(self.train_loader):
    #             x, y = x.to('cuda'), y.to('cuda')
    #             with torch.no_grad():
    #                 outputs = model(x)
    #             features.extend(model.feature(x).cpu().detach().numpy())
    #             labels.extend(y.cpu().numpy()) 
    #     features = np.array(features)
    #     labels = np.array(labels)
        
    #     unique_labels = np.unique(labels)
    #     # Sort the unique labels in ascending order
    #     # unique_labels = np.sort(unique_labels)
    #     centroids = np.array([features[labels == label].mean(axis=0) for label in range(min(unique_labels),max(unique_labels)+1)])
    #     model.to("cpu")
    #     return centroids


class PROX(AVG):
    def __init__(self, batch_size, epochs, train_dataset, groups, dataset_name):
        super(PROX, self).__init__(batch_size, epochs, train_dataset, groups, dataset_name)
        self.mu = 0.01

    def train(self, model, lr, teacher, generator_server, glob_iter_):
        model.to('cuda')
        model.train()
        global_model = deepcopy(model)
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        for epoch in range(self.local_epoch):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to('cuda'), y.to('cuda')
                logits = model(x)
                opt.zero_grad()
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = self.criterion_fn(logits, y) + (self.mu / 2) * proximal_term
                loss.backward()
                opt.step()
        model.to('cpu')


class ORACLE(AVG):
    def __init__(self, batch_size, epochs, train_dataset, groups, dataset_name):
        super(ORACLE, self).__init__(batch_size, epochs, train_dataset, groups, dataset_name)

    def set_next_t(self):
        self.current_t += 1
        current_group = []
        for task in range(self.current_t + 1):
            current_group.extend(self.groups[task])
        self.set_dataloader(current_group)

    def train(self, model, lr, teacher, generator_server, glob_iter_):
        model.to('cuda')
        model.train()
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        for epoch in range(self.local_epoch):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to('cuda'), y.to('cuda')
                logits = model(x)
                loss = self.criterion_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.to('cpu')
