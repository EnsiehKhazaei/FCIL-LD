import os
import PIL
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict

from data_prep.common import create_lda_partitions
from constant import *

import random 
import os
import pickle
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
class PPMI(Dataset):
    def __init__(self, args, Train=False, dataidxs=None, transform=None):
        # self.data_dir = root_dir
        self.dataidxs = dataidxs
        self.transform = transform
        self.org_imgs = []
        self.pixelized_imgs = []
        self.targets = []
        self.classes = ['bassoon', 'cello', 'clarinet', 'erhu', 'flute',
                        'frenchhorn', 'guitar', 'harp', 'recorder',
                        'saxophone', 'trumpet', 'violin']
        root_dir = '/nobackup1/ensieh/datasets/PPMI'
        original_folder = os.path.join(root_dir, 'norm_image')
        # pixelized_folder = os.path.join(root_dir, 'Pixelized_45_norm_image') #'Pixelized_norm_image')
        
        train_img_names = []
        test_img_names = []
        # Iterate through each class folder
        for type_folder in os.listdir(original_folder):
            type_folder_path = os.path.join(original_folder, type_folder)
            for class_folder in os.listdir(type_folder_path):
                class_folder_path = os.path.join(type_folder_path, class_folder)
                # Iterate through train and test folders
                for train_test_folder in os.listdir(class_folder_path):
                    train_test_folder_path = os.path.join(class_folder_path, train_test_folder)
                    for image_file in os.listdir(train_test_folder_path):
                        image_path = os.path.join(train_test_folder_path, image_file)
                        if train_test_folder == 'train':
                            train_img_names.append(image_file)# image_path.replace("\\","/"))
                        elif train_test_folder == 'test':
                            test_img_names.append(image_file)# image_path.replace("\\","/"))
        
        random.seed(42)
        random.shuffle(train_img_names)
        random.shuffle(test_img_names)
        # Saving train_img_names to a file
        with open(args.dataset + '_train_image_names_'+str(args.num_clients)+'.pkl', 'wb') as f:
            pickle.dump(train_img_names, f)
        # Saving test_img_names to a file
        with open(args.dataset + '_test_image_names_'+str(args.num_clients)+'.pkl', 'wb') as f:
            pickle.dump(test_img_names, f)
        
        # with open(config.dataset_name+'_'+config.partition+'_test_image_names_'+str(config.n_parties)+'.pkl', 'rb') as f:
        #     test_img_names = pickle.load(f)
        # with open(config.dataset_name+'_'+config.partition+'_train_image_names_'+str(config.n_parties)+'.pkl', 'rb') as f:
        #     train_img_names = pickle.load(f)
        if Train == True:        
            for img_name in train_img_names:
                if 'with' in img_name.lower():
                    type_folder = 'with_instrument'
                else:
                    type_folder = 'play_instrument'
                class_folder = (img_name.split('_')[2]).lower()
                
                org_img_path = os.path.join(
                    os.path.join(
                        os.path.join(
                            os.path.join(
                                original_folder, 
                                type_folder), 
                            class_folder), 
                        'train'), 
                    img_name)
                
                self.org_imgs.append(org_img_path.replace("\\","/"))
                self.targets.append(self.classes.index(class_folder))
        else:
            for img_name in test_img_names:
                if 'with' in img_name.lower():
                    type_folder = 'with_instrument'
                else:
                    type_folder = 'play_instrument'
                class_folder = (img_name.split('_')[2]).lower()
                
                org_img_path = os.path.join(
                    os.path.join(
                        os.path.join(
                            os.path.join(
                                original_folder, 
                                type_folder), 
                            class_folder), 
                        'test'), 
                    img_name)
                
                self.org_imgs.append(org_img_path.replace("\\","/"))
                self.targets.append(self.classes.index(class_folder))
            #self.labels.append(self.classes.index(img_name[:img_name.rfind('_')]))

    def __len__(self):
        if self.dataidxs is None:
            return len(self.org_imgs)
        else:
            return len(self.dataidxs)

    def __getitem__(self, index):
        org_img_path = self.org_imgs[index]
        img_name = org_img_path[org_img_path.rfind('/')+1:]
        org_img = Image.open(org_img_path).convert('RGB')
        label = self.classes.index((img_name.split('_')[2]).lower())
        if self.transform:
            org_img = self.transform(org_img)
        return org_img, label

class VOC2012(Dataset):
    def __init__(self, args, Train=False, dataidxs=None, transform=None):
        # self.data_dir = root_dir
        self.dataidxs = dataidxs
        self.transform = transform
        self.org_imgs = []
        # self.pixelized_imgs = []
        self.targets = []
        self.classes = ['jumping', 'phoning', 'playinginstrument', 'reading', 
                        'ridingbike', 'ridinghorse', 'running', 'takingphoto', 
                        'usingcomputer', 'walking']
        root_dir = '/nobackup1/ensieh/datasets/VOC2012'
        original_folder = os.path.join(root_dir, 'Action_Classification')
        # pixelized_folder = os.path.join(root_dir, 'Pixelized_Action_Classification') #'Pixelized_norm_image')
        
        train_img_names = []
        test_img_names = []
        # Iterate through each class folder
        for class_folder in os.listdir(original_folder):
            class_folder_path = os.path.join(original_folder, class_folder)
            # Iterate through train and test folders
            for train_test_folder in os.listdir(class_folder_path):
                    train_test_folder_path = os.path.join(class_folder_path, train_test_folder)
                    for image_file in os.listdir(train_test_folder_path):
                        image_path = os.path.join(train_test_folder_path, image_file)
                        if train_test_folder == 'train':
                            train_img_names.append(image_file)# image_path.replace("\\","/"))
                        elif train_test_folder == 'val':
                            test_img_names.append(image_file)# image_path.replace("\\","/"))
        random.seed(42)
        random.shuffle(train_img_names)
        random.shuffle(test_img_names)
        # # Saving train_img_names to a file
        with open(args.dataset + '_train_image_names_'+str(args.num_clients)+'.pkl', 'wb') as f:
            pickle.dump(train_img_names, f)
        # Saving test_img_names to a file
        with open(args.dataset + '_test_image_names_'+str(args.num_clients)+'.pkl', 'wb') as f:
            pickle.dump(test_img_names, f)
        # with open(config.dataset_name+'_'+config.partition+'_test_image_names_'+str(config.n_parties)+'.pkl', 'rb') as f:
        #     test_img_names = pickle.load(f)
        # with open(config.dataset_name+'_'+config.partition+'_train_image_names_'+str(config.n_parties)+'.pkl', 'rb') as f:
        #     train_img_names = pickle.load(f)
            
        if Train == True:        
            for img_name in train_img_names:
                for label in self.classes:
                    org_folder = os.path.join(
                        os.path.join(original_folder, 
                                     label), 
                        'train')

                    if img_name in os.listdir(org_folder):
                        self.org_imgs.append(os.path.join(org_folder, img_name).replace("\\","/"))
                        self.targets.append(self.classes.index(label))
        else:
            for img_name in test_img_names:
                for label in self.classes:
                    org_folder = os.path.join(
                        os.path.join(original_folder, 
                                     label), 
                        'val')

                    if img_name in os.listdir(org_folder):
                        self.org_imgs.append(os.path.join(org_folder, img_name).replace("\\","/"))
                        self.targets.append(self.classes.index(label))

    def __len__(self):
        if self.dataidxs is None:
            return len(self.org_imgs)
        else:
            return len(self.dataidxs)

    def __getitem__(self, index):
        org_img_path = self.org_imgs[index]
        img_name = org_img_path[org_img_path.rfind('/')+1:]
        org_img = Image.open(org_img_path).convert('RGB')
        label = self.targets[index] #self.classes.index((img_name.split('_')[2]).lower())
        if self.transform:
            org_img = self.transform(org_img)
        return org_img, label

class STANFORD40(Dataset):
    def __init__(self, args, Train=False, dataidxs=None, transform=None):
        # self.data_dir = root_dir
        self.dataidxs = dataidxs
        self.transform = transform
        self.org_imgs = []
        self.pixelized_imgs = []
        self.targets = []
        self.classes = ['riding_a_horse', 'reading', 'fixing_a_bike', 'running', 
                        'cutting_vegetables', 'cutting_trees', 'brushing_teeth', 
                        'watching_TV', 'using_a_computer', 'gardening', 'riding_a_bike', 
                        'taking_photos', 'smoking', 'pouring_liquid', 'walking_the_dog', 
                        'climbing', 'writing_on_a_board', 'blowing_bubbles', 'fishing', 
                        'holding_an_umbrella', 'looking_through_a_telescope', 'jumping', 
                        'texting_message', 'feeding_a_horse', 'cleaning_the_floor', 
                        'pushing_a_cart', 'playing_violin', 'fixing_a_car', 'playing_guitar', 
                        'looking_through_a_microscope', 'washing_dishes', 'phoning', 
                        'drinking', 'cooking', 'waving_hands', 'writing_on_a_book', 
                        'applauding', 'throwing_frisby', 'shooting_an_arrow', 'rowing_a_boat']
        root_dir = '/nobackup1/ensieh/contrastive_learning/datasets/Stanford40_JPEGImages'
        original_folder = os.path.join(root_dir, 'JPEGImages')
        # pixelized_folder = os.path.join(root_dir, 'Pixelized_JPEGImages')
        img_names = os.listdir(original_folder)
        train_img_names, test_img_names = train_test_split(img_names, test_size=0.1, random_state=42)

        # # Loading net_dataidx_map from the file
        # with open(config.dataset_name+'_'+config.partition+'_test_image_names_'+str(config.n_parties)+'.pkl', 'rb') as f:
        #     test_img_names = pickle.load(f)
        # with open(config.dataset_name+'_'+config.partition+'_train_image_names_'+str(config.n_parties)+'.pkl', 'rb') as f:
        #     train_img_names = pickle.load(f)
            
        if Train == True:
            for img_name in train_img_names:
                org_img_path = os.path.join(original_folder, img_name)
                # pixelized_img_path = os.path.join(pixelized_folder, img_name)
                self.org_imgs.append(org_img_path.replace("\\","/"))
                # self.pixelized_imgs.append(pixelized_img_path.replace("\\","/"))
                self.targets.append(self.classes.index(img_name[:img_name.rfind('_')]))
        else:
            for img_name in test_img_names:
                org_img_path = os.path.join(original_folder, img_name)
                # pixelized_img_path = os.path.join(pixelized_folder, img_name)
                self.org_imgs.append(org_img_path.replace("\\","/"))
                # self.pixelized_imgs.append(pixelized_img_path.replace("\\","/"))
                self.targets.append(self.classes.index(img_name[:img_name.rfind('_')]))
            #self.labels.append(self.classes.index(img_name[:img_name.rfind('_')]))

    def __len__(self):
        if self.dataidxs is None:
            return len(self.org_imgs)
        else:
            return len(self.dataidxs)

    def __getitem__(self, index):
        org_img_path = self.org_imgs[index]
        # pixelized_img_path = self.pixelized_imgs[index]
        img_name = org_img_path[org_img_path.rfind('/')+1:]
        org_img = Image.open(org_img_path).convert('RGB')
        # pixelized_img = Image.open(pixelized_img_path).convert('RGB')
        label = self.classes.index(img_name[:img_name.rfind('_')])
        if self.transform:
            org_img = self.transform(org_img)
            # pixelized_img = self.transform(pixelized_img)
        return org_img, label

def get_data(args):
    test_transform = transforms.Compose([
                                    transforms.Resize((224,224)),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
                                    
                                    ])
    train_transform = transforms.Compose([
                                    #transforms.RandomResizedCrop(224),
                                    transforms.Resize((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(20),
                                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3)),
                                    #transforms.ColorJitter(brightness=0.2, hue=0.3),
                                    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
                                    #transforms.Normalize(mean=[0.4686, 0.4407, 0.4008],
                                    #                     std=[0.2407, 0.2326, 0.2359])
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #                     std=[0.229, 0.224, 0.225])
                                    ])
    if args.dataset == 'ppmi':        
        test_dataset = PPMI(args, Train=False, transform=test_transform)
        train_dataset = PPMI(args, Train=True, transform=train_transform)
    if args.dataset == 'voc2012':        
        test_dataset = VOC2012(args, Train=False, transform=test_transform)
        train_dataset = VOC2012(args, Train=True, transform=train_transform)
    if args.dataset == 'stanford':        
        test_dataset = STANFORD40(args, Train=False, transform=test_transform)
        train_dataset = STANFORD40(args, Train=True, transform=train_transform)
    return train_dataset, test_dataset

def get_dataset(args):
    # print(args.dataset)
    if args.dataset == CIFAR100:
        return get_cifar100(args)
    elif args.dataset == CIFAR10:
        return get_cifar10(args)
    elif args.dataset == tinyImageNet:
        return get_tiny(args)
    elif args.dataset == 'ppmi':
        return get_data(args)
    elif args.dataset == 'voc2012':
        return get_data(args)
    elif args.dataset == 'stanford':
        return get_data(args)
    else:
        raise NotImplementedError


def get_cifar100(args):
    args.num_classes = 100
    normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    args.img_size = 32
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR100(root=args.path, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=args.path, train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset

def get_cifar10(args):
    args.num_classes = 10
    normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    args.img_size = 32
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR10(root=args.path, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=args.path, train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset

def get_tiny(args):
    args.num_classes = 200

    def parse_classes(file):
        classes = []
        filenames = []
        with open(file) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        for x in range(len(lines)):
            tokens = lines[x].split()
            classes.append(tokens[1])
            filenames.append(tokens[0])
        return filenames, classes

    class TinyImageNetDataset(torch.utils.data.Dataset):
        """Dataset wrapping images and ground truths."""
        def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
            self.img_path = img_path
            self.transform = transform
            self.gt_path = gt_path
            self.class_to_idx = class_to_idx
            self.classidx = []
            self.imgs, self.classnames = parse_classes(gt_path)
            for classname in self.classnames:
                self.classidx.append(self.class_to_idx[classname])
            self.targets = self.classidx

        def __getitem__(self, index):
            img = None
            with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
            y = self.classidx[index]
            return img, y

        def __len__(self):
            return len(self.imgs)

    data_path = args.path
    args.img_size = 64
    train_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'tiny-imagenet-200', 'train'),
        transform=transforms.Compose(
            [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    )

    test_dataset = TinyImageNetDataset(
        img_path=os.path.join(data_path, 'tiny-imagenet-200', 'val', 'images'),
        gt_path=os.path.join(data_path, 'tiny-imagenet-200', 'val', 'val_annotations.txt'),
        class_to_idx=train_dataset.class_to_idx.copy(),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    )
    return train_dataset, test_dataset


class CL_dataset():
    def __init__(self, args):
        self.args = args
        self.name = args.dataset
        self.train_dataset, self.test_dataset = get_dataset(self.args)
        self.classes = np.arange(len(np.unique(self.train_dataset.targets)))
        self.train_ds, self.cl_test_loaders, total_test = [], [], []
        self.full_test_loaders, current_train, current_test = [], [], []
        self.n_classes_per_task = len(self.classes) // self.args.n_tasks
        for i, label in enumerate(self.classes):
            current_train.extend(np.where(self.train_dataset.targets == label)[0].tolist())
            current_test.extend(np.where(self.test_dataset.targets == label)[0].tolist())
            if i % self.n_classes_per_task == (self.n_classes_per_task - 1):
                self.train_ds += [current_train]
                total_test.extend(current_test)
                self.cl_test_loaders.append(DataLoader(Subset(self.test_dataset, current_test), batch_size=self.args.batch_size, shuffle=False))
                self.full_test_loaders.append(DataLoader(Subset(self.test_dataset, deepcopy(total_test)), batch_size=self.args.batch_size, shuffle=False))
                current_train, current_test = [], []
        self.groups = defaultdict(list)
        for task_id in range(args.n_tasks):
            task_group = self.get_task_group(task_id, args.num_clients)
            for client in range(args.num_clients):
                data_i = task_group[client]
                self.groups[client].append(data_i)

    def get_task_group(self, task_id, num_users):
        train_indx = self.train_ds[task_id]
        targets = np.array(self.train_dataset.targets)[train_indx]
        groups, _ = create_lda_partitions(dataset=targets, num_partitions=num_users, concentration=self.args.alpha, accept_imbalanced=True)#True)#False)
        groups = [(np.array(train_indx)[groups[i][0]]).tolist() for i in range(num_users)]
        return groups

    def get_full_train(self, task_id):
        indexes = []
        for t in range(task_id + 1):
            indexes.extend(self.train_ds[t])
        # print(len(indexes))
        return DataLoader(Subset(self.train_dataset, indexes), batch_size=128, shuffle=True)#, num_workers=4, pin_memory=True)

    def get_cl_train(self, task_id):
        indexes = []
        for t in range(task_id, task_id + 1):
            indexes.extend(self.train_ds[t])
        # print(len(indexes))
        return DataLoader(Subset(self.train_dataset, indexes), batch_size=128, shuffle=True)#, num_workers=4, pin_memory=True)


    def get_full_test(self, t):
        return self.full_test_loaders[t]

    def get_cl_test(self, t):
        return self.cl_test_loaders[:t + 1]
