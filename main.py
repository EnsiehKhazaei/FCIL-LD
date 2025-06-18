import os
import numpy as np
from copy import deepcopy

import models
from constant import *
from clients.MFCL import MFCL_client
from clients.DFRD import DFRD_client
from models.ResNet import ResNet18, ResNets
from models.myNetwork import network
from data_prep.data import CL_dataset
from clients.simple import AVG, PROX, ORACLE
from data_prep.super_imagenet import SuperImageNet
from utiles import setup_seed, fedavg_aggregation, evaluate_accuracy_forgetting, evaluate_accuracy, train_gen, start
from utiles import update_gen, create_protos, evaluate_local_accuracy

import logging
import datetime
import copy
import torch

from models.generator import Generator_ACGan, generator_model
import itertools

args = start()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuID
setup_seed(args.seed)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
log_path = log_file_name + '.log'
file_handler = logging.FileHandler(os.path.join('./results', log_path))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info('method: '+args.method)
logger.info('dataset: '+args.dataset)
logger.info('model: '+args.model)
logger.info('num_clients: '+ str(args.num_clients))
logger.info('local epochs: ' + str(args.epochs))
logger.info('local lr: ' + str(args.lr))
logger.info('#rounds per task: ' + str(args.global_round))
logger.info('#selected clients in each round: ' + str(args.frac))
logger.info('batch size: '+str(args.batch_size))
logger.info('alpha distribution: ' + str(args.alpha))

alpha = 0.001

if args.dataset == CIFAR100 or args.dataset == CIFAR10:
    dataset = CL_dataset(args)
    feature_extractor = ResNet18(args.num_classes, cifar=True)
    ds = dataset.train_dataset
elif args.dataset == tinyImageNet:
    dataset = CL_dataset(args)
    feature_extractor = ResNet18(args.num_classes, cifar=False)
    ds = dataset.train_dataset
    args.generator_model = 'TINYIMNET_GEN'
elif args.dataset == SuperImageNet:
    from models.imagenet_resnet import resnet18
    dataset = SuperImageNet(args.path, version=args.version, num_tasks=args.n_tasks, num_clients=args.num_clients, batch_size=args.batch_size)
    args.num_classes = dataset.num_classes
    feature_extractor = resnet18(args.num_classes)
    args.generator_model = 'IMNET_GEN'
    args.img_size = dataset.img_size
    ds = dataset
elif args.dataset == 'ppmi':
    dataset = CL_dataset(args)
    feature_extractor = ResNets(args, use_pretrained=True)
    ds = dataset.train_dataset
    args.generator_model = 'IMNET_GEN'
elif args.dataset == 'voc2012':
    dataset = CL_dataset(args)
    feature_extractor = ResNets(args, use_pretrained=True)
    ds = dataset.train_dataset
    args.generator_model = 'IMNET_GEN'
elif args.dataset == 'stanford':
    dataset = CL_dataset(args)
    feature_extractor = ResNets(args, use_pretrained=True)
    ds = dataset.train_dataset
    args.generator_model = 'IMNET_GEN'

global_model = network(dataset.n_classes_per_task, feature_extractor)
teacher, generator = None, None
gamma = np.log(args.lr_end / args.lr)
task_size = dataset.n_classes_per_task
counter, classes_learned = 0, task_size
gen_classes = 0
num_participants = int(args.frac * args.num_clients)
clients, max_accuracy = [], []
forgetting_list = []
if args.method == MFCL:
    generator = models.__dict__['generator'].__dict__[args.generator_model](zdim=args.z_dim, convdim=args.conv_dim)


class_counts = np.zeros((args.n_tasks * task_size, args.num_clients), dtype=int)
for i in range(args.num_clients):
    group = dataset.groups[i]
    client_idxs = list(itertools.chain(*group))
    for idx in client_idxs:
        _, label = ds[idx]
        class_counts[label][i] += 1


for i in range(args.num_clients):
    group = dataset.groups[i]
    if args.method == FedAVG:
        client = AVG(args.batch_size, args.epochs, ds, group, args.dataset)
    elif args.method == FedProx:
        client = PROX(args.batch_size, args.epochs, ds, group, args.dataset)
    elif args.method == ORACLE:
        client = ORACLE(args.batch_size, args.epochs, ds, group, args.dataset)
    elif args.method == MFCL:
        client = MFCL_client(args.batch_size, args.epochs, ds, group, args.w_kd, args.w_ft, args.syn_size, args.dataset)
    elif args.method == DFRD:
        client = DFRD_client(args.batch_size, args.epochs, ds, group, args.w_kd, args.w_ft, args.syn_size, args.dataset)
    
    clients.append(client)

protos = None
features = []
outputs = []
for t in range(args.n_tasks):
    test_loader = dataset.get_full_test(t)
    [client.set_next_t() for client in clients]
    for round in range(args.global_round):
        weights = []
        lr = args.lr * np.exp(round / args.global_round * gamma)
        
        selected_clients = [clients[idx] for idx in np.random.choice(args.num_clients, num_participants, replace=False)]
        for user in selected_clients:
            model = deepcopy(global_model)
            user.train(args, model, lr, teacher, generator, gen_classes, protos)
            weights.append(model.state_dict())
        global_model.load_state_dict(fedavg_aggregation(weights))
        if (round + 1) % args.eval_int == 0 or (round+1) == args.global_round:
            correct, total, _, _ = evaluate_accuracy(global_model, test_loader, args.method)
            print(f'round {counter}, accuracy: {100 * correct / total}')
            logger.info(f'round {counter}, accuracy: {100 * correct / total}')
            accuracies = evaluate_local_accuracy(model, dataset.get_cl_test(t), weights)
            print(f"Avg accuracy for clients: {np.mean(np.array(accuracies), axis=0)}, {np.mean(accuracies)}")
            logger.info(f"Avg accuracy for clients: {np.mean(np.array(accuracies), axis=0)}, {np.mean(accuracies)}")
            
            if t == 0:
                generator = generator_model(args)
                teacher = copy.deepcopy(global_model)
            if (round + 1) / args.eval_int == 1:
                gen_classes = task_size * (t+1)
            if t != args.n_tasks - 1:
                feature, _ = create_protos(global_model, dataset.get_cl_train(t))
                features.extend(feature)
                protos = [np.array(features), outputs]
                generator = update_gen(copy.deepcopy(global_model), gen_classes, generator, args, protos)#class_counts, selected_clients_idx)
        counter += 1
    if t == 0:
        max_accuracy.append(100 * correct / total)
    if t > 0:
        correct, total, accuracies = evaluate_accuracy_forgetting(global_model, dataset.get_cl_test(t), args.method)
        print(f"total_accuracy_{t}: {accuracies}")
        logger.info(f"total_accuracy_{t}: {accuracies}")
        max_accuracy.append(accuracies[-1])
        forgetting = 0
        for k in range(t):
            forgetting += (max_accuracy[k] - accuracies[k]) / t
        forgetting_list.append(forgetting)
    if t != args.n_tasks - 1:
        if args.method == MFCL or args.method == DFRD:
            teacher = copy.deepcopy(global_model)
            for client in clients:
                client.last_valid_dim = classes_learned
                client.valid_dim = classes_learned + task_size
        classes_learned += task_size
        global_model.Incremental_learning(classes_learned)
print('forgetting:', sum([max_accuracy[i] - accuracies[i] for i in range(args.n_tasks)]) / args.n_tasks)
for i in range(args.n_tasks-1):
    logger.info('forgetting_' + str(i) + ': ' + str(forgetting_list[i]))
logger.info('forgetting: ' + str(sum([max_accuracy[i] - accuracies[i] for i in range(args.n_tasks)]) / args.n_tasks))

