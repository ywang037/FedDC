import time
import os
import sys
import copy
import logging
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms


def get_dataset(dataset):
    data_set, data_info = {}, {}
    # num_classes=10
    if dataset in ['MNIST', 'mnist']:
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform) # no augmentation
        data_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        train_labels = data_train.targets.numpy()
        test_labels = data_test.targets.numpy()
        mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U1')

    elif dataset in ['EMNIST', 'emnist']:
        channel = 1
        im_size = (28, 28)
        num_classes = 62
        mean = [0.5]
        std = [0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.EMNIST(root="./data", split="byclass", download=True, train=True, transform=transform)
        data_test = datasets.EMNIST(root="./data", split="byclass", download=True, train=False, transform=transform)   
        class_names = [str(c) for c in range(num_classes)]
        train_labels = data_train.targets.numpy()
        test_labels = data_test.targets.numpy()
        mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
        'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
        'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], dtype='<U1')

    elif dataset in ["FMNIST", "fmnist"]:
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.5]
        std = [0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        data_train = datasets.FashionMNIST(root="./data/FMNIST", download=True, train=True, transform=transform)
        data_test = datasets.FashionMNIST(root="./data/FMNIST", download=True, train=False, transform=transform)
        class_names = data_train.classes
        train_labels = data_train.targets.numpy()
        test_labels = data_test.targets.numpy()
        mapp = np.array(class_names)
        # mapp = data_train.class_to_idx

    elif dataset in ['SVHN', 'svhn']:
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.SVHN(root="./data/SVHN", split='train', download=True, transform=transform)  # no augmentation
        data_test = datasets.SVHN(root="./data/SVHN", split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        train_labels = data_train.labels
        test_labels = data_test.labels
        mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U1')

    elif dataset in ['CIFAR10', 'cifar10']:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform) # no augmentation
        data_test = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)
        class_names = data_train.classes
        train_labels = np.array(data_train.targets, dtype=np.int32)
        test_labels = np.array(data_test.targets, dtype=np.int32)
        mapp = np.array(data_test.classes)
    
    elif dataset in ['CIFAR100', 'cifar100']:
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.CIFAR100(root='./data/CIFAR100', train=True, download=True, transform=transform) # no augmentation
        data_test = datasets.CIFAR100(root='./data/CIFAR100', train=False, download=True, transform=transform)
        class_names = data_train.classes
        train_labels = np.array(data_train.targets, dtype=np.int32)
        test_labels = np.array(data_test.targets, dtype=np.int32)
        mapp = np.array(data_test.classes)

    elif dataset in ['CINIC10', 'cinic10', 'CINIC10-IMAGENET', 'cinic10-imagenet']:
        if ('IMAGENET' in dataset) or ('imagenet' in dataset):
            cinic_directory = 'data/cinic-10-imagenet'
            cinic_train_dir = cinic_directory + '/train'
            cinic_test_dir = cinic_directory + '/test'
        else:
            cinic_directory = 'data/CINIC-10'
            cinic_train_dir = cinic_directory + '/train'
            cinic_test_dir = cinic_directory + '/test'
            cinic_val_dir = cinic_directory + '/valid'
        
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])
        data_train = datasets.ImageFolder(cinic_train_dir, transform)
        data_test = datasets.ImageFolder(cinic_test_dir, transform)
        class_names = data_train.classes
        train_labels = np.array(data_train.targets, dtype=np.int32)
        test_labels = np.array(data_test.targets, dtype=np.int32)
        
        if dataset in ['CINIC10', 'cinic10']:
            data_val = datasets.ImageFolder(cinic_val_dir, transform)
            val_labels = np.array(data_train.targets, dtype=np.int32)
        
        mapp = np.array(data_train.classes)


    else:
        sys.exit('unknown dataset: %s'%dataset)

    data_set['train_data']=data_train
    data_set['test_data']=data_test
    data_set['transform']=transform
    data_set['train_labels']=train_labels
    data_set['test_labels']=test_labels
    data_set['mapp']=mapp

    data_info['channel']=channel
    data_info['img_size']=im_size
    data_info['num_classes']=num_classes
    data_info['class_names']=class_names
    data_info['mean']=mean
    data_info['std']=std
    
    if dataset in ['CINIC10', 'cinic10']:
        data_set['valid_data']=data_val
        data_set['valid_labels']=val_labels
    # testloader_server = torch.utils.data.DataLoader(data_test, batch_size=256, shuffle=False, num_workers=0)
    
    return data_set, data_info

class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform
        
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y 

def partition_labeldir(targets, num_classes=10, n_parties=10, beta=1.0, distributions=None):
    """ This data partition function is a copy of that used in paper Federated Learning on Non-IID Data Silos: An Experimental Study.
        dataset: can be both train or test set
        targeets: should be train labels or test labels accordingly
        distributions: if using an existing distributions, pls specify one that generated by np.random.dirichlet()
    """
    min_size = 0
    min_require_size = 10

    # if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
    #     K = 2
    # elif dataset in ['MNIST', 'mnist', 'CIFAR10', 'cifar10', 'SVHN', 'svhn']:
    #     K=10
    # elif dataset in ['cifar100', 'CIFAR100']:
    #     K = 100
    # elif dataset == 'tinyimagenet':
    #     K = 200
    
    N = targets.shape[0]
    #np.random.seed(2020)
    net_dataidx_map = {}
    
    if distributions is None:
        distributions = np.random.dirichlet(np.repeat(beta, n_parties), num_classes)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            # proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = distributions[k]
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        
    return distributions, net_dataidx_map

def record_net_data_stats(y_train, net_dataidx_map, logger=None):

    net_cls_counts = {}
    if net_dataidx_map is not None:
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
            if logger is not None:
                logger.info('Client {:2d} total train data: {:5d}, distribution: {}'.format(net_i, len(dataidx), tmp))
            else:
                print('Client {:2d} total train data: {:5d}, distribution: {}'.format(net_i, len(dataidx), tmp))
    else:
        unq, unq_cnt = np.unique(y_train, return_counts=True)
        for i in range(len(unq)):
            net_cls_counts[unq[i]] = unq_cnt[i]
           
    return net_cls_counts

def make_client_dataset_from_partition(data, num_clients, data_idcs, transform=None):
    client_data = {}
    for client_id in range(num_clients):
        client_data[client_id] = CustomSubset(dataset=data, indices=data_idcs[client_id], subset_transform=transform)
    return client_data