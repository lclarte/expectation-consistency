import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
import torchvision

import random

def get_dataloader_noisy_validation_data(dataset_str, random_seed = 0, train_ratio = 0.8, batch_size_train = 128, batch_size_validation = 128, batch_size_test = 128, flip_ratio = 1.0):
    """
    Flips a proportion of the labels in the 1st class
    For now since we use this code only for the validation set, we can flip the training set as well it does not matter
    """
    random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    # create a transform to be used in the CIFAR10 data loading function that randomly changes the labels of the first class to any class
    target_transform = lambda x: random.randint(0, 9) if (x == 0 and np.random.rand() < flip_ratio) else x


    if dataset_str == 'cifar10_noisy':
        # here we add random flips to the first class so that it can be any of the 9 other classes
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dataset_train_0 = datasets.CIFAR10(root='Datasets/CIFAR10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transform, download=True)
        dataset_train, dataset_validation = torch.utils.data.random_split(dataset_train_0, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))
        dataset_test = torchvision.datasets.CIFAR10(root='Datasets/CIFAR10', train = False, download = True, transform = transforms.Compose([ transforms.ToTensor(), normalize ]), target_transform=target_transform)

    if dataset_str == 'svhn_noisy':
        pre_process = transforms.Compose([transforms.ToTensor()])
        dataset_train_0 = torchvision.datasets.SVHN(root = 'Datasets/SVHN', split='train', download = True, transform = pre_process, target_transform=target_transform)
        dataset_test  = torchvision.datasets.SVHN(root = 'Datasets/SVHN', split='test', download = True, transform = pre_process, target_transform=target_transform)
        dataset_train, dataset_validation = torch.utils.data.random_split(dataset_train_0, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))    

    if dataset_str == 'cifar100_noisy':
        cifar_mean = (0.5071, 0.4867, 0.4408)
        cifar_std  = (0.2673, 0.2564, 0.2761)
        t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(cifar_mean, cifar_std)])
        dataset_train = torchvision.datasets.CIFAR100(root='Datasets/CIFAR100', train = True, download = True, transform = t, target_transform=target_transform)

        # NOTE : Here we will split the test set into validation / test because we want to use pretrained models that will be used on the whole training set
        dataset_test = torchvision.datasets.CIFAR100(root='Datasets/CIFAR100', train = False, download = True, transform = t, target_transform=target_transform)
        # NOTE : here train_ratio will be size of validation set (ugly)
        dataset_validation, dataset_test = torch.utils.data.random_split(dataset_test, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size_train, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size = batch_size_validation, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size_test, shuffle=True)
            
    return train_loader, validation_loader, test_loader



def get_dataloader(dataset_str, random_seed = 0, train_ratio = 0.8, batch_size_train = 128, batch_size_validation = 128, batch_size_test = 128):
    # useful for random transforms
    random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    
    if dataset_str == 'binary_mnist':
        dataset_train = torchvision.datasets.MNIST(root = 'Datasets/', train = True, download=True, transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]), target_transform=lambda x : x % 2)
        dataset_test  = torchvision.datasets.MNIST(root = 'Datasets/', train = False, download = True, transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]), target_transform = lambda x : x % 2)
        dataset_train, dataset_validation = torch.utils.data.random_split(dataset_train, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))

    if dataset_str == 'cifar10':
        # cf https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151 for normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dataset_train_0 = datasets.CIFAR10(root='Datasets/CIFAR10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        dataset_train, dataset_validation = torch.utils.data.random_split(dataset_train_0, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))
        dataset_test = torchvision.datasets.CIFAR10(root='Datasets/CIFAR10', train = False, download = True, transform = transforms.Compose([ transforms.ToTensor(), normalize ]))

    if dataset_str == 'cifar100':
        cifar_mean = (0.5071, 0.4867, 0.4408)
        cifar_std  = (0.2673, 0.2564, 0.2761)
        t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(cifar_mean, cifar_std)])
        dataset_train = torchvision.datasets.CIFAR100(root='Datasets/CIFAR100', train = True, download = True, transform = t)
        # NOTE : Here we will split the test set into validation / test because we want to use pretrained models that will be used on the whole training set
        dataset_test = torchvision.datasets.CIFAR100(root='Datasets/CIFAR100', train = False, download = True, transform = t)
        # NOTE : here train_ratio will be size of validation set (ugly)
        dataset_validation, dataset_test = torch.utils.data.random_split(dataset_test, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))

    if dataset_str == 'svhn':
        pre_process = transforms.Compose([transforms.ToTensor()])
        dataset_train = torchvision.datasets.SVHN(root = 'Datasets/SVHN', split='train', download = True, transform = pre_process)
        dataset_test  = torchvision.datasets.SVHN(root = 'Datasets/SVHN', split='test', download = True, transform = pre_process)
        dataset_train, dataset_validation = torch.utils.data.random_split(dataset_train, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))    
    
    ###
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size_train, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size = batch_size_validation, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size_test, shuffle=True)
            
    return train_loader, validation_loader, test_loader
