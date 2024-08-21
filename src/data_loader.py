import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

def get_transforms():
    return Compose([ToTensor()])

def get_datasets(data_dir, transforms):
    # Descarga o carga los datasets
    train_dataset = MNIST(data_dir, train=True, transform=transforms, download=True)
    test_dataset = MNIST(data_dir, train=False, transform=transforms, download=True)
    return train_dataset, test_dataset

def get_dataloaders(train_dataset, test_dataset, batch_size):
    # Crea los DataLoader para los datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def setup_data_loader(data_dir, batch_size):
    transforms = get_transforms()
    train_dataset, test_dataset = get_datasets(data_dir, transforms)
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size)
    return train_loader, test_loader