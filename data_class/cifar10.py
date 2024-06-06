from data_class.Base import Base
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

class CIFAR10(Base):
    def __init__(self, batch_size, client_num, non_iid,alpha):
        num_cls=10
        super().__init__(batch_size, client_num, non_iid,num_cls,alpha)

    def init_data(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.train_data = torchvision.datasets.CIFAR10(root="CIFAR10_dataset",train=True,download=True,transform=transform)
        self.test_data = torchvision.datasets.CIFAR10(root="CIFAR10_dataset",train=False,download=True,transform=transform)

        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        # self.train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
        # self.test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        print("successfully read CiFAR10, train data len:",self.train_len, " test_len:", self.test_len)

# e = CiFAR10(64,0.2,10)