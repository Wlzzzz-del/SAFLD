from data_class.Base import Base
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

class CIFAR100(Base):
    def __init__(self, batch_size, client_num, non_iid,alpha):
        num_cls = 100
        super().__init__(batch_size, client_num, non_iid,num_cls,alpha)

    def init_data(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.train_data = torchvision.datasets.CIFAR100(root="CIFAR100_dataset",train=True,download=True,transform=transform)
        self.test_data = torchvision.datasets.CIFAR100(root="CIFAR100_dataset",train=False,download=True,transform=transform)
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)
        # self.train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
        # self.test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        print("successfully read CiFAR100, train data len:",self.train_len, " test_len:", self.test_len)
