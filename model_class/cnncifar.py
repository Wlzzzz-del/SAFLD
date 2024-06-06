import torch
import copy
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class CIFARcnn(nn.Module):
    def __init__(self,in_channels=3, channels=64, out_dim = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)# 32 32
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)# 31 31

        self.conv2 = nn.Conv2d(channels, 2*channels, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(2*channels,3*channels, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3*channels * 4 * 4, 128)
        self.dp = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, out_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dp(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x