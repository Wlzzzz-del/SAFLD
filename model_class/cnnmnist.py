import torch
import copy
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

# class MNISTcnn(nn.Module):
#     def __init__(self, in_channels=1,channels=16, outdim=10):
#         super().__init__()
#         # 定义卷积层和池化层
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=5, stride=1, padding=2)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=5, stride=1, padding=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         # 定义全连接层
#         self.fc1 = nn.Linear(2*channels * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, outdim)
#         # 定义激活函数
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         # 输入 x 的 shape 为 (batch_size, 1, 28, 28)
#         out = self.conv1(x)  # shape：(batch_size, 16, 28, 28)
#         out = self.relu(out)
#         out = self.pool1(out)  # shape：(batch_size, 16, 14, 14)
#         out = self.conv2(out)  # shape：(batch_size, 32, 14, 14)
#         out = self.relu(out)
#         out = self.pool2(out)  # shape：(batch_size, 32, 7, 7)
#         out = out.view(-1,32*7*7)  # 将张量展开为一维，以便进行全连接
#         out = self.fc1(out)  # shape：(batch_size, 128)
#         out = self.relu(out)
#         out = self.fc2(out)  # shape：(batch_size, 10)

#         return out


class MNISTcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor
