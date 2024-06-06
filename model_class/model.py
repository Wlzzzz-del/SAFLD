import torch
import copy
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class SimpleCNN(nn.Module):# model_rate缩放模型
    def __init__(self, input_size, hidden_size, classes_size):
        super(SimpleCNN, self).__init__()
        blocks = [nn.Conv2d(input_size, hidden_size[0], 3, 1, 1),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)
        ]
        for i in range(len(hidden_size)-1):
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i+1], 3, 1, 1),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)
            ])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),# nn.Flatten(0,-1)
                       nn.Linear(hidden_size[-1], classes_size)
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # 改写
        x = self.blocks(x)
        # x = torch.softmax(x,dim=1)
        return x



