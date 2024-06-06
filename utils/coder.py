import torch
import torch.nn as nn



# 最简单的编码解码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.fc = nn.Linear(input_size,hidden_size)
    def forward(self,x):
        encoded = self.fc(x)
        return encoded

class Decoder(nn.Module):
    def __init__(self,hidden_size,output_size) -> None:
        self.fc = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        decoded = self.fc(x)
        return decoded
    pass