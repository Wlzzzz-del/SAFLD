import torch
import copy
import copy
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader
from model_class.model import SimpleCNN
from model_class.cnncifar import CIFARcnn
from model_class.cnnmnist import MNISTcnn


class KAH_client():
    def __init__(self, cid,  model, lr, steps,data,batch_size, dev ,test_data,ratio):
        # ci - compute_source of clienti
        # pi - data volume of clienti
        # di - delay of clienti

        self.lr = lr
        self.cid = cid
        self.steps = steps
        self.dev = dev
        self.critierion = nn.CrossEntropyLoss()
        self.reset_model(model)

        self.data = data
        self.batch_size = batch_size
        self.global_test_set = test_data
        self.ratio = ratio
        self.clip_train_test_data()

        pass

    def reset_model(self,model):
        self.model = copy.deepcopy(model)
        self.model.to(self.dev)

    def clip_train_test_data(self):
        total = len(self.data.dataset)
        num1 = int(total*self.ratio)
        self.train_set = DataLoader(self.data.dataset[0:num1],batch_size=self.batch_size,shuffle=True)
        self.local_test_set = DataLoader(self.data.dataset[num1+1:total],batch_size=64)

    def load_grads(self,grads):
        for i,p in enumerate(self.model.parameters()):
            if p.grad is not None:
                p.grad += grads[i]

    def set_teacher(self, model):
        pass

    def run(self):
        total_loss = 0
        opt = optim.SGD(self.model.parameters(),lr=self.lr)
        gradient_ts = []# 存放梯度

        # 累加多次梯度
        opt.zero_grad()
        for t in range(self.steps):
            img,label = next(iter(self.data))
            img,label = img.to(self.dev), label.to(self.dev)
            y = self.model(img)

            loss = self.critierion(y, label)# 自带softmax
            loss.backward()

            opt.step()

            total_loss+=loss.item()

        for name, v in self.model.named_parameters():
            gradient_ts.append(v.grad.data)
        return total_loss,gradient_ts

    def global_test(self):
        # 利用全局测试集进行测试
        return self.__test(self.global_test_set)

    def local_test(self):
        # 利用本地测试集进行测试
        return self.__test(self.local_test_set)

    def __test(self,data):
        # 客户端进行本地测试
        total_loss = 0
        total_acc = 0
        total_item = 0

        img,label = next(iter(self.local_test_set))
        total_item += len(label)
        img,label = img.to(self.dev), label.to(self.dev)
        y = self.model(img)
        # label=label.float()
        # print(y)
        loss = self.critierion(y, label)
        total_loss+=loss.item()

        # 这边估计要改
        # y = torch.softmax(y.data,1)
        _,y = torch.max(y,1)
        # print(y)
        total_acc += (y == label).sum().item()

        total_acc = total_acc/total_item
        return total_loss,total_acc
