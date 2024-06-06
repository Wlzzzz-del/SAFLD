from client_class.client import client
import os
import math
import copy
import torch.optim as optim
import torch
from utils.add_trigger import add_trigger

class malicious_client(client):
    # 继承自changed_client
    # 重写run方法
    def __init__(self, cid,  model, lr, steps,data,batch_size, dev ,test_data,args):
        super().__init__(cid,model,lr,steps,data,batch_size,dev,test_data)
        # 根据不同的恶意类型进行不同的设置
        self.args = args
        # self.malicious_type = args.malicious_type
        self.malicious_type = "dba"

        # 后门攻击的任务是将Attack_goal 改变成Attack_label
        self.attack_label = args.attack_label
        self.attack_goal = args.attack_goal

        self.poison_frac = 0.5

    def run(self):
        if self.malicious_type == "badnet":
            # 坏网络攻击
            return self.train_malicious_badnet()
        elif self.malicious_type == "dba":
            # 分布式后门攻击
            return self.train_malicious_dba()
        else:
            print("Error Attack Type!")
            os._exit(0)
    
    def train_malicious_dba(self):
        total_loss = 0
        self.model.train()
        opt = optim.SGD(self.model.parameters(),lr=self.lr)
        epoch_loss = []
        gradient_ts = []# 存放梯度

        opt.zero_grad()
        for t in range(self.steps):
            img,label = next(iter(self.data))
            # 处理成恶意数据
            img,label = self.trigger_data(img,label)

            img,label = img.to(self.dev), label.to(self.dev)
            y = self.model(img)

            loss = self.critierion(y, label)# 自带softmax
            loss.backward()



            for name, v in self.model.named_parameters():
                gradient_ts.append(v.grad.data)

            opt.step()
            total_loss+=loss.item()

        return total_loss,gradient_ts



    def train_malicious_badnet(self):
        pass


    def add_trigger(self,image):
        return add_trigger(self.args,image)

    def trigger_data(self, images, labels):
        # 添加恶意数据
        #  attack_goal == -1 means attack all label to attack_label
        #  attack_goal == -1 意味着攻击所有标签
        if self.attack_goal == -1:
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
            else:
                for xx in range(len(images)):  # poison_frac% poison data
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    if xx > len(images) * self.poison_frac:
                        break
        else:  # trigger attack_goal to attack_label
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    if bad_label[xx]!= self.attack_goal:  # no in task
                        continue  # jump
                    bad_label[xx] = self.attack_label
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                    images = torch.cat((images, bad_data[xx].unsqueeze(0)), dim=0)
                    labels = torch.cat((labels, bad_label[xx].unsqueeze(0)))
            else:  # poison_frac% poison data
                # count label == goal label
                num_goal_label = len(labels[labels==self.attack_goal])
                counter = 0
                for xx in range(len(images)):
                    if labels[xx] != 0:
                        continue
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    counter += 1
                    if counter > num_goal_label * self.poison_frac:
                        break
        return images, labels