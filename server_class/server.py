from utils.selector import selector
from data_class.cifar10 import CIFAR10
from model_class.model import SimpleCNN
from client_class.client import client
from torch.utils.data import DataLoader
from collections import OrderedDict
from data_class.cifar100 import CIFAR100
from data_class.Fashion_MNIST import FashionMNIST
from data_class.EMNIST import EMNIST
from model_class.cnncifar import CIFARcnn
from model_class.cnnmnist import MNISTcnn
from client_class.malicious_client import malicious_client
from detector_class.SAFLD import SAFLD
from detector_class.FLDetector import FLDetector
from detector_class.FLDNorm import FLDNorm
import copy
import random
import torch
import numpy as np

class TWAFL:
    def __init__(self,dataset, batch_size, num_clients, out_rate, times, noniid, lr, alpha, T,dev, num_malicious,detector,K,file_name,args):

        self.p = num_clients
        self.cids = [i for i in range(num_clients)]

        self.batch_size = batch_size
        self.num_chosen=int((1-out_rate)*num_clients)
        self.lr = lr# 学习率
        self.alpha = alpha# 狄利克雷分布参数
        self.times = times# 客户端训练步数
        self.dev = dev# 设备
        self.T = T# 服务器上总的训练次数
        self.kernel_size=256
        self.K = K
        self.e = torch.exp(torch.tensor(1.))
        self.num_malicious = num_malicious
        self.detector = detector
        self.filename = file_name
        # 记录检测结果
        self.daccs = []
        self.recalls = []
        self.precisions = []
        self.fprs = []
        self.fnrs = []

        # 初始化数据集
        # non-iid= 1 \ 2 \ 3
        # 1----每个用户只持有若干类样本,CiFar10--4类，CiFar100--12类
        # 2----每个用户持有所有的类，但是其中有两类(CiFar100是4类)数量比其他类要多
        # 3----迪利克雷分布

        if dataset=="CIFAR10":
            self.all_data, self.test_data = CIFAR10(batch_size,num_clients+num_malicious,noniid,alpha).get_data()
        if dataset == "CIFAR100":
            self.all_data, self.test_data = CIFAR100(batch_size,num_clients+num_malicious,noniid, alpha).get_data()
            pass
        if dataset == "FashionMNIST":
            self.all_data, self.test_data = FashionMNIST(batch_size,num_clients+num_malicious,noniid, alpha).get_data()
            pass
        if dataset == "EMNIST":
            self.all_data, self.test_data = EMNIST(batch_size,num_clients+num_malicious,noniid,alpha).get_data()
            pass

        # 用于全局测试的数据集
        self.test_data = DataLoader(self.test_data,100,shuffle=True)

        self.ini_prior = [len(self.all_data[i]) for i in self.all_data]# 返回每个client数据集的大小
        print(self.ini_prior)


        # 节点选择
        self.selector = selector(self.cids,self.ini_prior)

        # 生成可变kernel_size大小的客户端模型,并初始化客户端
        # if(dataset == "CIFAR100"):
        #     self.global_model = SimpleCNN(input_size=3, hidden_size=[256,128,64,32],classes_size=100)
        # else:
        #     self.global_model = SimpleCNN(input_size=3, hidden_size=[256,128,64,32],classes_size=10)
        # self.global_model = CIFARcnn()
        if(dataset == "CIFAR10"):
            self.global_model = CIFARcnn()
        elif(dataset == "EMNIST"):
            self.global_model = MNISTcnn()
        else:
            raise ValueError("unknown dataset.")


        # 初始化客户端
        self.cs = [client(cid, copy.deepcopy(self.global_model), lr, times, DataLoader(self.all_data[cid], self.batch_size, shuffle=True),self.batch_size, self.dev, test_data=self.test_data) for cid in self.cids]
        self.global_model.to(self.dev)

        self.malicious = [malicious_client(i+len(self.cs),copy.deepcopy(self.global_model),lr, times,DataLoader(self.all_data[i+len(self.cs)],shuffle=True),self.batch_size,self.dev,test_data=self.test_data,args=args) for i in range(self.num_malicious)]
        self.cs.extend(self.malicious)# 把攻击者加入所有客户端中

        # For Test

        if self.detector == "SAFLD":
            self.detector = SAFLD()
            pass
        elif self.detector == "FLD":
            self.detector = FLDetector()
            pass
        elif self.detector == "FLDNorm":
            self.detector = FLDNorm()
            pass
        else:
            raise ValueError("unknown detector.")


    def node_select(self):
        return self.selector.random_select()


    def global_test_model(self,):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in self.test_data:
                img, label= img.to(self.dev),label.to(self.dev)
                y = self.global_model(img)
                total += len(img)
                label = label.float()
                y = torch.argmax(y,dim=1).float()
                correct += (y == label).sum().item()
        acc = correct/total
        print("准确率为：",acc)
        return acc

        pass

    def local_test_model(self):
        losses = []
        accs = []
        for c in self.cs:
            # 进行客户端本地测试
            los,acc = c.local_test()
            losses.append(los)
            accs.append(acc)
        print("本地模型平均准确率为：",sum(accs)/len(self.cs))
        print("本地模型平均loss为：",sum(losses)/len(self.cs))
        pass

    def aggreate_model(self,grads_and_tau):
        # grads and tau
        collect_grad = [torch.zeros_like(i) for i in (grads_and_tau[0][0])]
        for grad_tau  in grads_and_tau:
            grad = grad_tau[0]
            tau = grad_tau[1]
            weight = (self.e/2)**(-tau)# 根据权重过时调整
            # print(weight)
            for i in range(len(collect_grad)):
                # collect_grad[i] = 1/self.K*weight*grad[i]
                collect_grad[i] += weight*grad[i]

        # 聚合梯度
        temp_dict = copy.deepcopy(self.global_model.state_dict())
        for i,j in zip(temp_dict,collect_grad):
            temp_dict[i] -= self.lr*j

        # 更新梯度
        self.global_model.load_state_dict(temp_dict)
        pass

    def evaluate_detection(self,benign_c,malicious_c,all_client,malicious):
        # 测试指标
        all = len(all_client)
        TP = 0# 实际为恶意，预测为恶意
        FP = 0# 实际为良性，预测为恶意
        FN = 0# 实际为恶意，预测为良性
        TN = 0# 实际为良性，预测为良性
        for c in malicious_c:
            if c in malicious:
                TP+=1
            else:
                FP+=1
        for c in benign_c:
            if c not in malicious:
                TN+=1
            else:
                FN+=1
        print(TP,FP,FN,TN)
        dacc = (TP+TN)/all
        recall = (TP)/(TP+FN)
        precision = TP/(TP+FP)
        fpr = FP/(TN+FP)
        fnr = FN/(TP+FN)
        print("ACC:",dacc," recall:",recall," precision:",precision," fpr:",fpr," fnr:",fnr)
        self.daccs.append(dacc)
        self.recalls.append(recall)
        self.precisions.append(precision)
        self.fprs.append(fpr)
        self.fnrs.append(fnr)

        pass

    def run(self):
        file = open(self.filename,"w")
        print("Start Server now!")
        losses = []

        # K-Gradient part
        self.grad_client = {}
        self.tau_client = {}
        self.time_client={}

        # 初始化
        # self.cs.append(c for c in self.malicious)
        for c in self.cs:
            self.tau_client[c.cid] = 1# 所有客户端的初始tau=1
            self.time_client[c.cid] = 0# 所有客户端的初始训练轮=1
            loss, gradient = c.run()
            self.grad_client[c.cid] = gradient

        for t in range(self.T):
            # 训练模型
            # c_tomodel 保存每轮客户端以及对应的model
            # cs_curt 为该轮未缺席的客户端

            # 清空c_tomodel
            losses.clear()

            # 抽取K个客户端
            self.cs_curt = random.sample(self.cs,self.K)
            self.cs_curt.extend(self.malicious)
            # if(t > 50):
                # 如果t>50则开始检测
            # 开始检测
            if self.num_malicious!=0:
                benign_c,malicious_c = self.detector.detect(self.cs_curt,copy.deepcopy(self.global_model.state_dict()),t,self.tau_client,self.time_client)
                # evaluate
                if t>51:
                    # 传入良性客户端、恶意客户端、本轮客户端、所有的恶意客户端
                    self.evaluate_detection(benign_c,malicious_c,self.cs_curt,self.malicious)
                    self.cs_curt = benign_c

            cs_ntin = [c for c in self.cs if c not in self.cs_curt]

            for c in cs_ntin:# 当前轮没有被chosen的客户端tau+1
                self.tau_client[c.cid] += 1

            grads_and_tau = []#(grad,tau)
            for c in self.cs_curt:# 当前轮被chosen的客户端
                grads_and_tau.append((self.grad_client[c.cid],self.tau_client[c.cid]))
                self.tau_client[c.cid] = 1
                self.time_client[c.cid] = t
                pass

            # 把符合条件的梯度送入聚合的列表
            # 聚合模型
            self.aggreate_model(grads_and_tau)
            # 聚合之后调用记录

            if self.num_malicious>0:
                self.detector.record(copy.deepcopy(self.global_model.state_dict()),t)

            # 更新本地模型
            for c in self.cs_curt:
                gst = copy.deepcopy(self.global_model.state_dict())
                c.model.load_state_dict(gst)
                loss,gradient = c.run()
                losses.append(loss)

                del self.grad_client[c.cid]# 先清除上一轮的结果

                self.grad_client[c.cid] = gradient

            # print("successful")
            print("---------ROUND:",t,"-----------------")
            acc = self.global_test_model()# 全局的一个测试集测试
            file.write(str(acc)+"\n")
            file.flush()
            print("客户端平均损失为:",np.mean(losses)/len(self.cs_curt))
            print("-------------------------------------")

        file.write("dacc:"+str(np.mean(self.daccs))+"\n")
        file.write("recall:"+str(np.mean(self.recalls))+"\n")
        file.write("precisions:"+str(np.mean(self.precisions))+"\n")
        file.write("fpr:"+str(np.mean(self.fprs))+"\n")
        file.write("fnr:"+str(np.mean(self.fnrs))+"\n")
        file.close()

        pass

    pass

