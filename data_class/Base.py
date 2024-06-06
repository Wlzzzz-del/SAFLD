import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import copy
import random
# from data_class.direchlet import gen_direchlet
TEST_LEN=10000

class Base:
    # 所有数据集处理类的抽象类
    # 加入non-iid
    def __init__(self, batch_size, client_num, non_iid, num_cls, alpha):
        self.batch_size = batch_size
        self.client_num = client_num
        self.non_iid = non_iid
        self.num_cls = num_cls
        self.alpha = alpha


        self.init_data()

        self.gen_non_iid()
        # self.split_data_to_client()
        self.get_data()

    def gen_non_iid(self):
        if(self.non_iid == 1):
            if(self.num_cls == 10):
                self.split_non_iid1(self.num_cls, 4)
            elif(self.num_cls == 100):
                self.split_non_iid1(self.num_cls,12)
            pass
        elif(self.non_iid == 2):
            if(self.num_cls == 10):
                self.split_non_iid2(self.num_cls, 2)
            elif(self.num_cls == 100):
                self.split_non_iid2(self.num_cls, 6)
        elif(self.non_iid == 3):
            self.split_non_iid3()

    def split_non_iid3(self):
        # 迪利克雷分布
        s = np.random.dirichlet(np.ones(self.num_cls)*self.alpha, self.client_num)
        data_dist = np.zeros((self.client_num, self.num_cls))
        label_to_data , _ = self.get_term_dict(self.num_cls) 

        for j in range(self.client_num):
            data_dist[j] = ((s[j]*len(label_to_data[0])).astype("int")/ (s[j]*len(label_to_data[0])).astype("int").sum()*(len(label_to_data[0]))).astype("int")
            data_num = data_dist[j].sum()
            data_dist[j][np.random.randint(low=0,high = self.num_cls)]+=((len(label_to_data[0])-data_num))
            data_dist = data_dist.astype("int")
        self.all_data={cid:[] for cid in range(self.client_num)}

        for c in self.all_data:
            a = []
            for i in range(self.num_cls):
                if(data_dist[c][i]!=0):
                    d_index = np.random.randint(low=0, high = len(label_to_data[i]),size = data_dist[c][i])
                    for idx in d_index:
                        a.append(label_to_data[i][idx])
            self.all_data[c] = a

    def get_term_dict(self,num_cls, cls_per_c=2):
        # 每个用户持有所有的类，但是其中有两类(CiFar100是4类)数量比其他类要多
        clss = list(range(num_cls))
        cids = list(range(self.client_num))# 用户标签
        term_dict = {}
        for cid in cids:
            term = random.sample(clss,k=cls_per_c)# 抽样两类拥有较多样本
            term_dict[cid] = term
        # 获得标签对应数据的字典
        label_to_data = {c:[] for c in clss}# key:标签->item:数据
        for data in self.train_data:
            label_to_data[data[1]].append(data)
        return label_to_data,term_dict


    def split_non_iid1(self, num_cls, cls_per_c=4):
        # 每个用户只持有若干类样本,CiFar10--4类，CiFar100--12类
        cids = list(range(self.client_num))# 用户标签
        term_dict = {}
        clss = list(range(num_cls))# 类标签
        for cid in cids:
            term = random.sample(clss,k=cls_per_c)
            term_dict[cid] = term

        label_to_data = {c:[] for c in clss}# key:标签->item:数据
        for data in self.train_data:
            label_to_data[data[1]].append(data)
        self.all_data = {cid:[] for cid in cids}# key:cid->item:数据
        for c in cids:
            for cl in term_dict[c]:
                self.all_data[c]+=(label_to_data[cl])

    def split_non_iid2(self,num_cls, cls_per_c=2):

        assert(self.train_len!=0 and self.test_len!=0)
        len_per_client_train = int(self.train_len/self.client_num)
        len_per_client_test = int(self.test_len/self.client_num)
        train_dict = dict()
        label_to_data,term_dict = self.get_term_dict(num_cls,cls_per_c)
        for cid in range(self.client_num):
            seg = cid*len_per_client_train
            slicer = slice(seg, seg+(len_per_client_train-1), 1)
            train_dict[cid] = [self.train_data[i] for i in range(*slicer.indices(len(self.train_data)))]

        self.all_data = {}
        for cid in range(self.client_num):
            self.all_data[cid] = train_dict[cid]
            for c in term_dict[cid]:
                self.all_data[cid] += label_to_data[c]
        del train_dict
        del test_dict

    def get_data(self):
        return self.all_data,self.test_data