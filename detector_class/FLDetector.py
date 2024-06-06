from detector_class.Base_Detector import BASE_DETECTOR
from utils.defence import parameters_dict_to_vector_flt, lbfgs_torch,get_update2,fld_distance,detection1,detection
import numpy as np
import torch

class FLDetector(BASE_DETECTOR):
    def __init__(self) -> None:
        self.weight_record = []
        self.update_record = []
        self.w_glob = None
        self.last_weight = None
        self.last_update = None
        self.old_update_list = None
        self.N = 50
        self.malicious_score = torch.zeros((1, 100))
        pass

    def detect(self,cs_curt,w_glob,iter,tau_client=None,time_client=None):
        # 检测函数
        self.w_glob = w_glob
        local_update_list = [-1*parameters_dict_to_vector_flt(c.model.state_dict()).cpu() for c in cs_curt]
        weight = parameters_dict_to_vector_flt(self.w_glob)
        if iter > self.N+1:
            print("w_record:",self.weight_record)
            print("upd_record:",self.update_record)
            print("w_dif:",weight-self.last_weight)
            hvp = lbfgs_torch(self.weight_record,self.update_record,weight-self.last_weight)
            distance = fld_distance(self.old_update_list,local_update_list,hvp)
            distance = distance.view(1,-1)
            print("distance:",distance)

            self.malicious_score = distance
            # 预测恶意客户端
            label = detection(self.malicious_score.numpy(),3)

            # 排除恶意客户端
            benign_c = []
            malicious_c = []
            for idx in range(len(cs_curt)):
                if(label[idx] ==0):
                    malicious_c.append(cs_curt[idx])
                    continue
                benign_c.append(cs_curt[idx])

            return benign_c,malicious_c

        else:
            hvp = None
        # 前面必须要有聚合，然后聚合之后的权重为new_w_glob
        self.old_update_list = local_update_list
        return cs_curt,None

    def record(self,new_w_glob,iter):
        # 检测后记录历史信息
        # new_w_glob为聚合后的权重
        update = get_update2(self.w_glob, new_w_glob)
        update = parameters_dict_to_vector_flt(update)
        weight = parameters_dict_to_vector_flt(self.w_glob)
        if iter>0:
            self.weight_record.append(weight.cpu()-self.last_weight.cpu())
            self.update_record.append(update.cpu()-self.last_update.cpu())
        if iter>self.N:
            del self.weight_record[0]
            del self.update_record[0]

        # 记录
        self.last_weight = weight
        self.last_update = update