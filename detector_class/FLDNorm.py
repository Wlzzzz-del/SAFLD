from detector_class.Base_Detector import BASE_DETECTOR
from detector_class.FLDetector import FLDetector
from utils.defence import parameters_dict_to_vector_flt, lbfgs_torch,get_update2,fld_distance,detection1,detection,count_parameters_dict_norm
import numpy as np

class FLDNorm(FLDetector):
    def __init__(self) -> None:
        super().__init__()
        pass

    def detect(self,cs_curt,w_glob,iter,tau_client=None,time_client=None):
        # 检测函数
        self.w_glob = w_glob
        norm_list = [count_parameters_dict_norm(c.model.state_dict()).cpu() for c in cs_curt]
        if iter > self.N+1:
            self.malicious_score = norm_list
            print(norm_list)
            # 预测恶意客户端
            label = detection(np.array(self.malicious_score),3)

            # 排除恶意客户端
            benign_c = []
            malicious_c = []
            for idx in range(len(cs_curt)):
                if(label[idx] ==0):
                    malicious_c.append(cs_curt[idx])
                    continue
                benign_c.append(cs_curt[idx])

            return benign_c,malicious_c

        # 前面必须要有聚合，然后聚合之后的权重为new_w_glob
        return cs_curt,None
    pass