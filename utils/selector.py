import random
class selector:
    def __init__(self,clients, ini_prior):
        self.prior = ini_prior# 初始权重为数据集的大小
        delays = []
        power = []
        contri = []

        self.clients = clients
        pass

    def random_select(self, K=10):
        # 随机挑选K个客户端
        return random.choices(self.clients,self.prior,K)

    def update_prior(self, cid, contri):
        # 更新概率表中的概率
        # cid->用户id, contri->贡献度

        pass
    pass