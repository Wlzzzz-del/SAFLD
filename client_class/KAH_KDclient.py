from client_class.KAH_client import KAH_client

class KAH_KD_client(KAH_client):
    def __init__(self, cid, model, lr, steps, data, batch_size, dev, test_data, ratio):
        super().__init__(cid, model, lr, steps, data, batch_size, dev, test_data, ratio)