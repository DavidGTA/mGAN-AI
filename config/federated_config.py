# federated_config.py
class FederatedConfig:
    def __init__(self):
        # 设置联邦学习的客户端数量和其他参数
        self.num_clients = 10
        self.num_rounds = 15
        self.learning_rate = 0.002
        self.batch_size = 128
        self.model_save_path = './models/checkpoints/federated_model_final.pth'
