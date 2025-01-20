# mgan_config.py
class MGANConfig:
    def __init__(self):
        # 设置 mGAN-AI 的训练配置
        self.generator_learning_rate = 0.0002
        self.discriminator_learning_rate = 0.0002
        self.momentum = 0.5
        self.lambda_tv = 0.00015  # TV 正则化的 lambda
        self.beta = 1.25  # TV 正则化的 beta
