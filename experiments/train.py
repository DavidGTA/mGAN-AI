import random

import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from config.federated_config import FederatedConfig
from models.federated_model import FederatedModel
from models.classifier import load_model
from data.utils import get_mnist_data, split_data_for_clients

# 配置日志, 编码为utf-8

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/federated_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def train_federated_model():
    # 加载配置
    config = FederatedConfig()
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 初始化联邦模型
    # federated_model = FederatedModel(learning_rate=config.learning_rate)
    # 加载模型
    model_path = 'models/checkpoints/federated_model_round_10.pth'
    federated_model = FederatedModel(load_model(model_path), learning_rate=config.learning_rate)
    
    # 获取并分割数据
    train_dataset, test_dataset = get_mnist_data()
    client_datasets = split_data_for_clients(train_dataset, config.num_clients)

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 为每个客户端创建数据加载器
    client_dataloaders = [
        DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        for dataset in client_datasets
    ]
    
    # 联邦学习主循环
    for round in range(10, config.num_rounds):
        logging.info(f"开始第 {round + 1} 轮训练")
        
        # 收集所有客户端的模型更新
        client_updates = []
        
        # 每个客户端进行本地训练
        for client_id, dataloader in enumerate(client_dataloaders):
            # 克隆全局模型进行本地训练
            client_model = FederatedModel()
            client_model.model.load_state_dict(federated_model.model.state_dict())
            
            # 本地训练
            loss = client_model.train(dataloader, local_epochs=random.randint(5,10))
            logging.info(f"客户端 {client_id + 1} 训练损失: {loss:.4f}")
            
            # 计算模型更新（参数差异）
            updates = []
            for global_param, client_param in zip(
                federated_model.model.parameters(),
                client_model.model.parameters()
            ):
                updates.append(client_param.data - global_param.data)
            
            client_updates.append(updates)
        
        # 更新全局模型
        federated_model.update_model(client_updates)
        
        # 保存模型检查点
        if (round + 1) % 5 == 0:
            torch.save(
                federated_model.model.state_dict(),
                f"models/checkpoints/federated_model_round_{round+1}.pth"
            )
            logging.info(f"模型已保存到 models/checkpoints/federated_model_round_{round+1}.pth")

        if (round + 1) % 5 == 0:
            # 测试模型
            test_acc = federated_model.test(test_dataloader)
            logging.info(f"测试准确率: {test_acc:.4f}")
    
    # 保存最终模型
    torch.save(
        federated_model.model.state_dict(),
        config.model_save_path
    )
    logging.info("联邦学习训练完成")

if __name__ == "__main__":
    train_federated_model()