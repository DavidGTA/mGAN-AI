import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader

from data.utils import get_mnist_data
from models.classifier import mGAN_AITaskNetwork, load_model

class FederatedModel:
    def __init__(self, model=None, learning_rate=0.002):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model else mGAN_AITaskNetwork()
        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

    def update_model(self, updates):
        # 平均更新模型
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                # 获取当前层的所有更新，确保在同一设备上
                param_updates = [update[i].to(self.device) for update in updates]
                # 将所有更新平均后加到模型参数中
                avg_update = sum(param_updates) / len(updates)
                param.data.add_(avg_update)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    def train(self, train_loader, local_epochs=5):
        """
        在本地数据上训练模型多个轮次
        
        Args:
            train_loader: 训练数据加载器
            local_epochs: 本地训练轮数，默认为5轮
        """
        self.model.train()
        final_loss = 0
        
        for epoch in range(local_epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                real_fake_output, identity_output, class_output = self.model(data)
                loss = nn.CrossEntropyLoss()(class_output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            epoch_loss = total_loss / len(train_loader)
            logging.info(f"第 {epoch+1} 轮训练损失: {epoch_loss:.4f}")
            final_loss = epoch_loss  # 记录最后一轮的损失
            
        return final_loss

    def test(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                real_fake_output, identity_output, class_output = self.model(data)
                # 使用 argmax 提取每行最大的索引
                max_indices = torch.argmax(class_output, dim=1)
                total += target.size(0)
                correct += (max_indices == target).sum().item()
        return correct / total

if __name__ == "__main__":
    model_path = 'models/checkpoints/federated_model_round_5.pth'
    model = load_model(model_path)
    federated_model = FederatedModel(model)
    train_dataset, test_dataset = get_mnist_data()
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    test_acc = federated_model.test(test_loader=test_dataloader)
    print(test_acc)
