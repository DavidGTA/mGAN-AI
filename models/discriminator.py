import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 1)  # 真实/伪区分

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 1024)  # 扁平化
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
