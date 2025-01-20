import torch
import torch.nn as nn

class mGAN_AITaskNetwork(nn.Module):
    def __init__(self):
        super(mGAN_AITaskNetwork, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # 28x28x1 -> 14x14x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 14x14x32 -> 7x7x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 7x7x64 -> 7x7x128
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 7x7x128 -> 7x7x256
        
        # 全连接层
        self.fc = nn.Linear(7 * 7 * 256, 12544)  # Flatten to 12,544
        
        # 输出层
        self.fc_real_fake = nn.Linear(12544, 1)  # Real/Fake Discrimination
        self.fc_identity = nn.Linear(12544, 1)  # Identity Recognition
        self.fc_class = nn.Linear(12544, 10)  # Classification (10 classes for MNIST)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # 全连接层处理
        x = torch.relu(self.fc(x))
        
        # 输出：Real/Fake Discrimination (sigmoid for binary classification)
        real_fake_output = self.sigmoid(self.fc_real_fake(x))
        
        # 输出：Identity Recognition (sigmoid for binary classification)
        identity_output = self.sigmoid(self.fc_identity(x))
        
        # 输出：Classification (softmax for multi-class classification)
        # class_output = self.softmax(self.fc_class(x))
        class_output = self.fc_class(x)

        return real_fake_output, identity_output, class_output

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mGAN_AITaskNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

# if __name__ == "__main__":
#     model = mGAN_AITaskNetwork()
#     x = torch.randn(1, 1, 28, 28)
#     print(model(x))
