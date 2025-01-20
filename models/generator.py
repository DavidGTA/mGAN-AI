import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 将类别和身份标签通过嵌入层映射到100维
        self.embedding_cat = nn.Embedding(10, 100)  # 类别标签：10个类别，每个标签映射为100维
        self.embedding_id = nn.Embedding(2, 100)    # 身份标签：2种身份，每个标签映射为100维

        # 全连接层，将输入的噪声、类别和身份标签拼接后变为3×3×384的特征图
        self.fc1 = nn.Linear(100 + 100 + 100, 3 * 3 * 384)  # 输入: 噪声100 + 类别100 + 身份100

        # 反卷积层
        self.deconv1 = nn.ConvTranspose2d(384, 192, kernel_size=5, stride=2, padding=1)  # 7x7x384 -> 7x7x192
        self.deconv2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)  # 7x7x192 -> 14x14x96
        self.deconv3 = nn.ConvTranspose2d(96, 1, kernel_size=4, stride=2, padding=1)    # 14x14x96 -> 28x28x1

    def forward(self, z, category_label, identity_label):
        # 类别标签和身份标签通过嵌入层进行映射
        category_embed = self.embedding_cat(category_label)  # 将类别标签映射为100维
        identity_embed = self.embedding_id(identity_label)  # 将身份标签映射为100维
        
        # 将噪声、类别标签和身份标签拼接在一起
        x = torch.cat([z, category_embed, identity_embed], dim=1)  # 拼接：噪声 + 类别 + 身份

        # 通过全连接层将拼接后的输入映射为3×3×384的特征图
        x = self.fc1(x)
        x = x.view(x.size(0), 384, 3, 3)  # 将输出reshape为(batch_size, 384, 3, 3)
        
        # 反卷积操作
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))  # 使用tanh激活函数生成最终图像
        return x

# if __name__ == "__main__":
#     model = Generator()
#     z = torch.randn(1, 100)  # Random noise
#     category_labels = torch.randint(0, 10, (1,))  # Random category label (0 to 9 for 10 classes)
#     identity_labels = torch.randint(0, 2, (1,))  # Random identity label (0 or 1)
#     print(model(z, category_labels, identity_labels).shape)  # Print the output shape
