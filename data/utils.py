import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# 下载并处理 MNIST 数据集
def get_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform)

    return train_dataset, test_dataset

# 划分客户端数据
def split_data_for_clients(dataset, num_clients=10):
    # 获取数据集的所有类别
    labels = dataset.targets.clone().detach()  # 替代 torch.tensor(dataset.targets)
    num_classes = len(dataset.classes)
    
    # 为每个类别创建索引列表
    class_indices = [torch.where(labels == i)[0] for i in range(num_classes)]
    
    # 打乱每个类别的索引
    shuffled_class_indices = []
    for indices in class_indices:
        shuffled_indices = indices[torch.randperm(len(indices))]
        shuffled_class_indices.append(shuffled_indices)  # 保存打乱后的索引
    
    # 将每个类别的所有样本打乱合并
    all_shuffled_indices = torch.cat(shuffled_class_indices)
    all_shuffled_indices = all_shuffled_indices[torch.randperm(len(all_shuffled_indices))]

    # 计算每个客户端应获得的样本数
    samples_per_client = len(all_shuffled_indices) // num_clients
    
    # 为每个客户端分配数据
    client_data = []
    for client_id in range(num_clients):
        client_indices = all_shuffled_indices[client_id * samples_per_client:(client_id + 1) * samples_per_client]
        client_data.append(Subset(dataset, client_indices))
    
    return client_data

if __name__ == "__main__":
    train_dataset, test_dataset = get_mnist_data()
    client_data = split_data_for_clients(train_dataset, 10)
    
    # 验证每个客户端的数据分布
    print("数据集总大小:", len(train_dataset))
    for i, client_dataset in enumerate(client_data):
        labels = [train_dataset.targets[idx].item() for idx in client_dataset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n客户端 {i + 1} 的数据大小: {len(client_dataset)}")
        print(f"客户端 {i + 1} 的类别分布:")
        for label, count in zip(unique, counts):
            print(f"类别 {label}: {count} 个样本")
        print("---")


#     client_data_loader = DataLoader(client_data[0], batch_size=64, shuffle=True)

#     for i, (images, labels) in enumerate(client_data_loader):
#         print(images.shape)
#         print(labels.shape)
#         input()
