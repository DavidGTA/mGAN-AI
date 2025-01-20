import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from data.utils import get_mnist_data
from models.federated_model import FederatedModel
import logging
import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager

# # 查看所有字体路径
# font_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# print(font_paths)
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def visualize_predictions(model_path, num_samples=10):
    """
    可视化模型在测试集上的预测结果

    Args:
        model_path: 模型参数文件路径
        num_samples: 要显示的样本数量
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = FederatedModel()
    model.model.load_state_dict(torch.load(model_path, map_location=device))
    model.model.eval()

    # 获取测试数据
    _, test_dataset = get_mnist_data()

    # Set the font to one that supports Chinese characters (for Windows example)
    font_path = 'C:\\Windows\\Fonts\\msyh.ttc'  # Change path for your system
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

    # Create a clean style using seaborn
    sns.set(style='whitegrid')  # Use seaborn directly for a clean style

    # Create plot
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    fig.suptitle('模型预测结果可视化', fontsize=16)

    # Randomly select samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image and label
            image, label = test_dataset[idx]
            image = image.to(device)

            # Get model prediction
            _, _, class_output = model.model(image.unsqueeze(0))
            predicted = torch.argmax(class_output, dim=1).item()

            # Display original image
            img_display = image.cpu().squeeze().numpy()
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'真实: {label}', pad=10)

            # Display prediction
            axes[1, i].imshow(img_display, cmap='gray')
            axes[1, i].axis('off')
            color = 'green' if predicted == label else 'red'
            axes[1, i].set_title(f'预测: {predicted}', color=color, pad=10)

    # Adjust layout and save result
    plt.tight_layout()
    plt.savefig('experiments/results/prediction_visualization.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    logging.info(f"可视化结果已保存到 'experiments/results/prediction_visualization.png'")


def evaluate_model(model_path, num_test_samples=1000):
    """
    评估模型在测试集上的性能
    
    Args:
        model_path: 模型参数文件路径
        num_test_samples: 要测试的样本数量
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = FederatedModel()
    model.model.load_state_dict(torch.load(model_path, map_location=device))
    model.model.eval()

    # 获取测试数据
    _, test_dataset = get_mnist_data()

    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for i in range(min(num_test_samples, len(test_dataset))):
            image, label = test_dataset[i]
            image = image.to(device)

            # 获取模型预测
            _, _, class_output = model.model(image.unsqueeze(0))
            predicted = torch.argmax(class_output, dim=1).item()

            # 统计总体准确率
            total += 1
            correct += (predicted == label)

            # 统计每个类别的准确率
            class_total[label] += 1
            class_correct[label] += (predicted == label)

    # 打印总体准确率
    accuracy = 100 * correct / total
    logging.info(f'总体准确率: {accuracy:.2f}%')

    # 打印每个类别的准确率
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            logging.info(f'类别 {i} 的准确率: {class_accuracy:.2f}%')


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/evaluation.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # 指定模型路径
    model_path = 'models/checkpoints/federated_model_round_15.pth'  # 根据实际保存的模型路径修改

    # 可视化预测结果
    # visualize_predictions(model_path, num_samples=10)

    # 评估模型性能
    evaluate_model(model_path, num_test_samples=10000)

