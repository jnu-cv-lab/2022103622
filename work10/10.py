# 第10次实验：CNN图像分类进阶分析
# 包含优化器对比、学习率对比、卷积核可视化、Feature map可视化、
# 错误分类样本分析和混淆矩阵

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
set_seed()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==================== 数据加载 ====================
# CIFAR-10 类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# 为了加快实验，使用部分训练数据（可选，完整训练需要较长时间）
USE_FULL_DATASET = False  # 设为True使用全部50000张训练图片，False使用10000张
if not USE_FULL_DATASET:
    indices = np.random.choice(len(train_dataset), 10000, replace=False)
    train_dataset = Subset(train_dataset, indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

print(f'训练集大小: {len(train_dataset)}')
print(f'测试集大小: {len(test_dataset)}')

# ==================== CNN 模型定义 ====================
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ==================== 训练函数 ====================
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs=10):
    model = model.to(device)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return history

# 划分验证集
val_size = 1000
train_indices = list(range(len(train_dataset)))[val_size:]
val_indices = list(range(len(train_dataset)))[:val_size]
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)
train_loader_sub = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

# ==================== 任务2：优化器对比 ====================
def compare_optimizers():
    print("\n" + "="*50)
    print("任务2：优化器对比实验")
    print("="*50)
    
    optimizers_config = {
        'SGD': optim.SGD,
        'SGD+Momentum': lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
        'Adam': optim.Adam
    }
    
    histories = {}
    test_accuracies = {}
    
    for opt_name, opt_func in optimizers_config.items():
        print(f"\n--- 训练使用 {opt_name} 优化器 ---")
        model = CIFAR10CNN()
        if opt_name == 'SGD':
            optimizer = opt_func(model.parameters(), lr=0.01)
        elif opt_name == 'SGD+Momentum':
            optimizer = opt_func(model.parameters(), lr=0.01)
        else:
            optimizer = opt_func(model.parameters(), lr=0.001)
        
        history = train_model(model, optimizer, criterion, train_loader_sub, val_loader, epochs=10)
        histories[opt_name] = history
        
        # 测试准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = 100. * correct / total
        test_accuracies[opt_name] = test_acc
        print(f"{opt_name} Test Accuracy: {test_acc:.2f}%")
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for opt_name, history in histories.items():
        axes[0, 0].plot(history['train_loss'], label=opt_name)
        axes[0, 1].plot(history['val_loss'], label=opt_name)
        axes[1, 0].plot(history['train_acc'], label=opt_name)
        axes[1, 1].plot(history['val_acc'], label=opt_name)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150)
    plt.show()
    
    # 打印测试准确率
    print("\n测试准确率对比:")
    for opt_name, acc in test_accuracies.items():
        print(f"  {opt_name}: {acc:.2f}%")
    
    return histories, test_accuracies

# ==================== 任务3：学习率对比 ====================
def compare_learning_rates():
    print("\n" + "="*50)
    print("任务3：学习率对比实验 (使用Adam优化器)")
    print("="*50)
    
    lrs = [0.1, 0.01, 0.001]
    histories = {}
    
    for lr in lrs:
        print(f"\n--- 训练使用 learning rate = {lr} ---")
        model = CIFAR10CNN()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        history = train_model(model, optimizer, criterion, train_loader_sub, val_loader, epochs=10)
        histories[lr] = history
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for lr, history in histories.items():
        axes[0].plot(history['train_loss'], label=f'LR={lr}')
        axes[1].plot(history['val_loss'], label=f'LR={lr}')
    
    axes[0].set_title('Training Loss - Different Learning Rates')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].set_title('Validation Loss - Different Learning Rates')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('lr_comparison.png', dpi=150)
    plt.show()
    
    # 绘制准确率对比
    plt.figure(figsize=(12, 5))
    for lr, history in histories.items():
        plt.plot(history['val_acc'], label=f'LR={lr}')
    plt.title('Validation Accuracy - Different Learning Rates')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('lr_acc_comparison.png', dpi=150)
    plt.show()
    
    return histories

# ==================== 任务4：卷积核可视化 ====================
def visualize_filters(model):
    print("\n" + "="*50)
    print("任务4：第一层卷积核可视化")
    print("="*50)
    
    conv1_weights = model.conv1.weight.data.cpu().numpy()
    # 形状: [32, 3, 3, 3] (输出通道, 输入通道, 高度, 宽度)
    
    # 显示至少8个卷积核（每个核有3个通道，我们取平均或分别显示）
    n_filters = min(8, conv1_weights.shape[0])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(n_filters):
        # 对输入通道取平均，得到单通道的核可视化
        filter_img = conv1_weights[i].mean(axis=0)  # 平均RGB通道
        # 归一化到[0,1]范围
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
        axes[i].imshow(filter_img, cmap='gray')
        axes[i].set_title(f'Filter {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('First Convolutional Layer Filters (averaged over RGB channels)', fontsize=14)
    plt.tight_layout()
    plt.savefig('conv_filters.png', dpi=150)
    plt.show()
    
    print("\n观察分析：")
    print("- 训练后的卷积核呈现出不同的模式：有的像边缘检测器（明暗对比），有的像方向滤波器（水平/垂直条纹），")
    print("  有的呈现斑点状纹理。")
    print("- 卷积核是通过反向传播和梯度下降训练得到的：网络初始化随机权重，前向传播计算损失，")
    print("  反向传播计算损失对每个卷积核权重的梯度，优化器根据梯度更新权重，使损失逐渐减小。")
    print("- 浅层卷积核通常学习到低层次特征（边缘、颜色、纹理），深层卷积核学习到更抽象的特征（形状、物体部件）。")

# ==================== 任务5：Feature Map可视化 ====================
def visualize_feature_maps(model, test_loader):
    print("\n" + "="*50)
    print("任务5：Feature Map可视化")
    print("="*50)
    
    # 获取一张测试图片
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    img = images[0:1].to(device)  # 取第一张图片
    true_label = labels[0].item()
    
    # 注册hook来获取中间层输出
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册hook到第一层卷积层
    handle = model.conv1.register_forward_hook(get_activation('conv1'))
    
    # 前向传播
    output = model(img)
    handle.remove()
    
    # 获取第一层卷积输出 (batch, 32, 32, 32) -> (1, 32, 32, 32)
    feat_maps = activation['conv1'].cpu().squeeze(0).numpy()
    
    # 显示原图
    img_display = img.cpu().squeeze(0).permute(1, 2, 0).numpy()
    # 反归一化
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(img_display)
    plt.title(f'Original Image: {classes[true_label]}')
    plt.axis('off')
    
    # 显示至少8个feature maps
    n_maps = min(8, feat_maps.shape[0])
    for i in range(n_maps):
        plt.subplot(3, 3, i+2)
        fm = feat_maps[i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        plt.imshow(fm, cmap='viridis')
        plt.title(f'Feature Map {i+1}')
        plt.axis('off')
    
    plt.suptitle('First Convolutional Layer Feature Maps', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=150)
    plt.show()
    
    print("\n观察分析：")
    print("- 不同的feature map对图像的不同区域有强响应：有的关注边缘轮廓，有的关注纹理区域，有的关注颜色区域。")
    print("- 例如，一些feature map在物体边界处有高激活值（亮色），另一些在平坦区域激活较低。")
    print("- 不同卷积核提取了不同的图像特征：边缘检测核、方向检测核、纹理检测核等。")
    print("- 这体现了CNN的层次化特征提取能力：浅层提取局部低级特征，深层组合成高级语义特征。")

# ==================== 任务6：错误分类样本分析 ====================
def analyze_misclassifications(model, test_loader):
    print("\n" + "="*50)
    print("任务6：错误分类样本分析")
    print("="*50)
    
    model.eval()
    misclassified = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 找出错误分类的样本
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified.append({
                        'image': images[i].cpu(),
                        'true_label': labels[i].item(),
                        'pred_label': predicted[i].item()
                    })
    
    print(f"总测试样本数: {len(all_labels)}")
    print(f"错误分类样本数: {len(misclassified)}")
    print(f"准确率: {100 * (1 - len(misclassified)/len(all_labels)):.2f}%")
    
    # 分析混淆最多的类别
    from collections import Counter
    confusion_pairs = [(true, pred) for true, pred in zip(all_labels, all_preds) if true != pred]
    pair_counts = Counter(confusion_pairs)
    if pair_counts:
        most_confused = pair_counts.most_common(5)
        print("\n最常混淆的类别对:")
        for (true, pred), count in most_confused:
            print(f"  {classes[true]} -> {classes[pred]}: {count}次")
    
    # 显示至少8张错误分类图片
    n_display = min(8, len(misclassified))
    if n_display > 0:
        fig, axes = plt.subplots(2, 4, figsize=(14, 8))
        axes = axes.flatten()
        
        for i in range(n_display):
            img = misclassified[i]['image']
            # 反归一化
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            
            axes[i].imshow(img)
            axes[i].set_title(f'True: {classes[misclassified[i]["true_label"]]}\nPred: {classes[misclassified[i]["pred_label"]]}')
            axes[i].axis('off')
        
        plt.suptitle('Misclassified Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig('misclassified_samples.png', dpi=150)
        plt.show()
    else:
        print("没有错误分类样本！")
    
    # 分析
    print("\n分析：")
    print("- 哪些类别最容易被混淆？")
    if pair_counts:
        print(f"  从上面统计可以看出，{classes[most_confused[0][0][0]]}和{classes[most_confused[0][0][1]]}最容易混淆。")
    print("- 错误原因可能是什么？")
    print("  (1) 类别本身外观相似（如猫和狗、汽车和卡车、鹿和马）")
    print("  (2) 图像背景干扰、光照变化、遮挡等")
    print("  (3) 数据集类别不平衡")
    print("  (4) 模型容量有限，未能学习到精细区分特征")
    print("- 改进建议：")
    print("  (1) 数据方面：数据增强（旋转、缩放、色彩抖动）、补充难例样本")
    print("  (2) 模型结构：增加深度、使用残差连接、注意力机制")
    print("  (3) 训练方法：使用学习率调度、更长时间训练、标签平滑、集成学习")
    
    return misclassified

# ==================== 任务7：混淆矩阵 ====================
def plot_confusion_matrix(model, test_loader):
    print("\n" + "="*50)
    print("任务7：混淆矩阵")
    print("="*50)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on CIFAR-10 Test Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    print("\n混淆矩阵分析：")
    print("- 对角线元素代表正确分类的样本数量（真正例），值越大说明该类分类效果越好。")
    print("- 非对角线元素代表错误分类的样本数量，即把某类预测成了其他类。")
    
    # 找出混淆最严重的类别对
    np.fill_diagonal(cm, 0)  # 忽略对角线
    max_confusion_idx = np.unravel_index(np.argmax(cm), cm.shape)
    print(f"- 混淆最严重的类别对：{classes[max_confusion_idx[0]]} 被误分为 {classes[max_confusion_idx[1]]}，共 {cm[max_confusion_idx]} 次")
    
    # 各类别准确率
    class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\n各类别准确率：")
    for i, acc in enumerate(class_acc):
        print(f"  {classes[i]}: {acc*100:.2f}%")

# ==================== 主函数 ====================
def main():
    print("\n" + "="*60)
    print("第10次实验：CNN图像分类进阶分析")
    print("="*60)
    
    # 任务1：复用上次CNN模型 - 将使用同一个模型结构
    print("\n任务1：使用CIFAR-10 CNN模型（任务2-7将基于此模型）")
    
    # 任务2：优化器对比
    opt_histories, opt_test_accs = compare_optimizers()
    
    # 任务3：学习率对比
    lr_histories = compare_learning_rates()
    
    # 训练一个最终模型用于可视化任务（使用Adam，lr=0.001）
    print("\n" + "="*50)
    print("训练最终模型用于可视化任务...")
    print("="*50)
    final_model = CIFAR10CNN()
    final_optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    final_history = train_model(final_model, final_optimizer, criterion, 
                                 train_loader_sub, val_loader, epochs=10)
    
    # 任务4：卷积核可视化
    visualize_filters(final_model)
    
    # 任务5：Feature map可视化
    visualize_feature_maps(final_model, test_loader)
    
    # 任务6：错误分类样本分析
    misclassified = analyze_misclassifications(final_model, test_loader)
    
    # 任务7：混淆矩阵
    plot_confusion_matrix(final_model, test_loader)
    
    print("\n" + "="*60)
    print("所有实验任务完成！")
    print("生成的图片文件：")
    print("  - optimizer_comparison.png  (优化器对比)")
    print("  - lr_comparison.png         (学习率对比)")
    print("  - lr_acc_comparison.png     (学习率准确率对比)")
    print("  - conv_filters.png          (卷积核可视化)")
    print("  - feature_maps.png          (特征图可视化)")
    print("  - misclassified_samples.png (错误分类样本)")
    print("  - confusion_matrix.png      (混淆矩阵)")
    print("="*60)

if __name__ == '__main__':
    main()