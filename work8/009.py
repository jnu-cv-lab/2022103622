import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
torch.manual_seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 任务1：环境准备 ====================
print("\n=== 任务1：环境准备 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# 简单的张量操作测试
x = torch.tensor([1, 2, 3, 4, 5])
print(f"张量操作测试: {x + 10}")

# ==================== 任务2：加载数据集 ====================
print("\n=== 任务2：加载数据集 ===")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
full_train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 划分训练集和验证集 (50000训练, 10000验证)
train_size = 50000
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_dataset, [train_size, val_size]
)

# 创建DataLoader
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 显示8张样本图像
def show_sample_images(dataset, num_images=8):
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i in range(num_images):
        image, label = dataset[i]
        # 反归一化显示
        image_display = image.squeeze().numpy()
        axes[i].imshow(image_display, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.suptitle('MNIST Sample Images', fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150)
    plt.show()
    print("样本图像已保存为 sample_images.png")

show_sample_images(train_dataset)

# ==================== 任务3：定义CNN模型 ====================
print("\n=== 任务3：定义CNN模型 ===")

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 卷积层1: 输入1通道，输出32通道，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化层: 2x2
        
        # 卷积层2: 输入32通道，输出64通道，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入: 64*7*7，输出: 128
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 输出层: 10个类别
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel().to(device)
print("模型结构:")
print(model)

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# ==================== 任务4和5：训练和验证模型 ====================
print("\n=== 任务4和5：训练和验证模型 ===")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录训练和验证指标
num_epochs = 5
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_train_acc = 100 * correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    
    # 验证阶段
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    epoch_val_loss = running_val_loss / len(val_loader)
    epoch_val_acc = 100 * correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% '
          f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')

# ==================== 任务6：测试模型 ====================
print("\n=== 任务6：测试模型 ===")

model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss_avg = test_loss / len(test_loader)
test_accuracy = 100 * correct_test / total_test

print(f"测试集 Loss: {test_loss_avg:.4f}")
print(f"测试集 Accuracy: {test_accuracy:.2f}%")

# 显示8张测试图像及预测结果
def show_test_predictions(dataset, predictions, labels, num_images=8):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        image, true_label = dataset[i]
        pred_label = predictions[i]
        
        # 反归一化显示
        image_display = image.squeeze().numpy()
        axes[i].imshow(image_display, cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        axes[i].axis('off')
    plt.suptitle('Test Images: Predictions vs True Labels', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=150)
    plt.show()
    print("测试图像预测结果已保存为 test_predictions.png")

show_test_predictions(test_dataset, all_predictions, all_labels)

# ==================== 任务7：绘制训练曲线 ====================
print("\n=== 任务7：绘制训练曲线 ===")

# 绘制Loss曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)

# 绘制Accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, 'b-', label='Training Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, 'r-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("训练曲线已保存为 training_curves.png")

# ==================== 进阶任务1：修改网络结构（带Dropout） ====================
print("\n=== 进阶任务1：改进模型（添加Dropout） ===")

class ImprovedCNNModel(nn.Module):
    def __init__(self):
        super(ImprovedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

improved_model = ImprovedCNNModel().to(device)
print("改进模型结构:")
print(improved_model)

# 快速训练改进模型5个epoch进行比较
optimizer_improved = optim.Adam(improved_model.parameters(), lr=0.001)

print("\n训练改进模型...")
for epoch in range(5):
    improved_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_improved.zero_grad()
        outputs = improved_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_improved.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Epoch [{epoch+1}/5] Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {100*correct/total:.2f}%')

# 测试改进模型
improved_model.eval()
correct_improved = 0
total_improved = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = improved_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_improved += labels.size(0)
        correct_improved += (predicted == labels).sum().item()

improved_test_acc = 100 * correct_improved / total_improved
print(f"\n原始模型测试准确率: {test_accuracy:.2f}%")
print(f"改进模型测试准确率: {improved_test_acc:.2f}%")
print(f"准确率提升: {improved_test_acc - test_accuracy:.2f}%")

# ==================== 进阶任务2：比较不同优化器 ====================
print("\n=== 进阶任务2：比较不同优化器 ===")

optimizers_to_test = {
    'SGD': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': optim.Adam(model.parameters(), lr=0.001)
}

results = {}

for opt_name, optimizer_opt in optimizers_to_test.items():
    print(f"\n使用优化器: {opt_name}")
    model_temp = CNNModel().to(device)
    optimizer_opt = optimizers_to_test[opt_name]
    
    for epoch in range(5):
        model_temp.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_opt.zero_grad()
            outputs = model_temp(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_opt.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Epoch [{epoch+1}/5] Train Acc: {100*correct/total:.2f}%')
    
    # 测试
    model_temp.eval()
    correct_test_temp = 0
    total_test_temp = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_temp(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test_temp += labels.size(0)
            correct_test_temp += (predicted == labels).sum().item()
    
    test_acc_temp = 100 * correct_test_temp / total_test_temp
    results[opt_name] = test_acc_temp
    print(f"{opt_name} 测试准确率: {test_acc_temp:.2f}%")

print("\n优化器比较结果:")
print(f"{'Optimizer':<10} {'Learning Rate':<15} {'Test Accuracy':<15}")
print("-" * 40)
print(f"{'SGD':<10} {'0.01':<15} {results['SGD']:.2f}%")
print(f"{'Adam':<10} {'0.001':<15} {results['Adam']:.2f}%")

# ==================== 结果分析 ====================
print("\n=== 结果分析 ===")
print("\n1. 训练Loss是否随着epoch增加而下降？")
print(f"   是的。训练Loss从第1个epoch的{train_losses[0]:.4f}下降到第{num_epochs}个epoch的{train_losses[-1]:.4f}")

print("\n2. 验证Accuracy是否随着训练逐渐提升？")
print(f"   是的。验证Accuracy从第1个epoch的{val_accuracies[0]:.2f}%提升到第{num_epochs}个epoch的{val_accuracies[-1]:.2f}%")

print("\n3. 训练Accuracy和验证Accuracy是否存在明显差距？")
gap = train_accuracies[-1] - val_accuracies[-1]
print(f"   最终训练准确率: {train_accuracies[-1]:.2f}%, 验证准确率: {val_accuracies[-1]:.2f}%, 差距: {gap:.2f}%")
if gap > 5:
    print("   存在明显差距，可能存在过拟合。")
else:
    print("   差距较小，模型泛化能力较好。")

print("\n4. 如果存在明显差距，可能是什么原因？")
print("   - 模型复杂度较高，训练数据量不足")
print("   - 缺少正则化手段（如Dropout、BatchNorm等）")
print("   - 训练epoch过多导致过拟合")

# 计算每个类别的准确率
print("\n5. 哪些数字更容易被错误分类？")
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    accuracy_class = 100 * class_correct[i] / class_total[i]
    print(f'  数字 {i}: {accuracy_class:.2f}%')

print("\n6. MNIST和CIFAR-10哪个更难？为什么？")
print("   CIFAR-10更难。原因：")
print("   - CIFAR-10是彩色图像（3通道），MNIST是灰度图像（1通道）")
print("   - CIFAR-10图像尺寸更大（32x32），包含更多复杂背景")
print("   - CIFAR-10的物体变化更大（车辆、动物等），而MNIST是标准化的手写数字")
print("   - CIFAR-10类内差异更大，类间相似度更高（如猫和狗）")

# 保存模型
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("\n模型已保存为 mnist_cnn_model.pth")

print("\n=== 实验完成 ===")