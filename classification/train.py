import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入 tqdm

# 初始化存储训练过程指标的列表
train_acc_history = []
train_loss_history = []
test_acc_history = []
test_loss_history = []
lr_history = []


# 参数设置
batch_size = 50
epochs = 20
learning_rate = 0.01
warmup_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强与加载
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入层以支持单通道输入
model.fc = nn.Linear(512, 10)  # 修改全连接层以适应MNIST的10个类别
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Warmup学习率调度器
def warmup_lr_scheduler(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_scheduler)

# 训练与评估函数
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    train_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)# 默认 reduction='mean' 计算当期批次样本的平均损失
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        train_bar.set_postfix(loss=loss.item())

    return total_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        test_bar = tqdm(loader, desc="Testing", leave=False)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            test_bar.set_postfix(loss=loss.item())

    return total_loss / total, correct / total

# 训练循环
best_acc = 0.0
best_epoch = 0
save_path = './best_model.pth'

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    # 保存当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    scheduler.step()
    

    # 记录历史数据
    train_acc_history.append(train_acc)
    train_loss_history.append(train_loss)
    test_acc_history.append(test_acc)
    test_loss_history.append(test_loss)


    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 保存最优模型
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch + 1
        torch.save({'epoch': best_epoch, 'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 'accuracy': best_acc}, save_path)

print(f"Best Accuracy: {best_acc:.4f} at Epoch {best_epoch}")

# 可视化训练过程
plt.figure(figsize=(10, 6))

# 绘制训练和测试损失在同一张图上
plt.plot(train_loss_history, label='Train Loss', marker='o')
plt.plot(test_loss_history, label='Test Loss', marker='o')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# 绘制训练和测试准确率
plt.figure(figsize=(10, 6))
plt.plot(train_acc_history, label='Train Accuracy', marker='o')
plt.plot(test_acc_history, label='Test Accuracy', marker='o')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# 绘制学习率曲线
plt.figure(figsize=(10, 6))
plt.plot(lr_history, label='Learning Rate', marker='o')
plt.title('Learning Rate vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid()
plt.show()