
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from itertools import product
# 设置数据路径
data_path = '/hy-tmp/datasets/CUB_200_2011/images'
batch_size = 256


# 定义数据预处理和增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
full_data = ImageFolder(root=data_path, transform=transform)

# 设置数据集划分比例
train_size = int(0.6 * len(full_data))
val_size = int(0.2 * len(full_data))
test_size = len(full_data) - train_size - val_size

# 划分数据集
train_data, val_data, test_data = random_split(full_data, [train_size, val_size, test_size])

# 定义 DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

def train_val(pretrained=1, lr=0.0005, num_epochs=100, dropout=0.5, weight_decay=0.0001, log='runs3'):
    # 加载预训练的 ResNet-18 模型
    if pretrained:
        model = models.resnet18(weights = "IMAGENET1K_V1", )
    else:
        model = models.resnet18(weights = None)

    # 修改输出层大小为200
    num_ftrs = model.fc.in_features
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 200),
        nn.Dropout(dropout),
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # 定义学习率调度器
    scheduler = StepLR(optimizer, step_size=7, gamma=0.5)
    # print(model)

        # 创建TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=f'{log}/resnet18_pretrained{pretrained}_lr{lr}_epoch{num_epochs}_dropout{dropout}_weight_decay{weight_decay}')

    model = model.cuda()
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # for param in model.parameters():
            #     loss += weight_decay * torch.norm(param, 2)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() 
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss = train_loss / len(train_loader.batch_sampler)
        train_accuracy = train_correct / train_total
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    
        # 验证模型
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # for param in model.parameters():
                #     loss += weight_decay * torch.norm(param, 2)
                val_loss += loss.item() * inputs.size(0)  #不一定是数量完整的batch
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss / len(val_data)
        val_accuracy = val_correct / val_total
        # print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # 更新学习率
        scheduler.step()
    writer.close()

if __name__ == '__main__':
    num_epochs_list = [30, 50, 100]
    dropout_list = [0, 0.1, 0.2, 0.3]
    lr_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    pretrained_list = [1, 0]    
    # lr_decay_list = [True, False]
    for idx, (pretrained, lr, num_epochs, dropout) in enumerate(product(pretrained_list, lr_list, num_epochs_list, dropout_list)):
        train_val(pretrained=pretrained, lr=lr, num_epochs=num_epochs, dropout=dropout, weight_decay=0.0001, log='runs3')
