import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import time

# 设置数据路径
batch_size = 32
dropout = 0.5
num_epochs = 500
lr = 0.0001
pretrained = 0
optm = "adam"
lrsch = 1

# 数据路径
data_path = '/hy-tmp/datasets/CUB_200_2011/images'
split_file = '/hy-tmp/datasets/CUB_200_2011/train_test_split.txt'

# 定义数据预处理和增强

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),        # 随机裁剪并调整到指定大小
    transforms.RandomHorizontalFlip(),        # 随机水平翻转，增加左右方向上的泛化
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
    transforms.RandomRotation(20),            # 随机旋转±20度
    transforms.ToTensor(),                    # 转换成Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
])
val_transforms = transforms.Compose([
    transforms.Resize(256),                   
    transforms.CenterCrop(224),               
    transforms.ToTensor(),                    
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

# 加载数据集
full_data = ImageFolder(root=data_path, transform=None)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.dataset)


# 设置数据集划分比例
train_size = int(0.8 * len(full_data))
val_size = len(full_data) - train_size
# test_size = len(full_data) - train_size - val_size

# 划分数据集
# train_data, val_data, test_data = random_split(full_data, [train_size, val_size, test_size])
train_data, val_data = random_split(full_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_data = MyDataset(train_data, train_transforms)
val_data = MyDataset(val_data, val_transforms)
print(f"训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")
# 定义 DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)


# 加载预训练的 ResNet-18 模型
if pretrained:
    model = models.resnet18(weights = "IMAGENET1K_V1", )
else:
    model = models.resnet18(weights = None)


# 修改输出层大小为200
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 200),
    nn.Dropout(dropout),
)

for param in model.parameters():
    param.requires_grad = True
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
if optm == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optm == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
print(model)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# 创建TensorBoard SummaryWriter
suffix_num = timestamp_str = str(int(time.time()))[-6:]
writer = SummaryWriter(log_dir=f'runs/resnet18_pretrained{pretrained}_lr{lr}_epoch{num_epochs}_dropout{dropout}_{optm}_lrsch{lrsch}_{suffix_num}')

model = model.cuda()

# 训练模型
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) 
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
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)  #不一定是数量完整的batch
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(val_data)
    test_accuracy = test_correct / test_total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.6f}, Test Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.6f}")
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)

    if lrsch:
        scheduler.step()

    # 保存最好的模型
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        if (pretrained and best_acc > 0.76) or  (not pretrained and best_acc > 0.5):
            torch.save(model.state_dict(), f'saved_models/resnet18_pretrained{pretrained}_lr{lr}_epoch{num_epochs}_dropout{dropout}_{optm}_acc{best_acc:.6f}.pth')
        
writer.close()