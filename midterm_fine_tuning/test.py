import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

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
best_model_path = 'saved_models/resnet18_pretrained1_lr1e-05_epoch400_dropout0.5_adam_acc0.810008.pth'

# data

# 定义数据预处理和增强
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
val_data = MyDataset(val_data, val_transforms)
print(f"训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")
# 定义 DataLoader
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)


# model
model = models.resnet18(weights = None)

# 修改输出层大小为200
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 200),
    nn.Dropout(dropout),
)
model.load_state_dict(torch.load(best_model_path))

model = model.cuda()
criterion = nn.CrossEntropyLoss()
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
print(f"Test Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.6f}")

