import ghostnet
import torch
import matplotlib.pyplot as plt

import torchvision
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 32 # 大概需要2G的显存
EPOCHS = 20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

dataset =torchvision.datasets.ImageFolder(root='G:/LJH/DATASETS/flower_photos',transform=train_transform)

train_dataset, valid_dataset = train_test_split(dataset,test_size=0.2, random_state=0)
print(len(train_dataset))
print(len(valid_dataset))
train_loader =DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True,num_workers=0)#Batch Size定义：一次训练所选取的样本数。 Batch Size的大小影响模型的优化程度和速度。
valid_loader =DataLoader(valid_dataset,batch_size=BATCH_SIZE, shuffle=True,num_workers=0)#Batch Size定义：一次训练所选取的样本数。 Batch Size的大小影响模型的优化程度和速度。


model=ghostnet.ghostnet()
model.load_state_dict(torch.load('weigths/state_dict_93.98.pth'))
model.classifier=nn.Linear(1280,5)
model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()
avg_loss=[]
avg_acc=[]

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    avg_loss.append(test_loss)
    avg_acc.append(100. * correct / len(test_loader.dataset))

for epoch in range(1, 81):
    train(model,  DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, valid_loader)
torch.save(model, 'weigths/GhostNetFlowermodel.pth')
epoch=range(1,81)
#plt.plot(epoch, avg_loss, color='red')
plt.plot(epoch, avg_acc, label='acc changes',color='blue')
for a,b in zip(epoch,avg_acc):
    plt.text(a, b+0.05, '%.1f' % b, ha='center', va= 'bottom',fontsize=7)
plt.xlabel('epochs')# 横坐标描述
plt.ylabel('accuracy')# 纵坐标描述
plt.legend()#显示图例
plt.show()