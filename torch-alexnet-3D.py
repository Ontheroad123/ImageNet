#!/usr/bin/env python
# -*- coding:utf-8 -*-

from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageFilter
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianBlur(ImageFilter.Filter):
    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:

            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


def default_loader(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype="float32")
    img_tensor = transforms.ToTensor()(arr)
    # 高斯模糊处理,并且将图片转为channel first
    image1 = img.filter(GaussianBlur(radius=1))
    arr1 = np.asarray(image1, dtype="float32")
    image1_tensor = transforms.ToTensor()(arr1)

    image2 = img.filter(GaussianBlur(radius=3))
    arr2 = np.asarray(image2, dtype="float32")
    image2_tensor = transforms.ToTensor()(arr2)

    image3 = img.filter(GaussianBlur(radius=5))
    arr3 = np.asarray(image3, dtype="float32")
    image3_tensor = transforms.ToTensor()(arr3)
    # 合成四维矩阵
    new = np.empty((3, 224, 224, 4), dtype="float32")
    arr = img_tensor.numpy()
    arr1 = image1_tensor.numpy()
    arr2 = image2_tensor.numpy()
    arr3 = image3_tensor.numpy()
    new[:, :, :, 0] = arr
    new[:, :, :, 1] = arr1
    new[:, :, :, 2] = arr2
    new[:, :, :, 3] = arr3

    return new

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)

        return img, label

    def __len__(self):

        return len(self.imgs)


train_data=MyDataset(txt='train100.txt', transform=transforms.ToTensor())
test_data=MyDataset(txt='test100.txt', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32,shuffle=True)


class AlexNet(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(11,11,3), stride=(4,4,1), padding=(2,2,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3,3,1), stride=(2,2,1)),
            nn.Conv3d(64, 192, kernel_size=(5,5,3), padding=(2,2,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3,3,1), stride=(2,2,1)),
            nn.Conv3d(192, 384, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3,3,1), stride=(2,2,1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6* 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6 * 4)
        x = self.classifier(x)
        return x
alexnet = AlexNet()
if torch.cuda.device_count()>1:
    alexnet = nn.DataParallel(alexnet)
alexnet = alexnet.to(device)
#summary(alexnet,(3,224,224,4))


cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    running_loss = 0.0
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        labels = labels.long()
        outputs = alexnet(inputs)
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if i % 1000 == 999:
            print('[%d %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
torch.save(alexnet.state_dict(),'torch-alexnet-3D.pkl')
print('finished training')

correct = 0
total = 0

for data in test_loader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = alexnet(images)
    _, predicted = torch.max(outputs.data, 1)
    #print(predicted.shape)
    
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 5000 test images: %d %%' % (100 * correct / total))
