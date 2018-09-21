#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function, division
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data
import os
#from torchsummary import summary
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])

#使用ImageNet直接加载每一个类的文件
#data_dir = '/home/hq/desktop/ImageNet/data'
data_dir = '/share/users_root/heqiang/ImageNet'
traindir = os.path.join(data_dir, 'train100')
testdir = os.path.join(data_dir, 'test100')
train = datasets.ImageFolder(traindir, transform)
test = datasets.ImageFolder(testdir, transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True, num_workers=4)

alexnet = models.alexnet()

if torch.cuda.device_count()>1 :
    alexnet = nn.DataParallel(alexnet)
alexnet = alexnet.to(device)
#summary(alexnet,(3,224,224),batch_size=-1,device="cpu")
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        labels = labels.long()
        outputs = alexnet(inputs)
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]

        if i % 2000 == 1999:
            print('[%d %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
torch.save(alexnet.state_dict(),'torch-alexnet.pkl')
print('finished training')

correct = 0
total = 0

for data in test_loader:
    images, labels = data
    images, labels = images.to(device),labels.to(device)
    images, labels = images.to(device),labels.to(device)
    outputs = alexnet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 5000 test images: %d %%' % (100 * correct / total))
