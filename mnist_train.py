# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F #加载nn中的功能函数
import torch.optim as optim #加载优化器有关包
import torch.utils.data as Data
from torchvision import datasets,transforms #加载计算机视觉有关包
from torch.autograd import Variable 
import os
import numpy as np

BATCH_SIZE = 64
log_dir = './model/mnist.pth'

#加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor
train_dataset = datasets.MNIST(root='~/data/',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='~/data/',train=False,transform=transforms.ToTensor())

#加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)

#定义网络模型亦即Net 这里定义一个简单的全连接层784->10
class Model(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.avgpool1=nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 32, 5, bias=False)
        self.relu2 = nn.ReLU()
        self.avgpool2=nn.AvgPool2d(2)
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.avgpool2(out)
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out

model = Model().cuda(2) #实例化全连接层
loss = nn.CrossEntropyLoss() #损失函数选择，交叉熵函数
optimizer = optim.SGD(model.parameters(),lr = 0.01)
num_epochs = 100

losses = [] 
acces = []
eval_losses = []
eval_acces = []

if os.path.exists(log_dir):
    checkpoint = torch.load(log_dir, map_location='cuda:2')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("--------------成功加载模型--------------")
else:
    print("--------------训练全新模型--------------")

for echo in range(num_epochs):
    train_loss = 0   #定义训练损失
    train_acc = 0    #定义训练准确度    model.train()    #将网络转化为训练模式
    for i,(X,label) in enumerate(train_loader):     #使用枚举函数遍历train_loader   
        X = Variable(X).cuda(2)          #包装tensor用于自动求梯度
        label = Variable(label).cuda(2)
        out = model(X)           #正向传播
        lossvalue = loss(out,label)         #求损失值
        optimizer.zero_grad()       #优化器梯度归零
        lossvalue.backward()    #反向转播，刷新梯度值
        optimizer.step()        #优化器运行一步，注意optimizer搜集的是model的参数
        
        #计算损失
        train_loss += float(lossvalue)      
        #计算精确度
        _,pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print("echo:"+' ' +str(echo))
    print("lose:" + ' ' + str(train_loss / len(train_loader)))
    print("accuracy:" + ' '+str(train_acc / len(train_loader)))
    eval_loss = 0
    eval_acc = 0
    model.eval() #模型转化为评估模式
    for X,label in test_loader:
        X = Variable(X).cuda(2)
        label = Variable(label).cuda(2)
        testout = model(X)
        testloss = loss(testout,label)
        eval_loss += float(testloss)

        _,pred = testout.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print("testlose: " + str(eval_loss/len(test_loader)))
    print("testaccuracy:" + str(eval_acc/len(test_loader)) + '\n')
    state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict()}
    torch.save(state, log_dir)