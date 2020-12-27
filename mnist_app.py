# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F #加载nn中的功能函数

import os
import shutil
import numpy as np
from PIL import Image

BATCH_SIZE = 64
log_dir = './model/mnist.pth'
image_path = "data_new/5.png"

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

if os.path.exists(log_dir):
    checkpoint = torch.load(log_dir, map_location='cuda:2')
    model.load_state_dict(checkpoint['model'])
    print("--------------成功加载模型--------------")
else:
    print("--------------全新模型--------------")

img = Image.open(image_path)
img = img.resize((28, 28), Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))

for i in range(28):
    for j in range(28):
        if img_arr[i][j] < 200:
            img_arr[i][j] = 255
        else:
            img_arr[i][j] = 0

img_arr = img_arr / 255.0
img_arr = img_arr.reshape([1,1,img_arr.shape[0],img_arr.shape[1]])
x_predict = torch.Tensor(img_arr).cuda(2)
result = model(x_predict)

pred = torch.argmax(result, axis=1)
label = pred[0].cpu().numpy()

print("predict = " + str(label))

dir = os.listdir("data_base")
shutil.rmtree("similar_img")
os.mkdir("similar_img")
for file in dir:
    if "_"+str(label) in file:
        shutil.move("data_base/" + file, "similar_img/" + file)
