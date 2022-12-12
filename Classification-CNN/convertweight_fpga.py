"""class CNN(nn.Module):
def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(1024 ,1024)
    self.fc2 = nn.Linear(1024, 33)
    self.bn1 = nn.BatchNorm2d(16)
    self.bn2 = nn.BatchNorm2d(32)
    self.bn3 = nn.BatchNorm2d(64)
    self.bn4 = nn.BatchNorm2d(128)
    self.bn5 = nn.BatchNorm2d(256)
    self.bn6 = nn.BatchNorm2d(512)
    self.bn7 = nn.BatchNorm2d(1024)
def forward(self, x):
    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = self.pool(F.relu(self.bn3(self.conv3(x))))
    x = self.pool(F.relu(self.bn4(self.conv4(x))))
    x = self.pool(F.relu(self.bn5(self.conv5(x))))
    x = self.pool(F.relu(self.bn6(self.conv6(x))))
    x = F.relu(self.bn7(self.conv7(x)))
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
The model has following architecture, save weight and biases of each conv layer into a binary file. 
For the batchnorm, save mean and variance into binary file
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
from training_cnn.model import CNN
model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()
#saving weights and biases of conv layers
fpgaw_path = "fpga_weights"
for i in range(1,8):
    conv = getattr(model, 'conv'+str(i))
    weight = conv.weight.data.cpu().numpy()
    bias = conv.bias.data.cpu().numpy()
    weight.tofile(osp.join(fpgaw_path, 'conv'+str(i)+'_weight.bin'))
    bias.tofile(osp.join(fpgaw_path, 'conv'+str(i)+'_bias.bin'))
#saving mean and variance of batchnorm layers
for i in range(1,8):
    bn = getattr(model, 'bn'+str(i))
    mean = bn.running_mean.data.cpu().numpy()
    var = bn.running_var.data.cpu().numpy()
    mean.tofile(osp.join(fpgaw_path, 'bn'+str(i)+'_mean.bin'))
    var.tofile(osp.join(fpgaw_path, 'bn'+str(i)+'_var.bin'))
