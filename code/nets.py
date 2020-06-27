import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)



    def forward(self, obs):
        '''calculate cumulative return of trajectory'''
        x = obs.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 784)
        #x = x.view(-1, 1936)
        x = F.relu(self.fc1(x))
        #r = torch.tanh(self.fc2(x)) #clip reward?
        x = self.fc2(x)
        return x


class MediumNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
        self.fc1 = nn.Linear(6272, 1024)
        #self.fc1 = nn.Linear(1936,64)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256,1)



    def forward(self, obs):
        '''feed forward through network to get logits for binary reward classification'''

        x = obs.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 6272)
        #x = x.view(-1, 1936)
        x = F.relu(self.fc1(x))
        #r = torch.tanh(self.fc2(x)) #clip reward?
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Machado ICLR 2018 paper net
class BiggerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 64, 5, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(12544, 2048)
        #self.fc1 = nn.Linear(1936,64)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512,1)



    def forward(self, obs):
        '''feed forward through network to get logits for binary reward classification'''

        x = obs.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 12544)
        #x = x.view(-1, 1936)
        x = F.relu(self.fc1(x))
        #r = torch.tanh(self.fc2(x)) #clip reward?
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
