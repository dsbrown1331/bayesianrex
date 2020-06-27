import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        for x in traj:
            x = x.permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            #x = x.view(-1, 1936)
            x = F.leaky_relu(self.fc1(x))
            #r = torch.tanh(self.fc2(x)) #clip reward?
            r = self.fc2(x)
            #r = torch.sigmoid(r) #TODO: try without this
            sum_rewards += r
        ##    y = self.scalar(torch.ones(1))
        ##    sum_rewards += y
        #print(sum_rewards)
        return sum_rewards

    def predict_reward(self, obs):
        '''calculate cumulative return of trajectory'''
        x = obs
        x = x.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)
        #x = x.view(-1, 1936)
        x = F.leaky_relu(self.fc1(x))
        #r = torch.tanh(self.fc2(x)) #clip reward?
        r = self.fc2(x)
        r = torch.sigmoid(r) #TODO: try without this
        return r




    def state_features(self, traj):

        with torch.no_grad():
            accum = torch.zeros(1,64).float().to(self.device)
            for x in traj:
                x = x.permute(0,3,1,2) #get into NCHW format
                #compute forward pass of reward network
                x = F.leaky_relu(self.conv1(x))
                x = F.leaky_relu(self.conv2(x))
                x = F.leaky_relu(self.conv3(x))
                x = F.leaky_relu(self.conv4(x))
                x = x.view(-1, 784)
                #x = x.view(-1, 1936)
                x = F.leaky_relu(self.fc1(x))
                #print(x.size())
                accum.add_(x)
                #print(accum)
        return accum

    def state_feature(self, obs):
        with torch.no_grad():
            x = obs.permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            #x = x.view(-1, 1936)
            x = F.leaky_relu(self.fc1(x))
            #print(x.size())
        return x

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        #print([self.cum_return(traj_i), self.cum_return(traj_j)])
        cum_r_i = self.cum_return(traj_i)
        cum_r_j = self.cum_return(traj_j)
        #print(abs_r_i + abs_r_j)
        return cum_r_i, cum_r_j
