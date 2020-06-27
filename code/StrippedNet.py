#----------------------------------------------------------------------------------------
# Argument parsing
# import sys
# if len(sys.argv) < 2:
#     print("Usage: " + sys.argv[0] + " <model_path>")
#     sys.exit()
# model_path = sys.argv[1].strip()

# Long imports
import torch
import torch.nn as nn
import torch.nn.functional as F
#----------------------------------------------------------------------------------------

# Arbitrarily chosen number of dimensions in latent space
#ENCODING_DIMS = 30

class EmbeddingNet(nn.Module):
    def __init__(self, ENCODING_DIMS):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ENCODING_DIMS = ENCODING_DIMS
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 16, 3, stride=1)

        # This is the width of the layer between the convolved framestack
        # and the actual latent space. Scales with self.ENCODING_DIMS
        intermediate_dimension = min(784, max(64, self.ENCODING_DIMS*2))

        # Brings the convolved frame down to intermediate dimension just
        # before being sent to latent space
        self.fc1 = nn.Linear(784, intermediate_dimension)

        # This brings from intermediate dimension to latent space. Named mu
        # because in the full network it includes a var also, to sample for
        # the autoencoder
        self.fc_mu = nn.Linear(intermediate_dimension, self.ENCODING_DIMS)

        # This is the actual T-REX layer; linear comb. from self.ENCODING_DIMS
        self.fc2 = nn.Linear(self.ENCODING_DIMS, 1)

    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        mu = self.fc_mu(x)

        r = self.fc2(mu)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards#, sum_abs_rewards, mu

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, mu1 = self.cum_return(traj_i)
        cum_r_j, abs_r_j, mu2 = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j, mu1, mu2


    def state_features(self, traj):

        with torch.no_grad():
            accum = torch.zeros(1,self.ENCODING_DIMS).float().to(self.device)
            for x in traj:
                x = x.permute(0,3,1,2) #get into NCHW format
                #compute forward pass of reward network
                x = F.leaky_relu(self.conv1(x))
                x = F.leaky_relu(self.conv2(x))
                x = F.leaky_relu(self.conv3(x))
                x = F.leaky_relu(self.conv4(x))
                x = x.view(-1, 784)
                x = F.leaky_relu(self.fc1(x))
                mu = self.fc_mu(x)
                accum.add_(mu)
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
            x = F.leaky_relu(self.fc1(x))
            mu = self.fc_mu(x)

        return mu





### Usage example
#net = Net()
#net.load_state_dict(torch.load(model_path))
