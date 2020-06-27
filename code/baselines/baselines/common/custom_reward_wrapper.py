import gym
import numpy as np
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.trex_utils import preprocess
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReacherNet(nn.Module):
    def __init__(self, obs_in):
        super().__init__()

        self.fc1 = nn.Linear(obs_in, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,32)
        #self.fc1 = nn.Linear(1936,64)
        self.fc4 = nn.Linear(32, 1)



    def forward(self, traj):
        #assumes traj is of size [batch, height, width, channel]
        #this formatting should be done before feeding through network
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #print(traj.shape)
        #compute forward pass of reward network
        x = torch.relu(self.fc1(traj))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        #print(x)
        #r = torch.tanh(self.fc2(x)) #clip reward?
        #r = F.celu(self.fc2(x))
        rs = -torch.relu(self.fc4(x))
        #print(rs)
        return rs

class VecPyTorchMujocoReward(VecEnvWrapper):
    def __init__(self, venv, reward_net_path, env_name):
        VecEnvWrapper.__init__(self, venv)
        self.reward_net = ReacherNet(11)
        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.
        self.env_name = env_name
        self.ctrl_coeff = 0.0

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        #get actions taken to calculate control penalty
        acs = self.venv.last_actions
        #acs = self.venv.last_actions if hasattr(self.venv,'last_actions') else np.zeros((len(rews),2))
        #assert self.ctrl_coeff == 0.0 or (self.ctrl_coeff != 0.0 and acs.size > 0)
        #print()
        #print('actions', acs)
        #TODO? do I need to load the running mean?

        with torch.no_grad():
            rews_network = self.reward_net.forward(torch.from_numpy(np.array(obs)).float().to(self.device)).cpu().numpy().squeeze()
        #print(rews)
        #print(obs)
        #print(rews_network)

        #add in control l2_penalty on controls
        rews = rews_network - self.ctrl_coeff *np.sum(acs**2,axis=1)
        #print(rews)
        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs


#Ibarz network
class AtariNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1, bias=False)


    def forward(self, traj):
        '''calculate cumulative return of trajectory'''
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x) #clip reward?
        #r = self.fc2(x) #clip reward?
        return r

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
        self.fc2 = nn.Linear(self.ENCODING_DIMS, 1, bias=False)


    def forward(self, traj):
        '''calculate cumulative return of trajectory'''
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
        #r = self.fc2(x) #clip reward?
        return r



# class AtariNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.fc1 = nn.Linear(64 * 7 * 7, 512)
#         self.output = nn.Linear(512, 1)
#
#     def forward(self, traj):
#         '''calculate cumulative return of trajectory'''
#         x = traj.permute(0,3,1,2) #get into NCHW format
#         #compute forward pass of reward network
#         conv1_output = F.relu(self.conv1(x))
#         conv2_output = F.relu(self.conv2(conv1_output))
#         conv3_output = F.relu(self.conv3(conv2_output))
#         fc1_output = F.relu(self.fc1(conv3_output.view(conv3_output.size(0),-1)))
#         r = self.output(fc1_output)
#         return r



class VecRLplusIRLAtariReward(VecEnvWrapper):
    def __init__(self, venv, reward_net_path, combo_param):
        VecEnvWrapper.__init__(self, venv)
        self.reward_net = AtariNet()
        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.lamda = combo_param #how much weight to give to IRL verus RL combo_param \in [0,1] with 0 being RL and 1 being IRL
        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
##Testing network to see why always giving zero rewards....
        #import pickle
        #filename = 'rand_obs.pkl'
        #infile = open(filename,'rb')
        #rand_obs = pickle.load(infile)
        #infile.close()
        #traj = [obs / 255.0] #normalize!
        #import matplotlib.pyplot as plt
        #plt.figure(1)
        #plt.imshow(obs[0,:,:,0])
        #plt.figure(2)
        #plt.imshow(rand_obs[0,:,:,0])
        #plt.show()
        #print(obs.shape)
        with torch.no_grad():
            rews_network = self.reward_net.cum_return(torch.from_numpy(np.array(obs)).float().to(self.device)).cpu().numpy().transpose()[0]
            #rews2= self.reward_net.cum_return(torch.from_numpy(np.array([rand_obs])).float().to(self.device)).cpu().numpy().transpose()[0]
        #self.rew_rms.update(rews_network)
        #r_hat = rews_network
        #r_hat = np.clip((r_hat - self.rew_rms.mean) / np.sqrt(self.rew_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        #print(rews1)
        #   print(rews2)

        #print(obs.shape)
        # obs shape: [num_env,84,84,4] in case of atari games

        #combine IRL and RL rewards using lambda parameter like Yuke Zhu's paper "Reinforcement and Imitation Learningfor Diverse Visuomotor Skills"
        reward_combo = self.lamda * rews_network + (1-self.lamda) * rews

        return obs, reward_combo , news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs


class VecPyTorchAtariReward(VecEnvWrapper):
    def __init__(self, venv, reward_net_path, env_name):
        VecEnvWrapper.__init__(self, venv)
        self.reward_net = AtariNet()
        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.
        self.env_name = env_name

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        # obs shape: [num_env,84,84,4] in case of atari games
        #plt.subplot(1,2,1)
        #plt.imshow(obs[0][:,:,0])
        #crop off top of image
        #n = 10
        #no_score_obs = copy.deepcopy(obs)
        #obs[:,:n,:,:] = 0

        #Need to normalize for my reward function
        #normed_obs = obs / 255.0
        #mask and normalize for input to network
        normed_obs = preprocess(obs, self.env_name)
        #plt.subplot(1,2,2)
        #plt.imshow(normed_obs[0][:,:,0])
        #plt.show()
        #print(traj[0][0][40:60,:,:])

        with torch.no_grad():
            rews_network = self.reward_net.forward(torch.from_numpy(np.array(normed_obs)).float().to(self.device)).cpu().numpy().squeeze()

        return obs, rews_network, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs

class VecMCMCMAPAtariReward(VecEnvWrapper):
    def __init__(self, venv, reward_net_path, embedding_dim, env_name):
        VecEnvWrapper.__init__(self, venv)
        self.reward_net = EmbeddingNet(embedding_dim)
        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.
        self.env_name = env_name

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        # obs shape: [num_env,84,84,4] in case of atari games
        #plt.subplot(1,2,1)
        #plt.imshow(obs[0][:,:,0])
        #crop off top of image
        #n = 10
        #no_score_obs = copy.deepcopy(obs)
        #obs[:,:n,:,:] = 0

        #Need to normalize for my reward function
        #normed_obs = obs / 255.0
        #mask and normalize for input to network
        normed_obs = preprocess(obs, self.env_name)
        #plt.subplot(1,2,2)
        #plt.imshow(normed_obs[0][:,:,0])
        #plt.show()
        #print(traj[0][0][40:60,:,:])

        with torch.no_grad():
            rews_network = self.reward_net.forward(torch.from_numpy(np.array(normed_obs)).float().to(self.device)).cpu().numpy().squeeze()

        return obs, rews_network, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs


#TODO: need to test with RL
class VecMCMCMeanAtariReward(VecEnvWrapper):
    def __init__(self, venv, pretrained_reward_net_path, chain_path, embedding_dim, env_name):
        VecEnvWrapper.__init__(self, venv)
        self.reward_net = EmbeddingNet(embedding_dim)
        #load the pretrained weights
        self.reward_net.load_state_dict(torch.load(pretrained_reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #load the mean of the MCMC chain
        burn = 5000
        skip = 20
        reader = open(chain_path)
        data = []
        for line in reader:
            parsed = line.strip().split(',')
            np_line = []
            for s in parsed[:-1]:
                np_line.append(float(s))
            data.append(np_line)
        data = np.array(data)
        #print(data[burn::skip,:].shape)

        #get average across chain and use it as the last layer in the network
        mean_weight = np.mean(data[burn::skip,:], axis = 0)
        #print("mean weights", mean_weight[:-1])
        #print("mean bias", mean_weight[-1])
        #print(mean_weight.shape)
        self.reward_net.fc2 = nn.Linear(embedding_dim, 1, bias=False) #last layer just outputs the scalar reward = w^T \phi(s)

        new_linear = torch.from_numpy(mean_weight)
        print("new linear", new_linear)
        print(new_linear.size())
        with torch.no_grad():
            #unsqueeze since nn.Linear wants a 2-d tensor for weights
            new_linear = new_linear.unsqueeze(0)
            #print("new linear", new_linear)
            #print("new bias", new_bias)
            with torch.no_grad():
                #print(last_layer.weight)
                #print(last_layer.bias)
                #print(last_layer.weight.data)
                #print(last_layer.bias.data)
                self.reward_net.fc2.weight.data = new_linear.float().to(self.device)

            #TODO: print out last layer to make sure it stuck...
            print("USING MEAN WEIGHTS FROM MCMC")
            #with torch.no_grad():
            #    for param in self.reward_net.fc2.parameters():
            #        print(param)

        self.reward_net.to(self.device)

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.
        self.env_name = env_name

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        # obs shape: [num_env,84,84,4] in case of atari games
        #plt.subplot(1,2,1)
        #plt.imshow(obs[0][:,:,0])
        #crop off top of image
        #n = 10
        #no_score_obs = copy.deepcopy(obs)
        #obs[:,:n,:,:] = 0

        #Need to normalize for my reward function
        #normed_obs = obs / 255.0
        #mask and normalize for input to network
        normed_obs = preprocess(obs, self.env_name)
        #plt.subplot(1,2,2)
        #plt.imshow(normed_obs[0][:,:,0])
        #plt.show()
        #print(traj[0][0][40:60,:,:])

        with torch.no_grad():
            rews_network = self.reward_net.forward(torch.from_numpy(np.array(normed_obs)).float().to(self.device)).cpu().numpy().squeeze()

        return obs, rews_network, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs



class VecLiveLongReward(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        rews = np.ones_like(rews)

        #print(obs.shape)
        # obs shape: [num_env,84,84,4] in case of atari games

        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs


import tensorflow as tf
class VecTFRandomReward(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)

        self.graph = tf.Graph()

        config = tf.ConfigProto(
            device_count = {'GPU': 0}) # Run on CPU
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                self.obs = tf.placeholder(tf.float32,[None,84,84,4])

                self.rewards = tf.reduce_mean(
                    tf.random_normal(tf.shape(self.obs)),axis=[1,2,3])


    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        rews = self.sess.run(self.rewards,feed_dict={self.obs:obs})

        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs

class VecTFPreferenceReward(VecEnvWrapper):
    def __init__(self, venv, num_models, model_dir):
        VecEnvWrapper.__init__(self, venv)

        self.graph = tf.Graph()

        config = tf.ConfigProto(
            device_count = {'GPU': 0}) # Run on CPU
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                import os, sys
                dir_path = os.path.dirname(os.path.realpath(__file__))
                sys.path.append(os.path.join(dir_path,'..','..','..','..'))
                from preference_learning import Model

                print(os.path.realpath(model_dir))

                self.models = []
                for i in range(num_models):
                    with tf.variable_scope('model_%d'%i):
                        model = Model(self.venv.observation_space.shape[0])
                        model.saver.restore(self.sess,model_dir+'/model_%d.ckpt'%(i))
                    self.models.append(model)

        """
        try:
            self.save = venv.save
            self.load = venv.load
        except AttributeError:
            pass
        """

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        with self.graph.as_default():
            with self.sess.as_default():
                r_hat = np.zeros_like(rews)
                for model in self.models:
                    r_hat += model.get_reward(obs)

        rews = r_hat / len(self.models)

        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        return obs

if __name__ == "__main__":
    pass
