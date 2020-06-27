### This code will take in any pretrained network and compute the expected feature counts via Monte Carlo sampling according to the last
### layer of the pretrained network


import os
import sys
import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
#import matplotlib.pylab as plt
import argparse
from StrippedNet import EmbeddingNet
from baselines.common.trex_utils import preprocess
import utils




def get_policy_feature_counts(env_name, checkpointpath, feature_net, num_rollouts, no_op=False):
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    elif env_name == "montezumarevenge":
        env_id = "MontezumaRevengeNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"

    stochastic = True

    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })



    env = VecFrameStack(env, 4)


    agent = PPO2Agent(env, env_type, stochastic)  #defaults to stochastic = False (deterministic policy)
    #agent = RandomAgent(env.action_space)

    learning_returns = []
    fcount_rollouts = [] #keep track of the feature counts for each rollout
    num_steps = []

    print("using checkpoint", checkpointpath, "if none then using no-op policy")
    if not no_op:
        agent.load(checkpointpath)
    episode_count = num_rollouts

    f_counts = np.zeros(feature_net.fc2.in_features)


    for i in range(episode_count):
        done = False
        traj = []
        fc_rollout = np.zeros(feature_net.fc2.in_features)
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:#steps < 7000:
            if no_op:
                action = 0
            else:
                action = agent.act(ob, r, done)
            #print(action)
            ob, r, done, _ = env.step(action)
            ob_processed = preprocess(ob, env_name)
            #print(ob_processed.shape)
            phi_s = feature_net.state_feature(torch.from_numpy(ob_processed).float().to(device)).cpu().squeeze().numpy()
            #print(phi_s.shape)
            fc_rollout += phi_s
            f_counts += phi_s
            steps += 1
            #print(steps)
            acc_reward += r[0]
            if done:
                print("steps: {}, return: {}".format(steps,acc_reward))
                break
        fcount_rollouts.append(fc_rollout)
        learning_returns.append(acc_reward)
        num_steps.append(steps)



    env.close()
    #tf.reset_default_graph()
    del agent
    del env

    ave_fcounts = f_counts/episode_count
    print('ave', ave_fcounts)
    print('computed ave', np.mean(np.array(fcount_rollouts), axis=0))
    return learning_returns, ave_fcounts, fcount_rollouts, num_steps

    #return([np.max(learning_returns), np.min(learning_returns), np.mean(learning_returns)])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=1234, help="random seed for experiments")
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--checkpointpath', help="path to check point of policy to evaluate")
    parser.add_argument('--pretrained_network', help='path to pretrained network weights to form \phi(s) using all but last layer')
    parser.add_argument('--num_rollouts', type=int, help='number of rollouts to compute feature counts')
    parser.add_argument('--encoding_dims', default=64, type=int, help='number of dims to encode to')
    parser.add_argument('--fcount_file', help='location to save fcount file')
    parser.add_argument('--no_op', action='store_true', help='run no-op policy evaluation')

    args = parser.parse_args()
    env_name = args.env_name
  
    print("generating feature counts for",args.checkpointpath)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    network_file_loc = os.path.abspath(args.pretrained_network)
    print("Using network at", network_file_loc, "for features.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_net = EmbeddingNet(args.encoding_dims)
    state_dict = torch.load(network_file_loc, map_location=device)
    feature_net.load_state_dict(torch.load(network_file_loc, map_location=device))
    feature_net.to(device)

    if args.no_op:
        checkpointpath = None
    else:
        checkpointpath = args.checkpointpath
    print("evaluating", checkpointpath)
    print("*"*10)
    print(env_name)
    print("*"*10)
    returns, ave_feature_counts, fcounts, num_steps = get_policy_feature_counts(env_name, checkpointpath, feature_net, args.num_rollouts, args.no_op)
    print("returns", returns)
    print("feature counts", ave_feature_counts)
    writer = open(args.fcount_file, 'w')
    utils.write_line(ave_feature_counts, writer)
    for fc in fcounts:
        utils.write_line(fc, writer)
    utils.write_line(returns, writer)
    utils.write_line(num_steps, writer, newline=False)
    writer.close()
