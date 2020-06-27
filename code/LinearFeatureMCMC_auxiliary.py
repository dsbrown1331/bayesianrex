#make sure it uses the custom baselines package
import sys
sys.path.insert(0,'./baselines/')

import argparse
# coding: utf-8

# Take as input a compressed pretrained network or run T_REX before hand
# Then run MCMC and save posterior chain


import pickle
import copy
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
from StrippedNet import EmbeddingNet
from baselines.common.trex_utils import preprocess



def generate_debug_demos(env, env_name, agent, model_dir):
    checkpoint_min = 50
    checkpoint_max = 100
    checkpoint_step = 50
    checkpoints = []
    if env_name == "enduro":
        checkpoint_min = 3100
        checkpoint_max = 3650
    elif env_name == "seaquest":
        checkpoint_min = 10
        checkpoint_max = 65
        checkpoint_step = 5
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints.append('0000' + str(i))
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(checkpoints)



    demonstrations = []
    learning_returns = []
    learning_rewards = []
    for checkpoint in checkpoints:

        model_path = model_dir + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + env_name + "_5/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, _ = env.step(action)
                ob_processed = preprocess(ob, env_name)
                #ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
                traj.append(ob_processed)

                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break
            print("traj length", len(traj))
            print("demo length", len(demonstrations))
            demonstrations.append(traj)
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)

    return demonstrations, learning_returns, learning_rewards


def generate_novice_demos(env, env_name, agent, model_dir):
    checkpoint_min = 50
    checkpoint_max = 600
    checkpoint_step = 50
    checkpoints = []
    if env_name == "enduro":
        checkpoint_min = 3100
        checkpoint_max = 3650
    elif env_name == "seaquest":
        checkpoint_min = 10
        checkpoint_max = 65
        checkpoint_step = 5
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints.append('0000' + str(i))
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(checkpoints)



    demonstrations = []
    learning_returns = []
    learning_rewards = []
    for checkpoint in checkpoints:

        model_path = model_dir + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + env_name + "_5/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, _ = env.step(action)
                ob_processed = preprocess(ob, env_name)
                #ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
                traj.append(ob_processed)

                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break
            print("traj length", len(traj))
            print("demo length", len(demonstrations))
            demonstrations.append(traj)
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)

    return demonstrations, learning_returns, learning_rewards



def create_mcmc_likelihood_data(demonstrations):
    '''create all pairwise rankings given list of sorted demonstrations'''
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)
    for i in range(num_demos):
        for j in range(i+1,num_demos):
            #print(i,j)
            traj_i = demonstrations[i]
            traj_j = demonstrations[j]
            label = 1
            training_obs.append((traj_i, traj_j))
            training_labels.append(label)

    return training_obs, training_labels









def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array(s)).float().to(device)).item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))

def print_traj_returns(reward_net, demonstrations):
    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

def calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts, confidence=1):
    '''use (i,j) indices and precomputed feature counts to do faster pairwise ranking loss'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    #print(device)
    #don't need any gradients
    with torch.no_grad():

        #do matrix multiply with last layer of network and the demo_cnts
        #print(list(reward_net.fc2.parameters()))
        linear = last_layer.weight.data  #not using bias
        #print(linear)
        #print(bias)
        weights = linear.squeeze() #append bias and weights from last fc layer together
        #print('weights',weights)
        #print('demo_cnts', demo_cnts)
        demo_returns = confidence * torch.mv(demo_cnts, weights)

        #positivity prior
        if demo_returns[0] < 0.0:
            return torch.Tensor([-float("Inf")])


        loss_criterion = nn.CrossEntropyLoss(reduction='sum') #sum up losses
        cum_log_likelihood = 0.0
        outputs = torch.zeros(len(pairwise_prefs),2) #each row is a new pair of returns
        for p, ppref in enumerate(pairwise_prefs):
            i,j = ppref
            outputs[p,:] = torch.tensor([demo_returns[i], demo_returns[j]])
        labels = torch.ones(len(pairwise_prefs)).long()

        #outputs = outputs.unsqueeze(0)
        #print(outputs)
        #print(labels)
        cum_log_likelihood = -loss_criterion(outputs, labels)
            #if labels == 0:
            #    log_likelihood = torch.log(return_i/(return_i + return_j))
            #else:
            #    log_likelihood = torch.log(return_j/(return_i + return_j))
            #print("ll",log_likelihood)
            #cum_log_likelihood += log_likelihood
    return cum_log_likelihood




def random_search(reward_net, demonstrations, num_trials, stdev = 0.1):
    '''hill climbing random search'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_likelihood = -np.inf
    best_reward = copy.deepcopy(reward_net)
    #create the pairwise rankings for loss calculations
    demo_pairs, preference_labels = create_mcmc_likelihood_data(demonstrations)
    for i in range(num_trials):
        print()
        print("trial", i)
        reward_net_proposal = copy.deepcopy(best_reward)
        #add random noise to weights
        with torch.no_grad():
            for param in reward_net_proposal.parameters():
                param.add_(torch.randn(param.size()).to(device) * stdev)
        #print_traj_returns(reward_net_proposal, demonstrations)
        cum_loglik = calc_pairwise_ranking_loss(reward_net_proposal, demo_pairs, preference_labels)
        print("pair-wise ranking loglik", cum_loglik)
        if cum_loglik > best_likelihood:
            best_likelihood = cum_loglik
            best_reward = copy.deepcopy(reward_net_proposal)
            print("updating best to ", best_likelihood)
        else:
            print("rejecting")
    return best_reward

def generate_feature_counts(demos, reward_net):
    feature_cnts = torch.zeros(len(demos), reward_net.fc2.in_features) #no bias
    for i in range(len(demos)):
        traj = np.array(demos[i])
        traj = torch.from_numpy(traj).float().to(device)
        #print(len(trajectory))
        feature_cnts[i,:] = reward_net.state_features(traj).squeeze().float().to(device)
    return feature_cnts.to(device)

def get_weight_vector(last_layer):
    '''take fc2 layer and return numpy array of weights and bias'''
    linear = last_layer.weight.data
    #print(linear)
    #print(bias)
    with torch.no_grad():
        weights = linear.squeeze().cpu().numpy()
    return weights

def write_weights_likelihood(last_layer, loglik, file_writer):
    if args.debug:
        print("writing weights")
    #convert last layer to numpy array
    np_weights = get_weight_vector(last_layer)
    for w in np_weights:
        file_writer.write(str(w)+",")
    file_writer.write(str(loglik.item()) + "\n")

def compute_l1(last_layer):
    linear = last_layer.weight.data
    #print(linear)
    #print(bias)
    with torch.no_grad():
        weights = linear.squeeze().cpu().numpy()
    #print("output", np.sum(np.abs(weights)))
    return np.sum(np.abs(weights))

def compute_l2(last_layer):
    linear = last_layer.weight.data
    #print(linear)
    #print(bias)
    with torch.no_grad():
        weights = linear.squeeze().cpu().numpy()
    #print("output", np.sum(np.abs(weights)))
    return np.linalg.norm(weights)


def mcmc_map_search(reward_net, demonstrations, pairwise_prefs, demo_cnts, num_steps, step_stdev, weight_output_filename, weight_init):
    '''run metropolis hastings MCMC and record weights in chain'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    demo_pairs, preference_labels = create_mcmc_likelihood_data(demonstrations)

    writer = open(weight_output_filename,'w')

    last_layer = reward_net.fc2

    if weight_init == "randn":
        with torch.no_grad():
            print(last_layer.parameters())
            linear = last_layer.weight.data

            linear.add_(torch.randn(linear.size()).to(device) * step_stdev)
    elif ":" in weight_init:
        print(weight_init.strip().split(':'))
        weight_index, init_weight = weight_init.strip().split(':')
        init_weight = float(init_weight)
        weight_index = int(weight_index)
        print("weight index", weight_index, "init weight", init_weight)
        #initialize with one hot in weight_init position (i.e. initialize in one of the corners of the unit 1-norm sphere)
        with torch.no_grad():
            #get size of weight vector
            num_weights = reward_net.fc2.in_features  #not including the bias weight
            #set up initial weights
            new_linear = torch.zeros(num_weights)
            new_bias = torch.zeros(1)
            if weight_index < num_weights:
                new_linear[weight_index] = init_weight
            else:
                new_bias[0] = init_weight
            #unsqueeze since nn.Linear wants a 2-d tensor for weights
            new_linear = new_linear.unsqueeze(0)
            print("new linear", new_linear)
            print("new bias", new_bias)
            with torch.no_grad():
                #print(last_layer.weight)
                #print(last_layer.bias)
                #print(last_layer.weight.data)
                #print(last_layer.bias.data)
                last_layer.weight.data = new_linear.to(device)
                last_layer.bias.data = new_bias.to(device)
    else:
        print("not a valid weight initialization for MCMC")
        sys.exit()

    #normalize the weight vector to have unit 1-norm...why not unit 2-norm, WCFB won't work without expert...I guess we could do D-REX and estimate
    l2_norm = np.array([compute_l2(last_layer)])

    with torch.no_grad():
        linear = last_layer.weight.data
        print(last_layer.parameters())
        linear.div_(torch.from_numpy(l2_norm).float().to(device))

    if args.debug:
        print("normalized last layer", compute_l2(last_layer))
        print("weights", get_weight_vector(last_layer))

    #import time
    #start_t = time.time()
    #starting_loglik = calc_pairwise_ranking_loss(reward_net, demo_pairs, preference_labels)
    #end_t = time.time()
    #print("slow likelihood", starting_loglik, "time", 1000*(end_t - start_t))
    #start_t = time.time()
    starting_loglik = calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts)
    #end_t = time.time()
    #print("new fast? likelihood", new_starting_loglik, "time", 1000*(end_t - start_t))
    #print(bunnY)

    map_loglik = starting_loglik
    map_reward = copy.deepcopy(reward_net.fc2)

    cur_reward = copy.deepcopy(reward_net.fc2)
    cur_loglik = starting_loglik



    reject_cnt = 0
    accept_cnt = 0

    for i in range(num_steps):
        if args.debug:
            print("step", i)
        #take a proposal step
        proposal_reward = copy.deepcopy(cur_reward)
        #add random noise to weights of last layer
        with torch.no_grad():
            for param in proposal_reward.parameters():
                param.add_(torch.randn(param.size()).to(device) * step_stdev)
        l2_norm = np.array([compute_l2(proposal_reward)])
        #normalize the weight vector...
        with torch.no_grad():
            for param in proposal_reward.parameters():
                param.div_(torch.from_numpy(l2_norm).float().to(device))
        if args.debug:
            print("normalized last layer", compute_l2(proposal_reward))
        #debugging info
        #print_traj_returns(proposal_reward, demonstrations)
        #calculate prob of proposal
        prop_loglik = calc_linearized_pairwise_ranking_loss(proposal_reward, pairwise_prefs, demo_cnts)
        if args.debug:
            print("proposal loglik", prop_loglik.item())
            print("cur loglik", cur_loglik.item())
        if prop_loglik > cur_loglik:
            #print()
            #accept always
            if args.debug:
                print("accept")
            accept_cnt += 1
            cur_reward = copy.deepcopy(proposal_reward)
            cur_loglik = prop_loglik

            #check if this is best so far
            if prop_loglik > map_loglik:
                map_loglik = prop_loglik
                map_reward = copy.deepcopy(proposal_reward)
                print()
                print("step", i)

                print("proposal loglik", prop_loglik.item())

                print("updating map to ", prop_loglik)
        else:
            #accept with prob exp(prop_loglik - cur_loglik)
            if np.random.rand() < torch.exp(prop_loglik - cur_loglik).item():
                #print()
                #print("step", i)
                if args.debug:
                    print("proposal loglik", prop_loglik.item())
                    print("probabilistic accept")
                accept_cnt += 1
                cur_reward = copy.deepcopy(proposal_reward)
                cur_loglik = prop_loglik
            else:
                #reject and stick with cur_reward
                if args.debug:
                    print("reject")
                reject_cnt += 1

        #save chain of weights
        write_weights_likelihood(cur_reward, cur_loglik, writer)
    print("num rejects", reject_cnt)
    print("num accepts", accept_cnt)
    writer.close()
    return map_reward



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--map_reward_model_path', default='', help="name and location for learned model params, e.g. ../mcmc_data/breakout_map.params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default = ".", help="path to directory that contains checkpoint models for demos are stored")
    parser.add_argument('--num_mcmc_steps', default=2000, type = int, help="number of proposals to generate for MCMC")
    parser.add_argument('--mcmc_step_size', default = 0.005, type=float, help="proposal step is gaussian with zero mean and mcmc_step_size stdev")
    parser.add_argument('--pretrained_network', help='path to pretrained network weights to form \phi(s) using all but last layer')
    parser.add_argument('--weight_outputfile', help='filename including path to write the chain to')
    parser.add_argument('--debug', help='use fewer demos to speed things up while debugging', action='store_true' )
    parser.add_argument('--plot', help='plot out the feature counts over time for demos', action='store_true' )
    parser.add_argument('--weight_init', help="defaults to randn, specify integer value to start in a corner of L1-sphere", default="randn")
    parser.add_argument('--encoding_dims', help="size of latent space", type=int)

    args = parser.parse_args()


    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"
    print(env_type)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    stochastic = True


    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    if args.debug:
        print("*"*30)
        print("*"*30)
        print("\tDEBUG MODE")
        print("*"*30)
        print("*"*30)
        demonstrations, learning_returns, learning_rewards = generate_debug_demos(env, env_name, agent, args.models_dir)
    else:
        demonstrations, learning_returns, learning_rewards = generate_novice_demos(env, env_name, agent, args.models_dir)

    #sort the demonstrations according to ground truth reward to simulate ranked demos

    print([a[0] for a in zip(learning_returns, demonstrations)])
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)


    # Now we download a pretrained network to form \phi(s) the state features where the reward is now w^T \phi(s)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = EmbeddingNet(args.encoding_dims)
    reward_net.load_state_dict(torch.load(args.pretrained_network, map_location=device))
    #reinitialize last layer
    num_features = reward_net.fc2.in_features

    print("reward is linear combination of ", num_features, "features")
    reward_net.fc2 = nn.Linear(num_features, 1, bias=False) #last layer just outputs the scalar reward = w^T \phi(s)
    reward_net.to(device)
    #freeze all weights so there are no gradients (we'll manually update the last layer via proposals so no grads required)
    for param in reward_net.parameters():
        param.requires_grad = False

    #get num_demos by num_features + 1 (bias) numpy array with (un-discounted) feature counts from pretrained network
    demo_cnts = generate_feature_counts(demonstrations, reward_net)
    print("demo counts")
    print(demo_cnts)
    if args.plot:
        plotable_cnts = demo_cnts.cpu().numpy()
        import matplotlib.pyplot as plt
        for f in range(num_features):
            #plt.figure(f)
            if plotable_cnts[0,f] < plotable_cnts[-1,f]: #increasing
                plt.figure(0)
                plt.plot(plotable_cnts[:,f], '-o', label='feature ' + str(f))
                plt.legend()
            elif plotable_cnts[0,f] > plotable_cnts[-1,f]: #decreasing
                plt.figure(1)
                plt.plot(plotable_cnts[:,f], '-o', label='feature ' + str(f))
                plt.legend()
            else: #unknown
                plt.figure(2)
                plt.plot(plotable_cnts[:,f], '-o', label='feature ' + str(f))
                plt.legend()

        plt.show()
        #print(demo_cnts.shape)

    #just need index tuples (i,j) denoting j is preferred to i. Assuming all pairwise prefs for now
    #check if really better, there might be ties!
    pairwise_prefs = []
    for i in range(len(demonstrations)):
        for j in range(i+1, len(demonstrations)):
            if sorted_returns[i] < sorted_returns[j]:
                pairwise_prefs.append((i,j))
            else: # they are equal
                print("not using equal prefs", i, j, sorted_returns[i], sorted_returns[j])
                #pairwise_prefs.append((i,j))
                #pairwise_prefs.append((j,i))


    #run random search over weights
    #best_reward = random_search(reward_net, demonstrations, 40, stdev = 0.01)
    best_reward_lastlayer = mcmc_map_search(reward_net, demonstrations, pairwise_prefs, demo_cnts, args.num_mcmc_steps, args.mcmc_step_size, args.weight_outputfile, args.weight_init)
    #turn this into a full network
    best_reward = EmbeddingNet(args.encoding_dims)
    #best_reward.fc2 = nn.Linear(num_features, 1, bias=False)
    best_reward.load_state_dict(torch.load(args.pretrained_network, map_location=device))
    best_reward.fc2 = best_reward_lastlayer

    best_reward.to(device)
    #print(best_reward.state_dict())
    #save best reward network
    torch.save(best_reward.state_dict(), args.map_reward_model_path)
    demo_pairs, preference_labels = create_mcmc_likelihood_data(demonstrations)
    print_traj_returns(best_reward, demonstrations)


    #add random
