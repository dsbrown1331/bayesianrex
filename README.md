# Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences

Daniel S. Brown, Russell Coleman, Ravi Srinivasan, Scott Niekum

<p align="center">
  <a href="https://arxiv.org/abs/2002.09089">View on ArXiv</a> |
  <a href="https://sites.google.com/view/bayesianrex/">Project Website</a>
</p>


<p align=center>
  <img src='assets/BREXslide.pdf' width=600>
</p>



This repository contains a code used to conduct experiments reported in the paper "Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences" published at ICML 2020.

If you find this repository is useful in your research, please cite the paper:
```
@InProceedings{brown2020safe,
  title = {Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences},
  author = {Brown, Daniel S. and  Coleman, Russell and Srinivasan, Ravi and Niekum, Scott},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  year = {2020}
}
```



# Instructions for running source code

## Set up conda environment and dependencies
conda env create -f environment.yml

## Install baselines
Follow the instructions in code/baselines/README.md

## Download demonstration data
For demonstrations we used the data from Brown et al. "Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations", ICML, 2019.
To download the demonstration data, download from here (https://github.com/dsbrown1331/learning-rewards-of-learners/releases/), and extract under the models directory.


For simplicity, in the following examples we have used Breakout as the environment, but this can be replaced with any of the other environments in the ALE.

## Pre-train reward embedding
cd code/
conda activate bayesianrex
export CUDA_VISIBLE_DEVICES=0
python LearnAtariRewardLinear.py --env_name breakout --reward_model_path ../pretrained_networks/breakout_pretrained.params --models_dir ../

To train using just self-supervised add the argument --loss_fn ss
To train just using the ranking loss use the argument --loss_fn trex

## Strip network down to just the embedding layers
cd code/scripts/
bash strip_to_embedding_networks.sh ../../pretrained_networks/ breakout_pretrained.params


## Learning the reward function posterior via Bayesian REX
The main file to run is: LinearFeatureMCMC_auxiliary.py
This will run mcmc over the pretrained network weights for Atari.
Here's an example of how to run it:

cd code/
python LinearFeatureMCMC_auxiliary.py --env_name breakout --models_dir ../models/ --weight_outputfile ../mcmc_data/breakout_mcmc.txt --num_mcmc_steps 200000 --map_reward_model_path ../mcmc_data/breakout_map.params --pretrained_network ../pretrained_networks/breakout_pretrained.params_stripped.params --encoding_dims 64

This will generate a text file "breakout_mcmcm.txt" of the weights and loglikelihoods from MCMC. It will also produce a file "breakout_map.params" with the parameters of the MAP reward function found via MCMC.


## To run RL with the mean reward from MCMC:
cd code/
conda activate bayesianrex
export CUDA_VISIBLE_DEVICES=0
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=~/tflogs/breakout_mean python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --custom_reward mcmc_mean --custom_reward_path ../mcmc_data/breakout_map.params --mcmc_chain_path ../mcmc_data/breakout_mcmc.txt --seed 0 --num_timesteps=5e7  --save_interval=43000 --num_env 9 --embedding_dim 64


## To run RL with the MAP reward from MCMC:
code code
conda activate bayesianrex
export CUDA_VISIBLE_DEVICES=0
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=~/tflogs/breakout_mean python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --custom_reward mcmc_map --custom_reward_path ../mcmc_data/breakout_map.params --seed 0 --num_timesteps=5e7  --save_interval=43000 --num_env 9 --embedding_dim 64

## To evaluate the performance of RL policy
cd code/
python evaluateLearnedPolicy.py --checkpointpath ~/tflogs/breakout_mean/checkpoints/43000

This will write the output to the code/eval/ folder. You can then run the helper script
python compute_mean_std.py [name of generated file] 
to compute the mean and standard deviation of the policy performance on the ground truth reward.


## High confidence policy evaluation
First perform policy evaluation to get the expected feature counts of the policy using 
python computePolicyExpectedFeatureCountsNetwork.py --env_name breakout --checkpointpath 

For example, to run expected feature counts for the MAP policy learned via Bayesian REX run:
python computePolicyExpectedFeatureCountsNetwork.py --env_name breakout --checkpointpath ~/tflogs/breakout_map/checkpoints/43000 --pretrained_network ../pretrained_networks/breakout_pretrained.params_stripped.params --fcount_file ../policy_evals/breakout_map_fcounts.txt

To eval a no-op policy simply add the flag --no_op

To evalute the performance of a policy under the posterior distribution simply run
cd code/scripts/
python analyze_return_distribution.py --env_name breakout --eval_fcounts ../policy/evals/breakout_map_fcounts.txt --alpha 0.05 --mcmc_file ../mcmc_data/breakout_mcmc.txt


## Example To record videos of learned behaviors
python run_test.py --env_id BreakoutNoFrameskip-v4 --env_type atari --model_path ../models/breakout/checkpoints/03600 --record_video --episode_count 1 --render

You can omit the last flag --record_video. When it is turned on, then the videos will be recorded in a videos/ directory below the current directory. If --render is omitted then it will simply print returns to the command line.


## Visualizations of learned embeddings
See the following files in the code/ directory for reproducing the visualizations of the latent space found in the Appendix.

### `DemoGraph.py`
Generates demonstration videos from pretrained RL agents, and plots the encoding into the latent space as well as the decoding over time. Takes one argument, which is a pretrained network.

### `DemoGraphRunner.py`
Runs `DemoGraph.py` over every file `.params` in a folder, used to generate many demo graphs at once. Takes one argument, which is the folder in which the `.params` files can be found.

### `LatentVisualizer.py`
Opens a GUI in which the user can examine samples from the latent space, generate random samples, and slide individual dimensions to see the effect on the decoded image.

### `RandomSample.py`
Takes in a folder containing `.params` files much like `DemoGraphRunner.py`, and generates random samples from the latent space, zero samples, forward dynamics rollouts, and visualizations of greatest dimensions for a given pretrained feature encoding network.
