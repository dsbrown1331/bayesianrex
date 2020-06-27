import numpy as np
import helper
import argparse


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', help="name of the environment, e.g. 'breakout'")
parser.add_argument('--eval_fcounts', help='file with policy fcounts'
parser.add_argument('--mcmc_file', help="name of mcmc file for chain")
parser.add_argument('--alpha', type=float, help="value of alpha-VaR, e.g. alpha = 0.05")


args = parser.parse_args()
print(args.env_name)
#read in the weights as a 2-d array and the feature counts of the policy
W = helper.get_weightchain_array(args.mcmc_file)
print(np.mean(W, axis=0))

print("mean & 0.05-VaR & gt & ave length")

fcounts, returns, lengths, all_fcounts = helper.parse_fcount_policy_eval(args.eval_fcounts)
return_dist = np.dot(W,fcounts)

print("{:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\".format(np.mean(return_dist),helper.worst_percentile(return_dist, args.alpha), np.mean(returns), np.mean(lengths)))
