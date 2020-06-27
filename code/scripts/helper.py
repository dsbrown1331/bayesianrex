import numpy as np

def get_weightchain_array(mcmc_chain_filename, burn=5000, skip=20, return_likelihood=False, preburn_until_accept=True):
    #load the mean of the MCMC chain
    reader = open(mcmc_chain_filename)
    data = []
    likelihood = []
    for line in reader:
        parsed = line.strip().split(',')
        np_line = []
        for s in parsed[:-1]: #don't get last element since it's the likelihood
            np_line.append(float(s))
        likelihood.append(float(parsed[-1]))
        data.append(np_line)
    likelihood = np.array(likelihood)
    data = np.array(data)
    data = data[likelihood != -float('inf')]
    print(data.shape)
    num_preburn = np.sum(likelihood == -float('inf'))
    likelihood = likelihood[likelihood != -float('inf')]
    print('preburning ', num_preburn, "to find first accept")
    data = data[burn::skip,:]
    print(data.shape)
    likelihood = likelihood[burn::skip]
    print("chain shape", data.shape)
    reader.close()
    assert(len(likelihood) == len(data))
    if not return_likelihood:
        return data
    else:
        return data, likelihood

#deprecated...use newer version for parsing newer policy eval output files
def parse_avefcount_array_noop(fcount_file):
    '''returns a list of returns and a numpy array of ave feature counts'''
    reader = open(fcount_file)
    weights = []
    returns = []
    for i,line in enumerate(reader):
        if i == 0: #read in the np_weights
            parsed = line.strip().split(',')
            for w in parsed:
                weights.append(float(w))
        elif i == 2: #read in the returns
            parsed = line.strip().split(',')
            for r in parsed:
                returns.append(float(r))
    return returns, weights

def parse_fcount_policy_eval(fcount_file):
    '''returns a list of returns and a numpy array of ave feature counts'''
    reader = open(fcount_file)
    all_lines = []
    for line in reader:
        all_lines.append(line)
        #print(line)
    #last line is the number of steps for each rollout
    rollout_lengths = all_lines[-1].split(',')
    rollout_lengths = [float(l) for l in rollout_lengths]

    #second to last line is the returns for each rollout
    rollout_returns = all_lines[-2].split(',')
    rollout_returns = [float(r) for r in rollout_returns]

    #first line is the expected features counts of the policy averaged over rollouts
    ave_fcounts = all_lines[0].split(',')
    ave_fcounts = [float(f) for f in ave_fcounts]

    #the things in between are the fcounts for each individual rollout
    rollout_fcounts = []
    for line in all_lines[1:-2]:
        rollout_fcounts.append([float(f) for f in line.split(',')])


    assert(len(rollout_fcounts) == len(rollout_lengths) == len(rollout_returns))
    return np.array(ave_fcounts), np.array(rollout_returns), np.array(rollout_lengths), np.array(rollout_fcounts)

    #
    # weights = []
    # returns = []
    # for i,line in enumerate(reader):
    #     if i == 0: #read in the expected feature count np_weights
    #         parsed = line.strip().split(',')
    #         for w in parsed:
    #             weights.append(float(w))
    #     elif i == 1: #read in the returns
    #         parsed = line.strip().split(',')
    #         for r in parsed:
    #             returns.append(float(r))
    # return returns, weights

def worst_percentile(_sequence, alpha):
    sorted_seq = sorted(_sequence)
    #find alpha percentile
    alpha_indx = int(np.floor(alpha * len(_sequence)))
    if alpha_indx >= len(sorted_seq):
        return sorted_seq[-1]
    return sorted_seq[alpha_indx]

def average_var(_sequence, alpha):
    sorted_seq = sorted(_sequence)
    #find alpha percentile
    alpha_indx = int(np.floor(alpha * len(_sequence)))
    return np.mean(sorted_seq[:alpha_indx])


