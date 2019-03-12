""" Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

utils.py

This file provides various utility functions used for running matrix communication games (MCGs)

Author: Ryan Lowe
"""

import numpy as np
import torch
import os
import pickle


""" Miscellaneous functions """


def discount_rewards(rews, gamma, n_steps=None):
    # Takes rewards in a batch, and calculates discounted returns
    # Note that returns are truncated at the end of the batch
    # nstep controls how many steps you take the sum over
    rews_disc = np.zeros((len(rews),))
    rews_temp = np.zeros((len(rews),))
    for i in range(len(rews) - 1, -1, -1):
        rews_temp[i] = rews[i]
        rews_temp[i+1:] *= gamma
        if n_steps is None:
            rews_disc[i] = sum(rews_temp)
        else:
            rews_disc[i] = sum(rews_temp[i: i + n_steps])
    if n_steps is None:
        return rews_disc
    return rews_disc[:len(rews)]


def index_to_onehot(m, v_len=2):
    # Converts an index (converted to integer) into a one-hot vector
    m = to_int(m)
    n = np.zeros(v_len)
    n[to_int(m)] = 1
    return n


def to_int(n):
    # Converts various things to integers
    if type(n) is int:
        return n
    elif type(n) is float:
        return int(n)
    else:
        return int(n.data.numpy())


def probs_from_counts(l, ldim, eps=0):
    # Outputs a probability distribution (list) of length ldim, by counting event occurrences in l
    l_c = [eps] * ldim
    for i in l:
        l_c[i] += 1. / len(l)
    return l_c


def save_data(args, header, data):
    # Saves data from the training loop of mcg.py. Saves two different things:
    #   (1) the arguments that were passed into parser in mcg.py -> params.txt
    #   (2) a dictionary of data produced by the main loop -> data.pickle
    folder = args.save_dir + args.exp_name + '/'
    f_data = folder + 'data.pickle'
    f_params = folder + 'params.txt'
    data = list(map(list, zip(*data)))
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(f_params, 'w') as f1:
        f1.write(str(args))
        f1.write('Saved params:\n')
        f1.write(str(header))
    with open(f_data, 'wb') as f1:
        pickle.dump(data, f1)


def save_model(args, agents, i):
    # Saves the final model parameters of each agent into a .txt file, using torch.save
    folder = args.save_dir + args.exp_name + '/'
    f_model = []
    for j, ag in enumerate(agents):
        f_model.append(f'{folder}ag{j}_model_{i}.txt')
    if not os.path.exists(folder):
        os.makedirs(folder)
    for ag, f_m in zip(agents, f_model):
        torch.save(ag.state_dict(), f_m)


def get_named_params(agent, s):
    # Returns all parameters for a given agent that contain the substring s
    p = []
    for name, param in agent.named_parameters():
        if s in name:
            p.append(param)
    return p


def get_all_but_named_params(agent, s):
    # Returns all parameters for a given agent that *do not* contain the substring s
    p = []
    for name, param in agent.named_parameters():
        if s not in name:
            p.append(param)
    return p


""" Calculating statistics about comms and acts """


def calc_stats(comms, acts, n_comm, n_acts, stats):
    # Produces a matrix ('stats') that counts co-occurrences of messages and actions
    # Can update an existing 'stats' matrix (=None if there is none)
    # Calls bin_acts to do the heavy lifting
    comms = [to_int(m) for m in comms]
    acts = [to_int(a) for a in acts]
    stats = bin_acts(comms, acts, n_comm, n_acts, stats)
    return stats


def bin_acts(comms, acts, n_comm, n_acts, b=None):
    # Binning function that creates a matrix that counts co-occurrences of messages and actions
    if b is None:
        b = np.zeros((n_comm, n_acts))
    for a, c in zip(acts, comms):
        b[c][a] += 1
    return b


