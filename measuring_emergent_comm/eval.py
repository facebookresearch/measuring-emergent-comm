""" Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

eval.py

The code in this file provides the evaluation functions used to calculate various metrics measuring the emergence
of communication.

It includes both metrics that are tracked over the course of training (SC and IC) and metrics that are calculated
from saved models (CIC).

author: Ryan Lowe
"""

import numpy as np
import math
import torch
import utils as U
import models as M
from mcg import MCG
import torch.nn.functional as F


""" Calculating metrics that only take in list of actions and messages (and don't need access to model) """


def calc_mutinfo(acts, comms, n_acts, n_comm):
    # Calculate mutual information between actions and messages
    # Joint probability p(a, c) is calculated by counting co-occurences, *not* by performing interventions
    # If the actions and messages come from the same agent, then this is the speaker consistency (SC)
    # If the actions and messages come from different agents, this is the instantaneous coordinatino (IC)
    comms = [U.to_int(m) for m in comms]
    acts = [U.to_int(a) for a in acts]

    # Calculate probabilities by counting co-occurrences
    p_a = U.probs_from_counts(acts, n_acts)
    p_c = U.probs_from_counts(comms, n_comm)
    p_ac = U.bin_acts(comms, acts, n_comm, n_acts)
    p_ac /= np.sum(p_ac)  # normalize counts into a probability distribution

    # Calculate mutual information
    mutinfo = 0
    for c in range(n_comm):
        for a in range(n_acts):
            if p_ac[c][a] > 0:
                mutinfo += p_ac[c][a] * math.log(p_ac[c][a] / (p_c[c] * p_a[a]))
    return mutinfo


def calc_entropy(comms, n_comm):
    # Calculates the entropy of the communication distribution
    # p(c) is calculated by averaging over episodes
    comms = [U.to_int(m) for m in comms]
    eps = 1e-9

    p_c = U.probs_from_counts(comms, n_comm, eps=eps)
    entropy = 0
    for c in range(n_comm):
        entropy += - p_c[c] * math.log(p_c[c])
    return entropy


def calc_context_indep(acts, comms, n_acts, n_comm):
    # Calculates the context independence (Bogin et al., 2018)
    comms = [U.to_int(m) for m in comms]
    acts = [U.to_int(a) for a in acts]
    eps = 1e-9

    p_a = U.probs_from_counts(acts, n_acts, eps=eps)
    p_c = U.probs_from_counts(comms, n_comm, eps=eps)
    p_ac = U.bin_acts(comms, acts, n_comm, n_acts)
    p_ac /= np.sum(p_ac)

    p_a_c = np.divide(p_ac, np.reshape(p_c, (-1, 1)))
    p_c_a = np.divide(p_ac, np.reshape(p_a, (1, -1)))

    ca = np.argmax(p_a_c, axis=0)
    ci = 0
    for a in range(n_acts):
        ci += p_a_c[ca[a]][a] * p_c_a[ca[a]][a]
    ci /= n_acts
    return ci


""" Calculating causal influence of communication, which requires access to model. 
    CIC calculation done by loading trained agents, not over the course of training. """


def calc_cic(p_a_given_do_c, p_c, n_comm, n_acts):
    # Calculate the one-step causal influence of communication, i.e. the mutual information using p(a | do(c))
    p_ac = p_a_given_do_c * np.expand_dims(p_c, axis=1)  # calculate joint probability p(a, c)
    p_ac /= np.sum(p_ac)  # re-normalize
    p_a = np.mean(p_ac, axis=0)  # compute p(a) by marginalizing over c

    # Calculate mutual information
    cic = 0
    for c in range(n_comm):
        for a in range(n_acts):
            if p_ac[c][a] > 0:
                cic += p_ac[c][a] * math.log(p_ac[c][a] / (p_c[c] * p_a[a]))
    return cic


def get_p_a_given_do_c(agents, env, self=False):
    # Calculates p(a | do(c)) for both agents, i.e. the probability distribution over agent 1's actions given that
    # we intervene at agent 2 to send message c (and vice-versa)
    # If self = True, calculates p(a | do(c)) if we intervene at agent 1 to send message c, i.e. the effect of
    # agent 1's message on its own action (and similarly for agent 2)

    # Cache payoff matrices to ensure they are kept consistent
    payoff_a = env.payoff_mat_a
    payoff_b = env.payoff_mat_b
    p_a_given_do_c = [np.zeros((env.n_comm, env.n_acts)), np.zeros((env.n_comm, env.n_acts))]

    # For both agents
    for ag in range(2):
        # Iterated over this agent's possible messages
        for i in range(env.n_comm):
            _ = env.reset()  # get rid of any existing messages in the observation
            env.payoff_mat_a = payoff_a  # restore payoffs undone by .reset()
            env.payoff_mat_b = payoff_b
            ob_c, _ = env.step_c_single(i, ag)  # intervene in environment with message i
            if self:
                # Calculate p(a|do(c)) of same agent
                logits_c, logits_a, v = agents[ag].forward(torch.Tensor(ob_c[ag]))
            else:
                # Calculate p(a|do(c)) of other agent
                logits_c, logits_a, v = agents[1 - ag].forward(torch.Tensor(ob_c[1 - ag]))

            # Convert logits to probability distribution
            probs_a = F.softmax(logits_a, dim=0)
            p_a_given_do_c[ag][i, :] = probs_a.data.numpy()

    return p_a_given_do_c


def calc_model_cic(model_file_ag1, model_file_ag2, num_games=1000, args=None):
    # Given trained model files in model_file_ag1 and model_file_ag2 (.txt files saved with torch.save), calculates
    # the one-step CIC, averaged over num_games training games
    # args are used to specify MCG game, and structure of the agent

    if args is None:
        args = {
            'n_comm': 4,
            'n_acts': 2,
            'n_hid': 40,
            'mem_size': 0,
            'comm_type': 'turns',
            'game': 'random',
            'seed': 0
        }

    # Instantiate MCG and agents
    env = MCG(n_comm=args.n_comm, game='random', n_acts=args.n_acts, mem_size=args.mem_size)
    ag_kwargs = {
        'n_inp': env.n_obs,
        'n_hid': args.n_hid,
        'n_out': args.n_acts,
        'n_comm': args.n_comm
    }
    agent1, agent2 = M.ReinforceCommAgent(**ag_kwargs), M.ReinforceCommAgent(**ag_kwargs)

    # Load agents from file
    agent1.load_state_dict(torch.load(model_file_ag1))
    agent2.load_state_dict(torch.load(model_file_ag2))
    agents = [agent1, agent2]

    # Iterate over games
    cics = [[], []]
    for i in range(num_games):
        # Get a new game (which is random even if args.game = fixed)
        env.payoffs_a = [3 * np.random.randn(env.n_acts, env.n_acts)]
        env.payoffs_b = [3 * np.random.randn(env.n_acts, env.n_acts)]
        ob_c = env.reset()

        # Calculate p(a | do(c)) for both agents and messages c
        p_a_given_do_c = get_p_a_given_do_c(agents, env, n_comm=args.n_comm)

        # For each agent, calculate the one-step CIC
        for ag in range(2):
            # Calcualte p(c) of other agent (1-ag) by doing a forward pass through network
            logits_c, logits_a, v = agents[1 - ag].forward(torch.Tensor(ob_c[ag]))
            probs_c = F.softmax(logits_c, dim=0).data.numpy()
            cic = calc_cic(p_a_given_do_c[ag], probs_c, env.n_comm, env.n_acts)
            cics[ag].append(cic)

    return cics
