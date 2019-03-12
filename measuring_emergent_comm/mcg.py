""" Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

mcg.py

The code in this file contains the definition of matrix communication games (MCGs).

author: Ryan Lowe
"""

import numpy as np
import utils as U
from collections import deque


class MCG(object):
    """CLass for creating matrix communication games (MCGs)

    You can define a set of payoffs to be sampled from, or have them sampled randomly at each iteration.
    You can also set the agents to be purely cooperative (adv_coeff = -1) or purely competitive (adv_coeff = 1)
    Agents' memory is kept as a part of this class, as it is given as observation
    """

    def __init__(self, n_comm=2, game=None, n_acts=2, mem_size=0, adv_coeff=0, payoffs_a=None, payoffs_b=None):
        """
        Initializes variables and returns initial observation.

        Args:
            n_comm:     size of communication channel
            game:       string, determines the structure of the MCG payoffs. can be 'fixed' (payoffs sampled from
                        a fixed set), 'randomfixed' (payoffs randomly generated but fixed), or 'random' (payoffs
                        generated randomly at each time step
            n_acts:     numnber of actions each agent can take (i.e. size of the payoff matrix)
            mem_size:   number of previous actions and messages that are given
            adv_coeff:  0 default. if 1, agents are purely competitive. if -1, agents are purely cooperative
            payoffs_a:  list of payoff matrices to be sampled from for agent 1, if game == 'fixed'
            payoffs_b:  same, for agent 2
        """
        # Initialize payoff matrices based on game type
        if game == 'fixed':
            # Payoff matrices are fixed over time, and specified by the user
            assert payoffs_a is not None and payoffs_b is not None
            self.rew_mats_a = payoffs_a
            self.rew_mats_b = payoffs_b
            self.n_acts = self.rew_mats_a[0].shape[0]
            self.random = False
        elif game == 'randomfixed':
            # Payoff matrices are fixed over time, and randomly generated
            self.n_acts = n_acts
            self.rew_mats_a = [3 * np.random.randn(n_acts, n_acts)]
            self.rew_mats_b = [3 * np.random.randn(n_acts, n_acts)]
            self.random = False
        elif game == 'random':
            # Payoff matrices are randomly generated at every time step
            self.n_acts = n_acts
            self.random = True
        else:
            raise ValueError('Invalid game argument provided')
        self.payoff_mat_a = None
        self.payoff_mat_b = None

        # Initialize passed-in values
        self.n_comm = n_comm
        self.mem_size = mem_size
        self.adv_coeff = adv_coeff
        self.comm_a, self.comm_b = np.zeros((self.n_comm,)), np.zeros((self.n_comm,))
        self.act_a, self.act_b = np.zeros((self.n_acts,)), np.zeros((self.n_acts,))

        # Provide placeholders for pausing env
        self.payoff_mat_a_hold, self.payoff_mat_b_hold, self.comm_a_hold, self.comm_b_hold = None, None, None, None
        self.mem_hold = None

        # Calculate observation size
        self.n_obs = 2 * (self.n_acts ** 2) + self.n_comm * 2 + (self.n_comm + self.n_acts) * 2 * self.mem_size

        # Initialize agent memory
        self.mem = deque([np.zeros(2 * (self.n_comm + self.n_acts))] * self.mem_size, self.mem_size)

    def build_state(self, new_mats=True):
        # Constructs the state, based on the payoff matrices, memory, and agent communication that round
        # If new_mats = True, samples a new payoff matrix, depending on the game type

        if self.random and new_mats:
            # Sample a new payoff matrix from a Gaussian N(0, 3) distribution
            self.payoff_mat_a = 3 * np.random.randn(self.n_acts, self.n_acts)
            self.payoff_mat_b = 3 * np.random.randn(self.n_acts, self.n_acts)
        elif new_mats:
            # Select a payoff matrix from the list of payoffs in rew_mats
            rand_mat = np.random.randint(len(self.rew_mats_a))
            self.payoff_mat_a = np.choose(rand_mat, self.rew_mats_a)
            self.payoff_mat_b = np.choose(rand_mat, self.rew_mats_b)

        # Adjust payoffs, depending on the value of adv_coeff
        # i.e. whether the agents are cooperative (-1) or competitive (1) or neither (0)
        payoff_a_adjust = - self.adv_coeff * self.payoff_mat_b
        payoff_b_adjust = - self.adv_coeff * self.payoff_mat_a
        self.payoff_mat_a += payoff_a_adjust
        self.payoff_mat_b += payoff_b_adjust

        return [np.concatenate([*list(self.mem), self.payoff_mat_a.flatten(), self.payoff_mat_b.flatten(),
                                self.comm_a, self.comm_b])]*2

    def step_c(self, comm_a, comm_b):
        # Update the state based on both agent's communications
        self.comm_a = U.index_to_onehot(comm_a, v_len=self.n_comm)
        self.comm_b = U.index_to_onehot(comm_b, v_len=self.n_comm)
        return self.build_state(new_mats=False)

    def step_c_single(self, comm, ag, garble=False):
        # Update the state based on a single agent's communication
        if ag == 0:
            self.comm_a = U.index_to_onehot(comm, v_len=self.n_comm)
        elif ag == 1:
            self.comm_b = U.index_to_onehot(comm, v_len=self.n_comm)

        # If garble is True, the state is updated with a random message instead
        if garble:
            if ag == 0:
                self.comm_a = U.index_to_onehot(np.random.randint(self.n_comm), v_len=self.n_comm)
            elif ag == 1:
                self.comm_b = U.index_to_onehot(np.random.randint(self.n_comm), v_len=self.n_comm)

        return self.build_state(new_mats=False)

    def step_a(self, act_a, act_b, new_mats=True):
        # Update the state based on both agent's actions
        self.act_a, self.act_b = U.index_to_onehot(act_a, v_len=self.n_acts), U.index_to_onehot(act_b, v_len=self.n_acts)
        act_a, act_b = act_a.data.numpy(), act_b.data.numpy()
        rew_a = self.payoff_mat_a[act_a, act_b]
        rew_b = self.payoff_mat_b[act_a, act_b]

        # Add comms, acts to memory
        self.mem.append(np.concatenate([self.comm_a, self.comm_b, self.act_a, self.act_b]))
        self.comm_a, self.comm_b = np.zeros((self.n_comm,)), np.zeros((self.n_comm,))
        self.act_a, self.act_b = np.zeros((self.n_acts,)), np.zeros((self.n_acts,))
        return self.build_state(new_mats=new_mats), [rew_a, rew_b]

    def pause_state(self):
        # Pauses the state of the game
        self.mem_hold = self.mem
        self.comm_a_hold, self.comm_b_hold = self.comm_a, self.comm_b
        self.payoff_mat_a_hold, self.payoff_mat_b_hold = self.payoff_mat_a, self.payoff_mat_b

    def resume_state(self):
        # Resumes the state of the game
        self.mem = self.mem_hold
        self.comm_a, self.comm_b = self.comm_a_hold, self.comm_b_hold
        self.payoff_mat_a, self.payoff_mat_b = self.payoff_mat_a_hold, self.payoff_mat_b_hold

    def reset(self):
        # Resets the environment to its initial state, along with the memory
        self.mem = deque([np.zeros(2 * (self.n_comm + self.n_acts))] * self.mem_size, self.mem_size)
        self.comm_a, self.comm_b = np.zeros((self.n_comm,)), np.zeros((self.n_comm,))
        return self.build_state()
