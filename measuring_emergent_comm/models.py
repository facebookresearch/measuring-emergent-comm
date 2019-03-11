"""
models.py

The code in this file provides the classes for reinforcement learning (RL) algorithms that can play
matrix communication games (MCGs).

author: Ryan Lowe
"""

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F
import utils as U


class ReinforceAgent(nn.Module):
    """For comparison: a baseline policy gradient agent that does not have communication"""
    def __init__(self, gamma=0., n_inp=6, n_hid=12, n_out=2, ent_coeff=1e-2):
        super(ReinforceAgent, self).__init__()
        self.hid = nn.Linear(n_inp, n_hid)
        self.out = nn.Linear(n_hid, n_out)
        self.gamma = gamma
        self.n_out = n_out
        self.ent_coeff = ent_coeff

    def forward(self, x):
        x = self.hid(x)
        x = F.relu(x)
        return self.out(x)

    def act(self, x, test=False):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=0)
        if not test:
            return dists.Categorical(probs=probs).sample()
        else:
            return torch.max(logits)[1]

    def update(self, acts, rews, obs, optimizer):
        rews_disc = U.discount_rewards(rews, self.gamma)
        acts = torch.Tensor(acts)
        rews_disc = torch.Tensor(rews_disc)
        obs = torch.Tensor(obs)
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=1)

        # index logprobs by acts
        logprobs = dists.Categorical(probs=probs).log_prob(acts)

        loss = (-logprobs * rews_disc).mean()
        ent_loss = (-probs * torch.log(probs)).sum(dim=1).mean()
        loss -= self.ent_coeff * ent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class ReinforceCommAgent(nn.Module):
    """An agent that uses policy gradient (REINFORCE or A2C) to learn to play MCGs

    Uses a 2-layer MLP policy.
    Has separate action (a) and communication (c) outputs.
    Also contains code for training a separate communication network, and for training networks to predict the
    action of the opponent.
    """
    def __init__(self, gamma=0.9, n_inp=20, n_hid=20, n_out=2, n_comm=2, ent_coeff=1e-2, c_coeff=1e-1, val_coeff=1e-1,
                 n_steps=None, mask_c_learning=False, separate_comm_network=False):
        """Sets variables, and creates layers for networks

        Args:
            gamma:      discount factor
            n_inp:      input/ observation size
            n_hid:      size of the hidden layers
            n_out:      output size (for actions)
            n_comm:     size of communication outputs
            ent_coeff:  coefficient for entropy bonus
            c_coeff:    coefficient for loss associated with communication part of the network
            val_coeff:  coefficient for value learning
            n_steps:    number of steps used to calculate discounted reward.
                        if None, only the next reward is given
            mask_c_learning:        if True, zeros out all learning for communication (i.e. sets c_coeff = 0)
            separate_comm_network:  if True, a separate network is used for communication
        """

        super(ReinforceCommAgent, self).__init__()

        # Main network
        self.h1 = nn.Linear(n_inp, n_hid)
        self.h2 = nn.Linear(n_hid, n_hid)
        self.out_a = nn.Linear(n_hid, n_out)
        self.out_c = nn.Linear(n_hid, n_comm)
        self.v = nn.Linear(n_hid, 1)

        # Creates a separate network for communication outputs if necessary
        self.separate_comm_network = separate_comm_network
        if self.separate_comm_network:
            self.h1_c = nn.Linear(n_inp, n_hid)
            self.h2_c = nn.Linear(n_hid, n_hid)

        # Hidden layers for action prediction networks
        self.h1_apred_sep = nn.Linear(n_inp, n_hid)
        self.h2_apred_sep = nn.Linear(n_hid, n_hid)
        self.h1_apred_nocommsep = nn.Linear(n_inp, n_hid)
        self.h2_apred_nocommsep = nn.Linear(n_hid, n_hid)
        self.out_apred = nn.Linear(n_hid, n_out)
        self.out_apred_nocomm = nn.Linear(n_hid, n_out)
        self.apred_noinps = nn.Parameter(torch.zeros(n_out))
        self.out_apred_sep = nn.Linear(n_hid, n_out)
        self.out_apred_nocommsep = nn.Linear(n_hid, n_out)

        # Other properties and coefficients
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.c_coeff = c_coeff
        self.val_coeff = val_coeff
        self.mask_c_learning = 0 if mask_c_learning else 1
        self.n_comm = n_comm
        self.n_steps = n_steps

    def forward(self, x, apred="", input_is_batch=False):
        # Computes a forward pass through the network.
        # If apred == 'nocomm', 'sep', 'nocommsep', 'inps' or 'noinps', output will be the predict opponent's action
        # Otherwise, output will be the message c, action a, and value v
        # input_is_batch flag is True when a batch of data is passed in for updating parameters (vs. for acting)
        # This only matters when apred != "", as it is used to determine how to mask the communication

        # First, make predictions about opponent's action
        if apred == "nocomm":
            # There is no communication as input
            if input_is_batch:
                x[:, -2 * self.n_comm:] = 0  # mask the communication before feeding it through main network
            else:
                x[-2 * self.n_comm:] = 0
        if apred == "sep":
            # There is a separate network for predicting opponent's action
            x = self.h1_apred_sep(x)
            x = F.relu(x)
            x = self.h2_apred_sep(x)
            x = F.relu(x)
            return self.out_apred_sep(x)
        if apred == "nocommsep":
            # There is a separate network for predicting opponent's action, and there is no communication as input
            if input_is_batch:
                x[:, -2 * self.n_comm:] = 0  # mask the communication
            else:
                x[-2 * self.n_comm:] = 0
            x = self.h1_apred_nocommsep(x)
            x = F.relu(x)
            x = self.h2_apred_nocommsep(x)
            x = F.relu(x)
            return self.out_apred_nocommsep(x)

        # If a separate network is used for communication, calculate output communication
        if self.separate_comm_network:
            # Comm network
            y = self.h1_c(x)
            y = F.relu(y)
            y = self.h2_c(y)
            y = F.relu(y)
            c = self.out_c(y)

        # Main network, used to compute action a and value v
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        a = self.out_a(x)
        v = self.v(x)

        if not self.separate_comm_network:
            c = self.out_c(x)

        # Output appropriate action predictions if necessary
        if apred == "inps":
            # Prediction is made using linear layer after main network
            return self.out_apred(x)
        elif apred == "noinps":
            # Prediction is made using no network or inputs at all
            return self.apred_noinps
        elif apred == "nocomm":
            # Prediction is made using linear layer after main network, after communication was masked
            return self.out_apred_nocomm(x)

        return c, a, v

    def act(self, x, mode='', apred=""):
        # Returns a discrete sample from the main network
        # If apred is not an empty string, then output is the prediction of the opponent's actoin
        # If mode == 'comm', then output is a message
        # If mode == 'act', then output is an action

        if apred != "":
            logits_apred = self.forward(x, apred)
            return dists.Categorical(logits=logits_apred).sample()

        logits_c, logits_a, v = self.forward(x)
        if mode == 'comm':
            return dists.Categorical(logits=logits_c).sample()
        if mode == 'act':
            return dists.Categorical(logits=logits_a).sample()

    def update_apred(self, obs_a, acts, optimizer, apred="inps"):
        # Updates the action prediction networks
        obs_a = torch.Tensor(obs_a)
        acts = torch.Tensor(acts)

        # Calculate forward pass through model
        logits_apred = self.forward(obs_a, apred=apred, input_is_batch=True)
        # If predictions made without inputs, then no forward pass is needed
        if apred == "noinps":
            logits_apred = torch.stack(tuple([logits_apred] * obs_a.shape[0]))

        # Calculate loss, perform update
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_apred, acts.type(torch.LongTensor))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update(self, comms, acts, rews_a, obs_a, optimizer, obs_c=None, rews_c=None):
        # Updates main network
        # comms, acts are the lists of messages / actions actually taken
        # obs_a, rews_a: observations and rewards used when the network outputs an action
        # obs_c, rews_c: observations and rewards used when the network outputs an message
        # If no obs_c / rews_c are provided, they are assumed to be the same as obs_a / rews_a
        if obs_c is None:
            obs_c = obs_a
        if rews_c is None:
            rews_c = rews_a

        # Calculated discounted reward
        rews_disc_a = torch.Tensor(U.discount_rewards(rews_a, self.gamma, self.n_steps))
        rews_disc_c = torch.Tensor(U.discount_rewards(rews_c, self.gamma, self.n_steps))

        # Pass observations through main network, obtain probability distributions over actions/ messages
        comms, acts = torch.Tensor(comms), torch.Tensor(acts)
        obs_a, obs_c = torch.Tensor(obs_a), torch.Tensor(obs_c)
        # Note: V_c and V_a may be different if obs_c and obs_a are different, but both use the same value network
        logits_c, _, V_c = self.forward(obs_c)
        _, logits_a, V_a = self.forward(obs_a)
        probs_a = F.softmax(logits_a, dim=1)
        probs_c = F.softmax(logits_c, dim=1)

        # Convert to log probabilities, index by actions/ messages taken
        logprobs_a = dists.Categorical(probs=probs_a).log_prob(acts)
        logprobs_c = dists.Categorical(probs=probs_c).log_prob(comms)

        # Clamp probabilities to avoid NaNs when computing entropy bonus
        probs_a = torch.clamp(probs_a, 1e-6, 1)
        probs_c = torch.clamp(probs_c, 1e-6, 1)

        # Calculate losses using policy gradients
        loss_a = (-logprobs_a * (rews_disc_a - V_a)).mean()
        loss_c = (-logprobs_c * (rews_disc_c - V_c)).mean()
        # Calculate entropy bonuses
        loss_ent_a = (-probs_a * torch.log(probs_a)).sum(dim=1).mean()
        loss_ent_c = (-probs_c * torch.log(probs_c)).sum(dim=1).mean()
        # Calculate value function losses (MSE)
        loss_v_a = ((rews_disc_a - V_a) ** 2).mean()
        loss_v_c = ((rews_disc_c - V_c) ** 2).mean()

        # Total loss is the weighted sum of all previous losses.
        # If mask_c_learning is true, then there is no loss for the communication output
        loss = loss_a + self.c_coeff * self.mask_c_learning * loss_c \
            - self.ent_coeff * (loss_ent_a + self.mask_c_learning * loss_ent_c) \
            + self.val_coeff * (loss_v_a + self.mask_c_learning * loss_v_c)

        # Take a gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
