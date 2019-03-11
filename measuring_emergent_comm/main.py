"""
main.py

The code in this file runs the matrix communication games (MCGs), used in the paper:
    "On the Pitfalls of Measuring Emergent Communication",
    Ryan Lowe, Jakob Foerster, Y-Lan Boureau, Joelle Pineau, Yann Dauphin
    AAMAS 2019

author: Ryan Lowe
"""

import torch
import numpy as np
import models as M
import utils as U
from mcg import MCG
import eval
import argparse
import time

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def parse_args():
    parser = argparse.ArgumentParser("Experiments in matrix communication games (MCGs)")
    # Properties of the game
    parser.add_argument("--n-comm", type=int, default=4, help="size of agent vocabulary")
    parser.add_argument("--game", type=str, default="random", help="type of coordination game being played")
    parser.add_argument("--comm-type", type=str, default="turns", help="type of communication")
    parser.add_argument("--n-acts", type=int, default=2, help="size of matrices being played")
    parser.add_argument("--mem-size", type=int, default=0, help="size of memory (no. of prev. comms observed)")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--ent-coeff", type=float, default=1e-2, help="size of the entropy coefficient")
    parser.add_argument("--val-coeff", type=float, default=1e-1, help="size of the value coefficient")
    parser.add_argument("--c-coeff", type=float, default=1e-1, help="size of the communication coefficient")
    parser.add_argument("--gamma", type=float, default=0., help="discount factor")
    parser.add_argument("--n-hid", type=int, default=40, help="number of hidden units in the mlp. Note, to set this manually, also make set-manual-nhid=True")
    parser.add_argument("--set-manual-n-hid", action="store_true", default=False, help="if False, n-hid is scaled automatically")
    parser.add_argument("--nsteps", type=int, default=1, help="number of reward steps in TD")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--policy", type=str, default="REINFORCE", help="type of policy that is being used")
    parser.add_argument("--adv-coeff", type=float, default=0., help="degree of agent opposition")
    parser.add_argument("--mask-c-learning", action="store_true", default=False, help="if True, c_out doesn't learn")
    parser.add_argument("--separate-comm-network", action="store_true", default=False, help="if True, use a separate network for comms")
    # Checkpointing and saving
    parser.add_argument("--verbose", action="store_true", default=False, help="prints out more info during training")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--print-freq", type=int, default=1000, help="how frequently data is saved/ printed")
    parser.add_argument("--save-freq", type=int, default=None, help="how frequently model is saved")
    parser.add_argument("--log-saving", action="store_true", default=False, help="save params on log-ish scale")
    parser.add_argument("--num-episodes", type=int, default=250000, help="number of episodes")
    parser.add_argument("--save-dir", type=str, default="./policy/", help="directory where model and results are saved")
    parser.add_argument("--load-dir", type=str, default="", help="directory where model is loaded from")
    parser.add_argument("--exp-name", type=str, default=None, help="type of coordination game being played")
    return parser.parse_args()


def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Scale communication channel and MLP size based on size of game (unless set_manual_nhid = True)
    args.n_comm = args.n_acts + 2 if args.comm_type != "none" else 1
    if not args.set_manual_n_hid:
        args.n_hid = 20 + 10 * args.n_acts

    # Instantiate MCG class
    env = MCG(n_comm=args.n_comm, game=args.game, n_acts=args.n_acts, mem_size=args.mem_size, adv_coeff=args.adv_coeff)

    # Create policy
    if args.policy == 'REINFORCE':
        args.nsteps = None  # A2C has args.nsteps = 5
    ag_kwargs = {'gamma': args.gamma, 'n_inp': env.n_obs, 'n_hid': args.n_hid,
                 'n_out': args.n_acts, 'n_comm': args.n_comm, 'c_coeff': args.c_coeff,
                 'ent_coeff': args.ent_coeff, 'n_steps': args.nsteps,
                 'mask_c_learning': args.mask_c_learning, 'separate_comm_network': args.separate_comm_network}
    agent = [M.ReinforceCommAgent(**ag_kwargs), M.ReinforceCommAgent(**ag_kwargs)]

    # Get initial state
    ob_c = env.reset()

    # Accumulate intermediate values into lists
    # Actions, rewards, and observations
    rews, comms, acts, obs_c, obs_a, = [[], []], [[], []], [[], []], [[], []], [[], []]
    # Predicted actions
    apreds, apreds_noinps, apreds_nocomm, apreds_sep = [[], []], [[], []], [[], []], [[], []]
    apreds_nocommsep = [[], []]
    # Accuracy of predicted actions
    apred_acc, apred_noinps_acc, apred_nocomm_acc = [[], []], [[], []], [[], []]
    apred_sep_acc, apred_nocommsep_acc = [[], []], [[], []]
    # Context independence, mutual information, average reward
    ic, avg_rews, sc = [[], []], [[], []], [[], []]
    # L2 norm of policy parameters
    param_comm_l2, param_mat_l2 = [[], []], [[], []]
    # Matrix of (message, action) co-occurrences averaged over episodes
    stats = [None, None]
    # Keep track of these values across all episodes (not reset every print_freq), to be written to file
    sc_save, rews_save,  apred_acc_save, apred_noinps_acc_save = [[], []], [[], []], [[], []], [[], []]
    apred_nocomm_acc_save, apred_sep_acc_save, apred_nocommsep_acc_save = [[], []], [[], []], [[], []]
    iter_save, stats_save, ic_save = [], [[], []], [[], []]

    args.save_freq = args.num_episodes if args.save_freq is None else args.save_freq

    # Initialize optimizer
    optimizer = [torch.optim.Adam(ag.parameters(), args.lr) for ag in agent]

    # Use a different optimizer for the prediction of the other agent's acions
    optimizer_apred = [torch.optim.Adam(U.get_named_params(ag, 'out_apred'), args.lr*5) for ag in agent]
    optimizer_apred_noinps = [torch.optim.Adam(U.get_named_params(ag, 'apred_noinps'), args.lr*5) for ag in agent]
    optimizer_apred_nocomm = [torch.optim.Adam(U.get_named_params(ag, 'apred_nocomm'), args.lr*5) for ag in agent]
    optimizer_apred_sep = [torch.optim.Adam(U.get_named_params(ag, 'sep'), args.lr*5) for ag in agent]
    optimizer_apred_nocommsep = [torch.optim.Adam(U.get_named_params(ag, 'nocommsep'), args.lr*5) for ag in agent]

    # Create name that will be used to save files
    if args.exp_name is None:
        args.exp_name = f'coord_nc={args.n_comm}_gam={args.game}_com={args.comm_type}_na={args.n_acts}'
        args.exp_name += f'_mem={args.mem_size}_lr={args.lr}_entc={args.ent_coeff}_valc={args.val_coeff}'
        args.exp_name += f'_cc={args.c_coeff}_gnc={args.gn_coeff}_gam={args.gamma}_nh={args.n_hid}'
        args.exp_name += f'_nste={args.nsteps}_pi={args.policy}'
        args.exp_name += f'_bs={args.batch_size}_nep={args.num_episodes}_man_nh={args.set_manual_n_hid}'
        args.exp_name += f'_hnoise={args.hnoise_sigma}_advc={args.adv_coeff}_maskc={args.mask_c_learning}'
        args.exp_name += f'_sepc={args.separate_comm_network}'
        args.exp_name += f'_seed={args.seed}'

    last_time = time.time()

    # Main training loop
    for i in range(args.num_episodes):

        # Agents act in the environment
        with torch.no_grad():
            if args.comm_type == "turns" or args.comm_type == "garble":
                # Agents take turns communicating
                j = np.random.randint(2)  # randomize which agent goes first
                c1 = agent[j].act(torch.Tensor(ob_c[j]), mode='comm')
                ob_c2 = env.step_c_single(c1, j, garble=args.comm_type == "garble")
                c2 = agent[1-j].act(torch.Tensor(ob_c2[1-j]), mode='comm')
                ob_a = env.step_c_single(c2, 1-j, garble=args.comm_type == "garble")
                # fix agent obs/ comms lists for correct updating
                c = [c1*(1-j) + c2*j, c1*j + c2*(1-j)]
                ob_c = [ob_c[0]*(1-j) + ob_c2[0]*j, ob_c[1]*j + ob_c2[1]*(1-j)]
            elif args.comm_type == "oneag" or args.comm_type == "none":
                # Only one agent communicates, or neither does
                c1 = agent[0].act(torch.Tensor(ob_c[0]), mode='comm')
                ob_a = env.step_c_single(c1, 0)
                c = [c1, 0]
            elif args.comm_type == "simul":
                # Agents communicate simultaneously
                c = [ag.act(torch.Tensor(o), mode='comm') for o, ag in zip(ob_c, agent)]
                ob_a = env.step_c(*c)
            elif args.comm_type == "random":
                # A random message is sent by each agent
                c = [np.random.randint(args.n_comm), np.random.randint(args.n_comm)]
                ob_a = env.step_c(*c)

            # Produce action
            a = [ag.act(torch.Tensor(o), mode='act') for o, ag in zip(ob_a, agent)]

            # Predict the action of the opponent
            apred = [ag.act(torch.Tensor(o), mode='act', apred="inps") for o, ag in zip(ob_a, agent)]
            apred_noinps = [ag.act(torch.Tensor(o), mode='act', apred="noinps") for o, ag in zip(ob_a, agent)]
            apred_nocomm = [ag.act(torch.Tensor(o), mode='act', apred="nocomm") for o, ag in zip(ob_a, agent)]
            apred_sep = [ag.act(torch.Tensor(o), mode='act', apred="sep") for o, ag in zip(ob_a, agent)]
            apred_nocommsep = [ag.act(torch.Tensor(o), mode='act', apred="nocommsep") for o, ag in zip(ob_a, agent)]

            # Take actions, update observations and observe reward
            ob_c_new, rew = env.step_a(*a, new_mats=True)

        # Keep track of actions, observations for each agent
        for j in range(2):
            comms[j].append(c[j])
            acts[j].append(a[j])
            obs_c[j].append(ob_c[j])
            obs_a[j].append(ob_a[j])
            rews[j].append(rew[j])
            apreds[j].append(apred[j])
            apreds_noinps[j].append(apred_noinps[j])
            apreds_nocomm[j].append(apred_nocomm[j])
            apreds_sep[j].append(apred_sep[j])
            apreds_nocommsep[j].append(apred_nocommsep[j])

        ob_c = ob_c_new

        # Every batch_size iterations, perform an update
        if len(acts[0]) == args.batch_size:
            for j in range(2):
                # Update agent policy
                agent[j].update(comms[j], acts[j],  rews[j], obs_a[j], optimizer[j], obs_c=obs_c[j])

                # Update action prediction policies
                agent[j].update_apred(obs_a[j], acts[1 - j], optimizer_apred[j])
                agent[j].update_apred(obs_a[j], acts[1 - j], optimizer_apred_noinps[j], apred="noinps")
                agent[j].update_apred(obs_a[j], acts[1 - j], optimizer_apred_nocomm[j], apred="nocomm")
                agent[j].update_apred(obs_a[j], acts[1 - j], optimizer_apred_sep[j], apred="sep")
                agent[j].update_apred(obs_a[j], acts[1 - j], optimizer_apred_nocommsep[j], apred="nocommsep")

                # Accumulate statistics
                avg_rews[j].append(np.mean(rews[j]))
                stats[j] = U.calc_stats(comms[j], acts[j], args.n_comm, args.n_acts, stats[j])
                sc[j].append(eval.calc_mutinfo(acts[j], comms[j], args.n_acts, args.n_comm))
                ic[j].append(eval.calc_mutinfo(acts[1 - j], comms[j], args.n_acts, args.n_comm))

                # Calculate action prediction accuracies for different inputs
                # When predicting from last hidden state of policy network, with full state info
                apred_acc[j].append(np.mean([ap == ac for ap, ac in zip(apreds[j], acts[1-j])]))
                # No input given
                apred_noinps_acc[j].append(np.mean([ap == ac for ap, ac in zip(apreds_noinps[j], acts[1 - j])]))
                # When predicting from last hidden state of policy network, with no communication info
                apred_nocomm_acc[j].append(np.mean([ap == ac for ap, ac in zip(apreds_nocomm[j], acts[1 - j])]))
                # When using a separate network to predict, with full state info
                apred_sep_acc[j].append(np.mean([ap == ac for ap, ac in zip(apreds_sep[j], acts[1 - j])]))
                # When using a separate network to predict, with no communication info
                apred_nocommsep_acc[j].append(np.mean([ap == ac for ap, ac in zip(apreds_nocommsep[j], acts[1 - j])]))

                # Calculate L2 norm of the communication and non-communication parts of network
                param = U.get_named_params(agent[j], "h1")[0].data.numpy().T
                param_comm = param[-2 * env.n_comm:]
                param_mat = param[:-2 * env.n_comm]
                param_comm_l2[j].append(np.linalg.norm(param_comm))
                param_mat_l2[j].append(np.linalg.norm(param_mat))

            rews, comms, acts, obs_c, obs_a, apreds = [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]
            apreds_noinps, apreds_nocomm, apreds_sep, apreds_nocommsep = [[], []], [[], []], [[], []], [[], []]

        # Display values and append them to counters
        if (i + 1) % args.print_freq == 0:
            print_string = f'Iteration {i},\t\n ' \
                           f'Ag1 SC: {np.mean(sc[0]):.{4}},\t Ag2 SC: {np.mean(sc[1]):.{4}}\n' \
                           f'Ag1 Reward: {np.mean(avg_rews[0]):.{4}},\t Ag2 Reward: {np.mean(avg_rews[1]):.{4}}\n'
            if args.verbose:
                print_string += f'Ag1 IC: {np.mean(ic[0]):.{4}},\t  Ag2 IC: {np.mean(ic[1]):.{4}}\n' \
                                f'Ag1 action pred%: {np.mean(apred_acc[0]):.{4}},' \
                                f'\t Ag2 action pred%: {np.mean(apred_acc[1]):.{4}}\n' \
                                f'Ag1 action pred (noinp)%: {np.mean(apred_noinps_acc[0]):.{4}},' \
                                f'\t Ag2 action pred (noinp)%: {np.mean(apred_noinps_acc[1]):.{4}}\n' \
                                f'Ag1 action pred (nocomm)%: {np.mean(apred_nocomm_acc[0]):.{4}},' \
                                f'\t Ag2 action pred (nocomm)%: {np.mean(apred_nocomm_acc[1]):.{4}}\n' \
                                f'Ag1 action pred (sep)%: {np.mean(apred_sep_acc[0]):.{4}},' \
                                f'\t Ag2 action pred (sep)%: {np.mean(apred_sep_acc[1]):.{4}}\n' \
                                f'Ag1 action pred (nocommsep)%: {np.mean(apred_nocommsep_acc[0]):.{4}},' \
                                f'\t Ag2 action pred (nocommsep)%: {np.mean(apred_nocommsep_acc[1]):.{4}}\n' \
                                f'Ag1 param_comm norm: {np.mean(param_comm_l2[0]):.{4}},' \
                                f'\t Ag2 param_comm norm: {np.mean(param_comm_l2[1]):.{4}}\n' \
                                f'Ag1 param_mat norm: {np.mean(param_mat_l2[0]):.{4}},' \
                                f'\t Ag2 param_mat norm: {np.mean(param_mat_l2[1]):.{4}}\n' \
                                f'\t Ag1 (message, action) co-occurrences:\n {stats[0]} \n' \
                                f'\t Ag2 (message, action) co-occurrences:\n {stats[1]}\n'
            print(print_string)
            print(f'This batch took {time.time() - last_time:.{3}}s\n')
            last_time = time.time()

            # Append values to be saved
            for j in range(2):
                sc_save[j].append(np.mean(sc[j]))
                ic_save[j].append(np.mean(ic[j]))
                rews_save[j].append(np.mean(avg_rews[j]))
                apred_acc_save[j].append(np.mean(apred_acc[j]))
                apred_noinps_acc_save[j].append(np.mean(apred_noinps_acc[j]))
                apred_nocomm_acc_save[j].append(np.mean(apred_nocomm_acc[j]))
                apred_sep_acc_save[j].append(np.mean(apred_sep_acc[j]))
                apred_nocommsep_acc_save[j].append(np.mean(apred_nocommsep_acc[j]))
                stats_save[j].append(stats[j])
            iter_save.append(i+1)

            # Reset counters
            avg_rews, sc, ic = [[], []], [[], []], [[], []]
            apred_acc, apred_noinps_acc, apred_nocomm_acc = [[], []], [[], []], [[], []]
            apred_nocommsep_acc, apred_sep_acc = [[], []], [[], []]
            stats = [None, None]

        # Save values to file
        if i == args.num_episodes - 1 and args.exp_name is not None:
            data = {
                'Iterations': iter_save,
                'Ag1 SC': sc_save[0],
                'Ag2 SC': sc_save[1],
                'Ag1 Rew': rews_save[0],
                'Ag2 Rew': rews_save[1],
                'Ag1 Action prediction, from policy network, all state info': apred_acc_save[0],
                'Ag2 Action prediction, from policy network, all state info': apred_acc_save[1],
                'Ag1 Action prediction, no inputs': apred_noinps_acc_save[0],
                'Ag2 Action prediction, no inputs': apred_noinps_acc_save[1],
                'Ag1 Action prediction, from policy network, no communication': apred_nocomm_acc_save[0],
                'Ag2 Action prediction, from policy network, no communication': apred_nocomm_acc_save[1],
                'Ag1 Action prediction, separate network, all state info': apred_sep_acc_save[0],
                'Ag2 Action prediction, separate network, all state info': apred_sep_acc_save[1],
                'Ag1 Action prediction, separate network, no communication': apred_sep_acc_save[0],
                'Ag2 Action prediction, separate network, no communication': apred_sep_acc_save[1]
            }
            header = [key for key in data]
            U.save_data(args, header, data)

        # Save model
        logsave_list = [1000, 5000, 10000, 50000, 250000]
        if (i + 2) % args.save_freq == 0 or (args.log_saving and any([(i + 2) == k for k in logsave_list])):
            U.save_model(args, agent, i+2)


if __name__ == "__main__":
    args = parse_args()
    train(args)
