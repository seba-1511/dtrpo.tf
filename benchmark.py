#!/usr/bin/env python

import gym
import numpy as np
import randopt as ro

from random import random
from time import time
from mpi4py import MPI
from argparse import ArgumentParser
from trpo import TRPO
from utils import FCNet, numel

# Parse arguments
parser = ArgumentParser()
parser.add_argument('--expname', '-e', dest='exp', metavar='e', type=str,
        required=True, help='Name of the experiment to be run.')
parser.add_argument('--env', dest='env', type=str,
        required=True, help='Name of the environment to learn.')
parser.add_argument('--filter', dest='filter', type=bool,
        default=True, help='Whether to filter the environment\'s states.')
parser.add_argument('--solved', dest='solved', type=float,
        default=np.inf, help='Threshold at which the environment is considered solved.')
parser.add_argument('--n_iter', dest='n_iter', type=int,
        default=300, help='Number of updates to be performed.')
parser.add_argument('--timesteps_per_batch', dest='timesteps_per_batch', type=int,
        default=15000, help='Number of steps before updating parameters.')
parser.add_argument('--max_path_length', dest='max_path_length', type=int,
        default=5000, help='Max length for a trajectory/episode.')
parser.add_argument('--momentum', dest='momentum', type=float,
        default=0.0, help='Default momentum value.')
parser.add_argument('--gae', dest='gae', type=bool,
        default=True, help='Whether to use GAE.')
parser.add_argument('--gae_lam', dest='gae_lam', type=float,
        default=0.97, help='Lambda value for GAE.')
parser.add_argument('--delta', dest='delta', type=float,
        default=0.01, help='Max KL divergence for TRPO')
parser.add_argument('--cg_damping', dest='cg_damping', type=float,
        default=0.1, help='Damping used to make CG positive def.')
parser.add_argument('--gamma', dest='gamma', type=float,
        default=0.99, help='Discount factor.')
parser.add_argument('--record', dest='record', type=bool,
        default=False, help='Whether to record videos at test time.')


class Filter:

    def __init__(self, filter_mean=True):
        self.m1 = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def __call__(self, o):
        self.m1 = self.m1 * (self.n / (self.n + 1)) + o * 1 / (1 + self.n)
        self.v = self.v * \
                (self.n / (self.n + 1)) + (o - self.m1)**2 * 1 / (1 + self.n)
        self.std = (self.v + 1e-6)**.5  # std
        self.n += 1
        if self.filter_mean:
            o1 = (o - self.m1) / self.std
        else:
            o1 = o / self.std
        o1 = (o1 > 10) * 10 + (o1 < -10) * (-10) + (o1 < 10) * (o1 > -10) * o1
        return o1


# Define constants
f = Filter()
rank = MPI.COMM_WORLD.Get_rank()

if __name__ == '__main__':
    args = parser.parse_args()

    # Setup Experiment
    exp_name = args.exp + '_' + args.env
    exp = ro.Experiment(exp_name, {
        'exp': ro.Constant(args.exp),
        'env': ro.Constant(args.env),
        'filter': ro.Constant(args.filter),
        'solved': ro.Constant(args.solved),
        'n_iter': ro.Constant(args.n_iter),
        'timesteps_per_batch': ro.Constant(args.timesteps_per_batch),
        'max_path_length': ro.Constant(args.max_path_length),
        'momentum': ro.Constant(args.momentum),
        'gae': ro.Constant(args.gae),
        'gae_lam': ro.Constant(args.gae_lam),
        'delta': ro.Constant(args.delta),
        'cg_damping': ro.Constant(args.cg_damping),
        'gamma': ro.Constant(args.gamma),
    })

    # Define the environment
    env = gym.make(args.env)
    env.seed(1234)
    env.solved_threshold = args.solved

    # Instantiate the agent
    policy = FCNet(numel(env.observation_space), numel(env.action_space))
    agent = TRPO(env=env, policy=policy, optimizer=None, delta=args.delta,
                 gamma=args.gamma, update_freq=args.timesteps_per_batch, 
                 gae=args.gae, gae_lam=args.gae_lam, cg_damping=args.cg_damping)

    # Start training phase
    monitor_path = '/tmp/' + exp_name + str(random())
    if rank == 0:
        env.monitor.start(monitor_path)
    training_start = time()
    while agent.n_iterations < args.n_iter:
        state = env.reset()
        if args.filter:
            state = f(state)
        for path in xrange(args.max_path_length):
            action, action_info = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if args.filter:
                next_state = f(next_state)
            agent.learn(state, action, reward, next_state, done, action_info)
            if done or agent.done():
                break
            state = next_state
        agent.new_episode(done)
        if agent.done():
            break
    training_end = time()

    # Upload results
    if rank == 0:
        env.monitor.close()
        gym.upload(monitor_path, algorithm_id='dtrpo-' + args.exp)

    # Start Testing Phase
    test_n_iter = 100
    if args.record and rank == 0:
        env.monitor.start('./videos/' + args.env + str(time()) + '/')
    test_rewards = 0
    test_start = time()
    for iteration in xrange(test_n_iter):
        state = env.reset()
        if args.filter:
            state = f(state)
        while True:
            action, _ = agent.act(state)
            state, reward, done, _ = env.step(action)
            if args.filter:
                state = f(state)
            test_rewards += reward
            if done:
                break
    test_end = time()
    if args.record and rank == 0:
        env.monitor.close()

    # Save experiment run
    if rank == 0:
        res_data = {
                'test_n_iter': test_n_iter,
                'test_reward': test_rewards,
                'test_timing': test_end - test_start, 
                'training_timing': training_end - training_start, 
                'traning_stats': agent.stats,
                }
        exp.add_result(test_rewards / float(test_n_iter), res_data)

    # Print results
    print '\n'
    print 'Training Time: ', training_end - training_start
    print 'Testing Time: ', test_end - test_start
    print 'Average Test Reward:', test_rewards / float(test_n_iter)

    
