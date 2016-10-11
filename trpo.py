#!/usr/bin/env python

import numpy as np
import chainer as ch
import chainer.functions as F
from chainer.cuda import cupy as cp
from time import time
from variables import DTYPE, EPSILON
from utils import convert_type, discount, LinearVF, gauss_log_prob, numel


class TRPO(object):

    """ TRPO Implementation
    Args:
        env: a Gym environment
        policy: a differentiable tf function
        optimizer: an optimizer
        delta: max KL penalty
        gamma: discounting factor
    """

    def __init__(self, env, policy=None, optimizer=None, delta=0.01,
                 gamma=0.995, update_freq=100):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.delta = delta
        self.gamma = gamma
        self.update_freq = update_freq
        self.step = 0
        self.episodes = 0
        self.vf = LinearVF()
        self._reset_iter()
        self.action_logstd_param = 0.01 * np.random.randn(1, policy.out_dim).astype(DTYPE)

        self.start_time = time()

    @property
    def n_iterations(self):
        return self.step // self.update_freq

    def _reset_iter(self):
        self.iter_reward = 0
        self.iter_n_ep = 0
        self.iter_states = [[], ]
        self.iter_rewards = [[], ]
        self.iter_actions = [[], ]
        self.iter_action_mean = [[], ]
        self.iter_action_logstd = [[], ]

    def act(self, s):
        s = ch.Variable(convert_type(s))
        action_mean = self.policy(s)
        action_logstd = np.tile(self.action_logstd_param,
                                np.asarray((self.policy.out_dim, 1), dtype=DTYPE))
        info_stats = {
                'action_mean': action_mean[0][0],
                'action_logstd': action_logstd[0][0],
                }
        out = action_mean + np.exp(action_logstd) * np.random.randn(*action_logstd.shape)
        return out[0][0], info_stats

    def learn(self, s0, a, r, s1, end_ep, action_info=None):
        ep = self.iter_n_ep
        self.iter_actions[ep].append(a)
        self.iter_states[ep].append(s0)
        self.iter_rewards[ep].append(r)
        if action_info is not None:
            self.iter_action_mean[ep].append(action_info['action_mean'])
            self.iter_action_logstd[ep].append(action_info['action_logstd'])
        self.step += 1
        self.iter_reward += r

        if end_ep:
            ep = self.iter_n_ep = ep + 1
            self.episodes += 1
            self.iter_actions.append([])
            self.iter_states.append([])
            self.iter_rewards.append([])
            self.iter_action_mean.append([])
            self.iter_action_logstd.append([])

        if self.step % self.update_freq == 0:
            self.update()

    def update(self):
        baselines = []
        returns = []
        advantages = []

        for ep in xrange(self.iter_n_ep+1):
            r = discount(self.iter_rewards[ep], self.gamma)
            b = self.vf(self.iter_states[ep])
            baselines.append(b)
            returns.append(r)
            advantages.append(r - b)

        # Fit baseline for next iter
        self.vf.learn(self.iter_states, returns)

        # Standardize A() to have mean=0, std=1
        advantage = np.concatenate(advantages)
        advantage -= advantage.mean()
        advantage /= (advantage + EPSILON)

        states = np.concatenate(self.iter_states)
        actions = np.concatenate(self.iter_actions)
        actions = convert_type(actions)
        advantage = convert_type(advantage)
        means = np.concatenate(self.iter_action_mean)
        logstds = np.concatenate(self.iter_action_logstd)

        surrogate = self._surrogate(states, actions, means, logstds, advantage)
        grads = surrogate.backward()
        print 'Grads: ', grads

        update = self.optimizer(self.policy.params, grads)
        self.set_params(update)

        print '*' * 20, 'Iteration ' + str(self.n_iterations), '*' * 20
        print 'Average Reward on Iteration:', self.iter_reward / float(ep+1)
        print 'Total Steps: ', self.step
        print 'Total Epsiodes: ', self.episodes
        print 'Time Elapsed: ', time() - self.start_time
        print 'KL Divergence: ', self._kl()
        print 'Surrogate Loss: ', surrogate.data
        print 'Entropy: ', self._entropy()
        print '\n'
        self._reset_iter()

    def done(self):
        return False
        return self.iter_reward >= self.env.solved_threshold * 1.1

    def save(self, name):
        pass

    def set_params(self, params):
        self.policy.set_params(params)

    def _grads(self):
        return 0

    def _surrogate(self, states, actions, action_means, action_logstds, advantage):
        actions = ch.Variable(actions)
        a_means = ch.Variable(action_means)
        a_logstds = ch.Variable(action_logstds)
        old_logp_n = gauss_log_prob(a_means, a_logstds, actions)

        # TODO: Get correct log_p_n with net

        # new_a_means = self.policy(convert_type(states))
        # new_a_logstds = np.tile(self.action_logstd_param,
                             # np.asarray((self.policy.out_dim, 1), dtype=DTYPE))
        # print new_a_means.shape, new_a_logstds.shape
        # log_p_n = gauss_log_prob(new_a_means, new_a_logstds, actions)


        log_p_n = gauss_log_prob(a_means, a_logstds, actions)
        log_oldp_n = gauss_log_prob(a_means, a_logstds, actions)
        # log_oldp_n = gauss_log_prob(old_action_means, old_action_logstds, actions)
        ratio = F.exp(log_p_n - log_oldp_n)
        advantages = ch.Variable(advantage)
        advantages *= F.transpose(ratio)
        surrogate = -F.sum(advantages) / numel(advantages)
        return surrogate

    def _kl(self):
        return 0.0

    def _entropy(self):
        return 0.0
