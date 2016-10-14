#!/usr/bin/env python

import numpy as np
from keras import backend as K
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
        self.surrogate = self.build_surrogate(None, None, None, None, None)
        self.entropy = self.build_entropy()
        self.kl = self.build_kl()

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
        """
        Sample from the policy, where the outputs of the net are the mean and
        the param self.action_logstd_param gives you the logstd.
        """
        s = convert_type(s)
        s = s.reshape((1, -1))
        action_mean = self.policy(s)
        out = action_mean 
        out += (np.exp(self.action_logstd_param) *
                np.random.randn(*self.action_logstd_param.shape))
        return out[0], {'action_mean': action_mean,
                        'action_logstd': self.action_logstd_param}

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
        advantage /= (advantage.std() + EPSILON)

        states = np.concatenate(self.iter_states)
        actions = np.concatenate(self.iter_actions)
        actions = convert_type(actions)
        advantage = convert_type(advantage)
        means = np.concatenate(self.iter_action_mean).reshape(states.shape[0], -1)
        logstds = np.concatenate(self.iter_action_logstd).reshape(states.shape[0], -1)

        surrogate = self.surrogate(states, actions, means, logstds, advantage)
        grads = self.policy.get_grads()
        print 'Grads: ', grads

        update = self.optimizer(self.policy.params, grads)
        self.set_params(update)

        print '*' * 20, 'Iteration ' + str(self.n_iterations), '*' * 20
        print 'Average Reward on Iteration:', self.iter_reward / float(ep+1)
        print 'Total Steps: ', self.step
        print 'Total Epsiodes: ', self.episodes
        print 'Time Elapsed: ', time() - self.start_time
        print 'KL Divergence: ', self.kl(None)
        print 'Surrogate Loss: ', surrogate
        print 'Entropy: ', self.entropy(None)
        print '\n'
        self._reset_iter()

    def done(self):
        return False
        return self.iter_reward >= self.env.solved_threshold * 1.1

    def save(self, name):
        pass

    def set_params(self, params):
        self.policy.set_params(params)

    def build_surrogate(self, states, actions, action_means, action_logstds, advantage):
        return lambda x, a, s, d, f: 0.0
        actions = actions
        a_means = action_means
        a_logstds = np.zeros(a_means.shape, dtype=DTYPE) + action_logstds

        # Computes the gauss_log_prob on sampled data.
        log_oldp_n = gauss_log_prob(a_means, a_logstds, actions)

        # Here we compute the gauss_log_prob, but without sampling.
        new_a_means = self.policy(convert_type(states))
        new_a_logstds = (np.zeros(new_a_means.shape, dtype=DTYPE) +
                         convert_type(self.action_logstd_param))
        new_a_logstds = K.variable(new_a_logstds)
        log_p_n = gauss_log_prob(new_a_means, new_a_logstds, actions)

        ratio = K.exp(log_p_n - log_oldp_n)
        out = K.transpose(ch.Variable(advantage) * ratio)
        surrogate = -F.sum(out) / numel(out)
        return surrogate

    def build_kl(self):
        return lambda x: 0.0

    def build_entropy(self):
        return lambda x: 0.0
