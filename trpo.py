#!/usr/bin/env python

import numpy as np
import cPickle as pk
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
        self.np_action_logstd_param = 0.01 * np.random.randn(1, policy.out_dim).astype(DTYPE)
        self.action_logstd_param = K.variable(self.np_action_logstd_param)
        self.surrogate = self.build_surrogate()
        self.entropy = self.build_entropy()
        self.kl = self.build_kl()

        self.start_time = time()

    @property
    def n_iterations(self):
        return self.step // self.update_freq

    @property
    def params(self):
        return [self.action_logstd_param, ]  + self.policy.params

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
        out += (np.exp(self.np_action_logstd_param) *
                np.random.randn(*self.np_action_logstd_param.shape))
        return out[0], {'action_mean': action_mean,
                        'action_logstd': self.np_action_logstd_param}

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

        surr_loss, grads = self.surrogate(states, actions, means, logstds, advantage)
        update = self.optimizer(grads)
        self.update_params(update)

        print '*' * 20, 'Iteration ' + str(self.n_iterations), '*' * 20
        print 'Average Reward on Iteration:', self.iter_reward / float(ep+1)
        print 'Total Steps: ', self.step
        print 'Total Epsiodes: ', self.episodes
        print 'Time Elapsed: ', time() - self.start_time
        print 'KL Divergence: ', self.kl(None)
        print 'Surrogate Loss: ', surr_loss
        print 'Entropy: ', self.entropy(None)
        print '\n'
        self._reset_iter()

    def done(self):
        return False
        return self.iter_reward >= self.env.solved_threshold * 2.1

    def load(self, path):
        with open(path, 'wb') as f:
            params = pk.load(f)
            self.set_params(params)

    def save(self, path):
        params = K.batch_get_value(self.params)
        with open(path, 'wb') as f:
            pk.dump(params, f)

    def set_params(self, params):
        K.batch_set_value(self.params, params)

    def update_params(self, updates):
        for p, u in zip(self.params, updates):
            K.set_value(p, K.get_value(p) + u)

    def build_surrogate(self):
        # Build graph of surrogate
        a = K.placeholder(ndim=2)
        s = K.placeholder(ndim=2)
        a_means = K.placeholder(ndim=2)
        a_logstds = K.placeholder(ndim=2)
        advantages = K.placeholder(ndim=2)
        new_logstds_shape = K.placeholder(ndim=2) # Just a 0-valued tensor of the right shape

        # Computes the gauss_log_prob on sampled data.
        old_log_p_n = gauss_log_prob(a_means, a_logstds, a)

        # Here we compute the gauss_log_prob, but without sampling.
        new_a_means = self.policy.model(s)
        new_a_logstds = (new_logstds_shape + self.action_logstd_param)
        new_log_p_n = gauss_log_prob(new_a_means, new_a_logstds, a)

        # Compute the actual surrogate
        ratio = K.exp(new_log_p_n - old_log_p_n)
        surr = -K.mean(ratio * advantages)
        grad_surr = K.gradients(surr, self.params)

        inputs = [a, s, a_means, a_logstds, advantages, new_logstds_shape]
        inputs += self.params
	# TODO: Ridiculous that I have to run the function twice !
        surr_graph = K.function(inputs, [surr,], [])
	grad_surr_graph = K.function(inputs, grad_surr, [])

        def surrogate(states, actions, action_means, action_logstds, advantages):
            # TODO: Allocating new np.arrays is slow. Re-use them !
            logstds = np.zeros(action_means.shape, dtype=DTYPE) + action_logstds
            new_logstds = np.zeros(action_means.shape, dtype=DTYPE)
            args = [actions, states, action_means, logstds,
                      advantages, new_logstds]
            return surr_graph(args), grad_surr_graph(args)

        return surrogate

    def build_kl(self):
        return lambda x: 0.0

    def build_entropy(self):
        return lambda x: 0.0
