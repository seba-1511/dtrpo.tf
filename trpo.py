#!/usr/bin/env python

import numpy as np
import cPickle as pk
import tensorflow as tf
from keras import backend as K
from time import time
from variables import DTYPE, EPSILON
from utils import convert_type, discount, LinearVF, gauss_log_prob, numel, dot_not_flat

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

DISTRIBUTED = size > 1

def dprint(*args, **kwargs):
    if DISTRIBUTED and not rank == 0:
        return
    else: 
        print(args)


def sync_list(arrays, avg=True):
    buffs = [np.copy(a) for a in arrays]
    for a, b in zip(arrays, buffs):
        comm.Allreduce(a, b, MPI.SUM)
        if avg:
            b /= float(size)
    return buffs


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
                 gamma=0.99, update_freq=100, gae=True, gae_lam=0.97, 
                 cg_damping=0.1, momentum=0.0):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.delta = delta
        self.gamma = gamma
        self.update_freq = update_freq
        self.step = 0
        self.episodes = 0
        self.vf = LinearVF()
        self.gae = gae
        self.gae_lam = gae_lam
        self.cg_damping = cg_damping
        self.momentum = momentum
        self._reset_iter()
        self.np_action_logstd_param = convert_type(0.01 * np.random.randn(1, policy.out_dim))
        self.action_logstd_param = K.variable(self.np_action_logstd_param)
        self.previous = [0.0 for _ in self.params]
        self.stats = {
            'avg_reward': [],
            'total_steps': [],
            'total_ep': [],
            'total_time': [],
            'kl_div': [],
            'surr_loss': [],
            'entropy': [],
        }

        self.build_computational_graphs()

        self.start_time = time()
        if DISTRIBUTED:
            params = K.batch_get_value(self.params)
            params = sync_list(params)
            self.set_params(params)

    @property
    def n_iterations(self):
        return self.step // self.update_freq

    @property
    def params(self):
        return [self.action_logstd_param, ]  + self.policy.params

    def build_computational_graphs(self):
        # Build loss graphs
        a = K.placeholder(ndim=2, name='actions')
        s = K.placeholder(ndim=2, name='states')
        a_means = K.placeholder(ndim=2, name='action_means')
        a_logstds = K.placeholder(ndim=2, name='actions_logstds')
        advantages = K.placeholder(ndim=2, name='advantage_values')
        new_logstds_shape = K.placeholder(ndim=2, name='logstds_shape') # Just a 0-valued tensor of the right shape
        inputs = [a, s, a_means, a_logstds, advantages, new_logstds_shape]

        self.surrogate, self.surr_graph, self.grads_surr_graph = self.build_surrogate(inputs)
        self.entropy, self.ent_graph, self.grads_ent_graph = self.build_entropy(inputs)
        self.kl, self.kl_graph, self.grads_kl_graph = self.build_kl(inputs)
        self.losses = K.function(inputs, [self.surr_graph, self.ent_graph, self.kl_graph])

        # TODO: Clean the following scrap
        tangents = [K.placeholder(shape=K.get_value(p).shape) for p in self.params]
        new_a_means = self.policy.model(s)
        new_a_logstds = new_logstds_shape + self.action_logstd_param
        stop_means, stop_logstds = K.stop_gradient(new_a_means), K.stop_gradient(new_a_logstds)

        # NOTE: Following seems fishy. (cf: Sutskever's code, utils.py:gauss_selfKL_firstfixed())
        stop_var = K.exp(2 * stop_logstds)
        var = K.exp(2 * new_a_logstds)
        temp = (stop_var + (stop_means - new_a_means)**2) / (2*var)
        kl_first_graph = K.mean(new_a_logstds - stop_logstds + temp - 0.5)
        grads_kl_first = K.gradients(kl_first_graph, self.params)

        gvp_graph = [K.sum(g * t) for g, t in zip(grads_kl_first, tangents)]
        grads_gvp_graph = K.gradients(gvp_graph, self.params)
        # self.grads_gvp = K.function(inputs + tangents + self.params, [kl_first_graph, ])
        self.grads_gvp = K.function(inputs + tangents + self.params, grads_gvp_graph)


    def _reset_iter(self):
        self.iter_reward = 0
        self.iter_n_ep = 0
        self.iter_actions = [[], ]
        self.iter_states = [[], ]
        self.iter_rewards = [[], ]
        self.iter_action_mean = [[], ]
        self.iter_action_logstd = [[], ]
        self.iter_done = []

    def _remove_episode(self, ep):
        """
        ep: int or list of ints. Index of episodes to be remove from current 
        iteration.
        """
        self.iter_rewards = np.delete(self.iter_rewards, ep, 0)
        self.iter_actions = np.delete(self.iter_actions, ep, 0)
        self.iter_states = np.delete(self.iter_states, ep, 0)
        self.iter_action_mean = np.delete(self.iter_action_mean, ep, 0)
        self.iter_action_logstd = np.delete(self.iter_action_logstd, ep, 0)
        self.iter_done = np.delete(self.iter_done, ep, 0)
        if isinstance(ep, list):
            self.iter_n_ep -= len(ep)
            self.episodes -= len(ep)
        else:
            self.iter_n_ep -= 1
            self.episodes -= 1


    def act(self, s):
        """
        Sample from the policy, where the outputs of the net are the mean and
        the param self.action_logstd_param gives you the logstd.
        """
        s = convert_type(s)
        s = s.reshape((1, -1))
        action_mean = convert_type(self.policy(s))
        out = np.copy(action_mean) 
        out += (np.exp(self.np_action_logstd_param) *
                np.random.randn(*self.np_action_logstd_param.shape))
        return out[0], {'action_mean': action_mean,
                        'action_logstd': self.np_action_logstd_param}

    def new_episode(self, terminated=False):
        """
        terminated: whether the episode was terminated by the env (True), or 
        manually (False).
        """
        self.iter_done.append(terminated)
        if not terminated or self.step % self.update_freq != 0:
            self.iter_n_ep += 1
            self.episodes += 1
            self.iter_actions.append([])
            self.iter_states.append([])
            self.iter_rewards.append([])
            self.iter_action_mean.append([])
            self.iter_action_logstd.append([])

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

        if self.step % self.update_freq == 0:
            self.update()

    def update(self):
        """
        Performs the TRPO update of the parameters.
        """
        returns = []
        advantages = []

        # Compute advantage 
        for ep in xrange(self.iter_n_ep+1):
            r = discount(self.iter_rewards[ep], self.gamma)
            b = self.vf(self.iter_states[ep])
            if self.gae and len(b) > 0:
                terminated = len(self.iter_done) != ep and self.iter_done[ep] 
                b1 = np.append(b, 0 if terminated else b[-1])
                deltas = self.iter_rewards[ep] + self.gamma*b1[1:] - b1[:-1]
                adv = discount(deltas, self.gamma * self.gae_lam)
            else:
                adv = r - b
            returns.append(r)
            advantages.append(adv)

        # Fit baseline for next iter
        self.vf.learn(self.iter_states, returns)

        states = np.concatenate(self.iter_states)
        actions = np.concatenate(self.iter_actions)
        actions = convert_type(actions)
        means = np.concatenate(self.iter_action_mean).reshape(states.shape[0], -1)
        logstds = np.concatenate(self.iter_action_logstd).reshape(states.shape[0], -1)

        # Standardize A() to have mean=0, std=1
        advantage = np.concatenate(advantages).reshape(states.shape[0], -1)
        advantage -= advantage.mean()
        advantage /= (advantage.std() + EPSILON)
        advantage = convert_type(advantage)

        inputs = [actions, states, means, logstds, advantage]

        # TODO: The following is to be cleaned, most of it can be made into a graph
        #Begin of CG

        surr_loss, surr_gradients = self.surrogate(*inputs)
        def fisher_vec_prod(vectors):
            a_logstds = np.zeros(means.shape, dtype=DTYPE) + logstds
            new_logstds = np.zeros(means.shape, dtype=DTYPE)
            args = [actions, states, means, a_logstds, advantage, new_logstds]

            res = self.grads_gvp(args + vectors)
            # GRAPH: directly get grads_graph and extend it with the following
            return [r + (p * self.cg_damping) for r, p in zip(res, vectors)]

        def conjgrad(fvp, grads, cg_iters=10, residual_tol=1e-10):
            p = [np.copy(g) for g in grads]
            r = [np.copy(g) for g in grads]
            x = [np.zeros_like(g) for g in grads]
            # GRAPH: extend fvp, control with tf.cond, inputs are p, r, x
            rdotr = dot_not_flat(r, r)
            # rdotr = dot_not_flat(grads, grads)
            for i in xrange(cg_iters):
                z = fvp(p)
                pdotz = np.sum([np.sum(a*b) for a, b in zip(p, z)])
                v = rdotr / pdotz
                x = [a + (v*b) for a, b in zip(x, p)]
                r = [a - (v*b) for a, b in zip(r, z)]
                newrdotr = np.sum([np.sum(g**2) for g in grads])
                mu = newrdotr / rdotr
                p = [a + mu * b for a, b in zip(r, p)]
                rdotr = newrdotr
                if rdotr < residual_tol:
                    break
            return x

        grads = [-g for g in surr_gradients]
        stepdir = conjgrad(fisher_vec_prod, grads)
        # GRAPH: stepdir is a graph, extend it to shs
        shs = 0.5 * dot_not_flat(stepdir, fisher_vec_prod(stepdir))
        assert shs > 0

        lm = np.sqrt(shs / self.delta)
        fullstep = [s / lm for s in stepdir]
        neggdotdir = dot_not_flat(grads, stepdir)
        # GRAPH: All 5 lines above can be converted to a graph
        # End of CG

        # Begin Linesearch
        def loss(params):
            self.set_params(params)
            return self.surrogate(*inputs)[0]

        def linesearch(loss, params, fullstep, exp_improve_rate):
            # GRAPH: graph the following with tf.cond
            accept_ratio = 0.1
            max_backtracks = 10
            loss_val = loss(params)
            for (i, stepfrac) in enumerate(0.5 ** np.arange(max_backtracks)):
                update = [(stepfrac * f) for f in fullstep]
                new_params = [p + u for p, u in zip(params, update)]
                # new_params = [a + (stepfrac * b) for a, b in zip(params, fullstep)]
                new_loss_val = loss(new_params)
                actual_improve = loss_val - new_loss_val
                exp_improve = stepfrac * exp_improve_rate
                ratio = actual_improve / exp_improve
                if ratio > accept_ratio and actual_improve > 0:
                    return update
                    # return new_params
            return [0 for f in fullstep]
            # return params

        params = K.batch_get_value(self.params)
        # Apply momentum pre-linesearch:
        # params = [p + self.momentum * prev 
                  # for p, prev in zip(params, self.previous)]
        update = linesearch(loss, params, fullstep, neggdotdir / lm)
        update = [u + f for f, u in zip(fullstep, update)]
        if DISTRIBUTED:
            update = sync_list(update, avg=True)
            # fullstep = sync_list(fullstep, avg=True)
        self.previous = [self.momentum * prev + u
                         for prev, u in zip(self.previous, update)]
        # Apply momentum pre-linesearch:
        # new_params = [p + u for p, u in zip(params, update)]
        # Apply momentum post-linesearch:
        new_params = [p + u for p, u in zip(params, self.previous)]
        self.set_params(new_params)
        # End Linesearch

        a_logstds = np.zeros(means.shape, dtype=DTYPE) + logstds
        new_logstds = np.zeros(means.shape, dtype=DTYPE)
        args = [actions, states, means, a_logstds, advantage, new_logstds]

        surr, ent, kl = self.losses(args)

        stats = {
            'avg_reward': self.iter_reward / float(max(self.iter_n_ep, 1)),
            'total_steps': self.step,
            'total_ep': int(self.episodes),
            'total_time': time() - self.start_time,
            'kl_div': float(kl),
            'surr_loss': float(surr),
            'entropy': float(ent),
        }
        dprint('*' * 20, 'Iteration ' + str(self.n_iterations), '*' * 20)
        dprint('Average Reward on Iteration:', stats['avg_reward'])
        dprint('Total Steps: ', self.step)
        dprint('Total Epsiodes: ', self.episodes)
        dprint('Time Elapsed: ', stats['total_time'])
        dprint('KL Divergence: ', kl)
        dprint('Surrogate Loss: ', surr)
        dprint('Entropy: ', ent)
        dprint('\n')
        self.stats['avg_reward'].append(stats['avg_reward'])
        self.stats['total_steps'].append(stats['total_steps'])
        self.stats['total_ep'].append(stats['total_ep'])
        self.stats['total_time'].append(stats['total_time'])
        self.stats['kl_div'].append(stats['kl_div'])
        self.stats['surr_loss'].append(stats['surr_loss'])
        self.stats['entropy'].append(stats['entropy'])
        self._reset_iter()

    def done(self):
        return False
        # return self.iter_reward >= self.env.solved_threshold * 1.1

    def load(self, path):
        with open(path, 'wb') as f:
            params = pk.load(f)
            self.set_params(params)

    def save(self, path):
        params = K.batch_get_value(self.params)
        with open(path, 'wb') as f:
            pk.dump(params, f)

    def set_params(self, params):
        K.batch_set_value(zip(self.params, params))
        self.np_action_logstd_param = K.get_value(self.action_logstd_param)

    def update_params(self, updates):
        for p, u in zip(self.params, updates):
            K.set_value(p, K.get_value(p) + u)
        self.np_action_logstd_param = K.get_value(self.action_logstd_param)

    def build_surrogate(self, variables):
        # Build graph of surrogate loss
        a, s, a_means, a_logstds, advantages, new_logstds_shape = variables

        # Computes the gauss_log_prob on sampled data.
        old_log_p_n = gauss_log_prob(a_means, a_logstds, a)

        # Here we compute the gauss_log_prob, but wrt current params
        new_a_means = self.policy.model(s)
        new_a_logstds = new_logstds_shape + self.action_logstd_param
        new_log_p_n = gauss_log_prob(new_a_means, new_a_logstds, a)

        # Compute the actual surrogate
        ratio = K.exp(new_log_p_n - old_log_p_n)
        advantages = K.reshape(advantages, (-1, ))

        surr_graph = -K.mean(ratio * advantages)
        grad_surr_graph = K.gradients(surr_graph, self.params)

        inputs = variables + self.params
        graph = K.function(inputs, [surr_graph, ] + grad_surr_graph)
        # graph = K.function(inputs, [surr_graph, ])

        def surrogate(actions, states, action_means, action_logstds, advantages):
            # TODO: Allocating new np.arrays might be slow. 
            logstds = np.zeros(action_means.shape, dtype=DTYPE) + action_logstds
            new_logstds = np.zeros(action_means.shape, dtype=DTYPE)
            args = [actions, states, action_means, logstds,
                    advantages, new_logstds]
            res = graph(args)
            return res[0], res[1:]

        return surrogate, surr_graph, grad_surr_graph

    def build_kl(self, variables):
        a, s, a_means, a_logstds, advantages, new_logstds_shape = variables

        new_a_means = self.policy.model(s)
        new_a_logstds = new_logstds_shape + self.action_logstd_param

        # NOTE: Might be fishy. (cf: Sutskever's code, utils.py:gauss_KL())
        old_var = K.exp(2 * a_logstds)
        new_var = K.exp(2 * new_a_logstds)
        temp = (old_var + (a_means - new_a_means)**2) / (2 * new_var)
        kl_graph = K.mean(new_a_logstds - a_logstds + temp - 0.5)
        grad_kl_graph = K.gradients(kl_graph, self.params)

        inputs = variables + self.params
        # graph = K.function(inputs, [kl_graph, ] + grad_kl_graph)
        graph = K.function(inputs, [kl_graph, ])

        def kl(actions, states, action_means, action_logstds, advantages):
            logstds = np.zeros(action_means.shape, dtype=DTYPE) + action_logstds
            new_logstds = np.zeros(action_means.shape, dtype=DTYPE)
            args = [actions, states, action_means, logstds,
                      advantages, new_logstds]
            res = graph(args)
            return res[0], res[1:]

        return kl, kl_graph, grad_kl_graph

    def build_entropy(self, variables):
        a, s, a_means, a_logstds, advantages, new_logstds_shape = variables

        new_a_logstds = new_logstds_shape + self.action_logstd_param
        ent_graph = K.mean(new_a_logstds + 0.5 * np.log(2*np.pi*np.e))
        grad_ent_graph = K.gradients(ent_graph, self.params)

        inputs = variables + self.params
        # graph = K.function(inputs, [ent_graph, ] + grad_ent_graph)
        graph = K.function(inputs, [ent_graph, ])

        def entropy(actions, states, action_means, action_logstds, advantages):
            logstds = np.zeros(action_means.shape, dtype=DTYPE) + action_logstds
            new_logstds = np.zeros(action_means.shape, dtype=DTYPE)
            args = [actions, states, action_means, logstds,
                      advantages, new_logstds]
            res = graph(args)
            return res[0], res[1:]

        return entropy, ent_graph, grad_ent_graph
