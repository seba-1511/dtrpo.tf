#!/usr/bin/env python

import numpy as np
from scipy import signal
import chainer as ch
import chainer.functions as F
import chainer.links as L
from variables import DTYPE

half_log_2pi = np.log(2*np.pi) * 0.5


class FCNet(ch.Chain):
    """A simple policy network"""

    def __init__(self, in_dim, out_dim, layers=None):
        """layers: a list of the number of hidden units in the MLP"""
        if layers is None:
            layers = [64, 64]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers_dict = {}
        self.layers_list = []
        prev = in_dim
        for i, curr in enumerate(layers):
            l = L.Linear(prev, curr)
            self.layers_list.append(l)
            self.layers_dict['l'+str(i)] = l
            prev = curr
        l = L.Linear(prev, out_dim)
        self.layers_list.append(l)
        self.layers_dict['l'+str(len(layers))] = l
        # self.net = ch.Chain(**self.layers_dict)
        params = []
        for l in self.layers_list:
            params.append(l.W)
            params.append(l.b)
        self.params = params

    def __call__(self, x):
        """ x: a chainer variable"""
        # Define net:
        net = F.reshape(x, (-1, self.in_dim))
        net = self.layers_list[0](net)
        for l in self.layers_list[1:]:
            net = l(F.relu(net))
        self.net = net
        return net

        # out = F.reshape(x, (-1, self.in_dim))
        # out = self.layers_list[0](out)
        # for l in self.layers_list[1:]:
            # out = l(F.relu(out))
        # return out.data

    def get_grads(self):
        return [p.grad for p in self.params]

    def cleargrads(self):
        for p in self.params:
            p.cleargrads()

    def set_params(self, update):
        for p, u in zip(self.params, update):
            p.copydata(u)


class LinearVF(object):

    def __init__(self, W=None):
        self.W = W

    def extract_features(self, states):
        s = np.array(states, dtype=DTYPE)
        s = s.reshape(s.shape[0], -1)
        l = len(states)
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([s, s**2, al, al**2, np.ones((l, 1))], axis=1)

    def __call__(self, states):
        if self.W is None:
            # return ch.Variable(np.zeros((1, len(states)), dtype=DTYPE)).data
            return np.zeros((len(states), 1), dtype=DTYPE)
        features = ch.Variable(self.extract_features(states))
        return F.matmul(features, self.W).data

    def learn(self, list_states, list_returns):
        # TODO: Implement lstsq on GPU
        features = [self.extract_features(states) for states in list_states]
        features = np.concatenate(features)
        returns = np.concatenate(list_returns)
        n_col = features.shape[1]
        lamb = 2.0
        W = np.linalg.lstsq(
                features.T.dot(features) + lamb * np.identity(n_col),
                features.T.dot(returns)
        )[0]
        self.W = ch.Variable(W)


def gauss_log_prob(means, logstds, x):
    var = F.exp(2.0*logstds)
    gp = -((x - means)**2) / 2.0 * var - half_log_2pi - logstds
    return gp


def numel(x):
    return reduce(lambda x, y: x*y, x.shape)


def convert_type(x):
    return np.array(x).astype(DTYPE)


def discount(rewards, gamma):
    # return signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1].reshape(1, -1)
    return signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1].reshape(-1, 1)
