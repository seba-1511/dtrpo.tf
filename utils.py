#!/usr/bin/env python

import numpy as np
from scipy import signal
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from variables import DTYPE

half_log_2pi = K.variable(np.log(2*np.pi) * 0.5)


class FCNet(object):
    """A simple policy network"""

    def __init__(self, in_dim, out_dim, layer_sizes=None):
        """
            layers: a list of the number of hidden units in the MLP
            in_dim: int, flattened number of inputs
            out_dim: int, flttened number of outputs
        """
        if layer_sizes is None:
            layer_sizes = [64, 64]
        self.in_dim = in_dim
        self.out_dim = out_dim
        in_dim = (in_dim, )

        self.layers = []
        x = data = Input(shape=in_dim)
        for l in layer_sizes:
            x = Dense(l, activation='relu', init='glorot_normal')(x)
            self.layers.append(x)
        x = Dense(out_dim, activation='linear', init='glorot_normal')(x)
        self.layers.append(x)
        self.model = Model(input=data, output=x)
        self.net = K.function([data, ], [self.model(data)], [])

    @property
    def params(self):
        return self.model.trainable_weights

    def __call__(self, x):
        """ x: a chainer variable"""
        return self.net([x, ])[0]

    def set_params(self, update):
        return 0.0
        raise('not implemented !')


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
        features = np.array(self.extract_features(states), DTYPE)
        return np.dot(features, self.W)

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
        self.W = np.array(W, DTYPE)


def gauss_log_prob(means, logstds, x):
    var = K.exp(2 * logstds)
    gp = (-(x - means)**2) / (2 * var) - half_log_2pi - logstds
    return gp


def numel(x):
    if hasattr(x, 'shape'):
        return reduce(lambda x, y: x*y, x.shape)
    return x.n


def convert_type(x):
    return np.array(x).astype(DTYPE)


def discount(rewards, gamma):
    return signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1].reshape(-1, 1)

def dot_not_flat(a, b):
    return np.sum([np.sum(x*y) for x, y in zip(a, b)])
