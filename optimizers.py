#!/usr/bin/env python

from keras import backend as K


class ConjugateGradients(object):

    def __init__(self, alpha=0.001, damping=0.1):
        self.damping = damping
        self.alpha = K.variable(alpha)
        p = K.placeholder(ndim=2)
        g = K.placeholder(ndim=2)
        self.update = K.function([self.alpha, p, g], [p - self.alpha * g, ])

    def __call__(self, params, grads, hessian=None, epoch=None):
        return params
        res = []
        for p, u in zip(params, grads):
            res.append(self.update([self.alpha, p, u]))
        return res
