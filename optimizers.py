#!/usr/bin/env python

from keras import backend as K


class ConjugateGradients(object):

    def __init__(self, alpha=0.001, damping=0.1):
        self.damping = damping
        self.alpha = K.variable(alpha)
        g = K.placeholder(ndim=2)
        self.update = K.function([g, ], [-self.alpha * g, ])

    def __call__(self, grads, hessian=None, epoch=None):
        res = []
        grads_placeholders = [K.placeholder(shape=g.shape) for g in grads]
        update_graph = [-self.alpha * g for g in grads_placeholders]
        update_fn = K.function(grads_placeholders, update_graph)
        res = update_fn(grads)[0]
        return res


class FisherConjugateGradients(object):

    def __init__(self, damping=0.01, max_iters=10, residual_tol=1e-10):
        self.damping = damping
        self.max_iters = max_iters
        self.residual_tol = residual_tol

    def fisher_vec_prod(p):
        pass

    def __call__(self, grads, hessian=None, epoch=None):
        # Flatten gradients
        # Compute step direction
        # Compute full step
        pass
        
