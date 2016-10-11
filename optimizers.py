#!/usr/bin/env python


class ConjugateGradients(object):

    def __init__(self, alpha=0.1, damping=0.1):
        self.damping = damping
        self.alpha = alpha

    def __call__(self, params, grads, hessian=None, epoch=None):
        return params
        res = []
        updates = self.update(grads)
        for p, u in zip(params, updates):
            res.append(p + self.alpha * u)
        return res

    def update(self, grads):
        # TODO: Use real CG update
        return [-g for g in grads]
