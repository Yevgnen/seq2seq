#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano import shared


class Optimizer(object):
    def __init__(self, clip=0.0):
        self.iterations = shared(0, name='iterations')
        self.updates = [(self.iterations, self.iterations + 1)]
        self.clip = clip

    def get_gradients(self, loss, params):
        grads = [T.grad(loss, param) for param in params]
        if self.clip and self.clip > 0:
            grads = self.clipping(grads)
        return grads

    def clipping(self, grads):
        grads = [T.clip(g, -self.clip, self.clip) for g in grads]
        return grads


class SGD(Optimizer):
    def __init__(self, clip=0.0, lr=0.1, decay=0.0, momentum=0.0):
        super(SGD, self).__init__(clip=clip)
        self.lr = shared(lr, name='lr')
        self.decay = decay
        self.momentum = momentum

    def get_updates(self, loss, params):
        # gradient
        grads = self.get_gradients(loss, params)

        # update learning rate
        lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.updates.append((self.lr, lr))

        # momentum
        self.momentums = [shared(value=np.zeros_like(param.get_value(borrow=True))) for param in params]
        for p, g, m in zip(params, grads, self.momentums):
            # velocity
            v = self.momentum * m - lr * g
            self.updates.append((m, v))

            # update param
            new_p = p + v
            self.updates.append((p, new_p))
        return self.updates


class RMSprop(Optimizer):
    def __init__(self, clip=0.0, lr=0.001, decay=0.0, gamma=0.9, eps=1e-8):
        super(RMSprop, self).__init__(clip=clip)
        self.lr = shared(lr, name='lr')
        self.decay = decay
        self.gamma = gamma
        self.eps = eps

    def get_updates(self, loss, params):
        # gradients
        grads = self.get_gradients(loss, params)

        # update learning rate
        lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.updates.append((self.lr, lr))

        self.gradients = [shared(value=np.zeros_like(param.get_value(borrow=True))) for param in params]
        for p, g, h in zip(params, grads, self.gradients):
            new_h = self.gamma * h + (1 - self.gamma) * T.square(g)
            self.updates.append((h, new_h))

            new_p = p - (self.lr * g) / (T.sqrt(new_h) + self.eps)
            self.updates.append((p, new_p))

        return self.updates


class AdaGrad(Optimizer):
    def __init__(self, clip=0.0, lr=0.01, decay=0.0, eps=1e-8):
        super(AdaGrad, self).__init__(clip=clip)
        self.lr = shared(lr, name='lr')
        self.decay = decay
        self.eps = eps

    def get_updates(self, loss, params):
        # gradients
        grads = self.get_gradients(loss, params)

        # update learning rate
        lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.updates.append((self.lr, lr))

        self.gradients = [shared(value=np.zeros_like(param.get_value(borrow=True))) for param in params]
        for p, g, h in zip(params, grads, self.gradients):
            new_h = h + T.square(g)
            self.updates.append((h, new_h))

            new_p = p - (self.lr * g) / (T.sqrt(new_h) + self.eps)
            self.updates.append((p, new_p))

        return self.updates


class Adam(Optimizer):
    def __init__(self):
        return


class AdaDelta(Optimizer):
    def __init__(self):
        return


class GradientChecking(object):
    def __init__(self, model):
        self.model = model

    def numerical_gradient(self, test_x, test_y, eps=1e-5):
        x = T.dmatrix('x')
        y = T.ivector('y')

        compute_loss = theano.function(
            inputs=[x, y],
            outputs=self.model.loss(x, y)
        )

        grads = []

        for param in self.model.params:
            param_val = param.get_value(borrow=True)
            grad = np.zeros_like(param_val)

            if param_val.ndim == 2:
                (row, col) = param_val.shape
                for i in range(row):
                    for j in range(col):
                        pij = param_val[i, j]
                        param_val[i, j] = pij + eps
                        setattr(self.model, param.name, param_val)
                        l2 = compute_loss(test_x, test_y)

                        param_val[i, j] = pij - eps
                        setattr(self.model, param.name, param_val)
                        l1 = compute_loss(test_x, test_y)
                        grad[i, j] = (l2 - l1) / (2 * eps)

                        param_val[i, j] = pij
                        setattr(self.model, param.name, param_val)
            elif param_val.ndim == 1:
                length = param_val.shape[0]
                for i in range(length):
                    pi = param_val[i]
                    param_val[i] = pi + eps
                    setattr(self.model, param.name, param_val)
                    l2 = compute_loss(test_x, test_y)

                    param_val[i] = pi - eps
                    setattr(self.model, param.name, param_val)
                    l1 = compute_loss(test_x, test_y)
                    grad[i] = (l2 - l1) / (2 * eps)

                    param_val[i] = pi
                    setattr(self.model, param.name, param_val)
            grads.append(grad)

        return grads

    def check_gradient(self, test_x, test_y):
        x = T.dmatrix('x')
        y = T.ivector('y')

        # Auto differentiation
        compute_grads = theano.function(
            inputs=[x, y],
            outputs=[T.grad(self.model.loss(x, y), param) for param in self.model.params]
        )
        grads = compute_grads(test_x, test_y)

        # Numerical differentiation
        ngrads = self.numerical_gradient(test_x, test_y)

        for param, grad, ngrad in zip(self.model.params, grads, ngrads):
            print('Gradient checking of param: {0}, max differences: {1}'.format(
                param.name, np.absolute(grad - ngrad).max()))
