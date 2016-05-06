#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from datetime import datetime

import numpy as np
import theano
import theano.tensor as T
from numpy import random as rng
from theano import shared

ACTIVATION = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softmax': T.nnet.softmax, 'relu': T.nnet.relu}


class Layer(object):
    def __init__(self, input_size, output_size, activation='tanh'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class FullConnected(Layer):
    def __init__(self, input_size, output_size, activation='tanh', regularizer='l2'):
        super(FullConnected, self).__init__(input_size, output_size, activation)

        W_value = np.asarray(
            rng.uniform(low=-np.sqrt(6. / (input_size + output_size)),
                        high=np.sqrt(6. / (input_size + output_size)),
                        size=(output_size, input_size)),
            dtype=theano.config.floatX)
        if activation == 'tanh':
            W_value *= 4
        self.W = shared(value=W_value, name='W', borrow=True)

        b_value = np.zeros(output_size, dtype=theano.config.floatX)
        self.b = shared(value=b_value, name='b', borrow=True)

        self.params = [self.W, self.b]

        self.activation = ACTIVATION[activation]

    def forward(self, x):
        sum = T.dot(x, self.W.T) + self.b
        return self.activation(sum)


class LSTM(Layer):
    def __init__(self, vocab_size, hidden_size, activation='softmax'):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        param_names = ['Wz', 'Wi', 'Wf', 'Wo', 'Rz', 'Ri', 'Rf', 'Ro', 'pi', 'pf', 'po', 'bz', 'bi', 'bf', 'bo', 'V', 'c']
        params = []

        for param_name in param_names:
            prefix = param_name[0]
            # Determine param shape
            if prefix == 'W':
                shape = (hidden_size, vocab_size)
            elif prefix == 'R':
                shape = (hidden_size, hidden_size)
            elif prefix == 'V':
                shape = (vocab_size, hidden_size)
            elif prefix == 'c':
                shape = (vocab_size,)
            else:
                shape = (hidden_size,)

            # Init weights
            if (len(shape) == 2):
                param_value = np.asarray(rng.uniform(
                    low=-np.sqrt(1. / shape[1]),
                    high=np.sqrt(1. / shape[1]),
                    size=shape), dtype=theano.config.floatX)
            # Init biases
            else:
                param_value = np.zeros(shape)

            param = shared(value=param_value, name=param_name, borrow=True)
            params.append(param)
            setattr(self, param_name, param)
        self.params = params

        self.activation = ACTIVATION[activation]

    def time_step(self, x, y, prev_h, prev_c):
        # Block input
        zbar = T.dot(x, self.Wz.T) + T.dot(prev_h, self.Rz.T) + self.bz
        z = T.tanh(zbar)

        # Input gate
        ibar = T.dot(x, self.Wi.T) + T.dot(prev_h, self.Ri.T) + self.pi * prev_c + self.bi
        i = T.nnet.sigmoid(ibar)

        # Forget gate
        fbar = T.dot(x, self.Wf.T) + T.dot(prev_h, self.Rf.T) + self.pf * prev_c + self.bf
        f = T.nnet.sigmoid(fbar)

        # Cell
        c = z * i + prev_c * f
        a = T.tanh(c)

        # Output gate
        obar = T.dot(x, self.Wo.T) + T.dot(prev_h, self.Ro.T) + self.po * c + self.bo
        o = T.nnet.sigmoid(obar)

        # Block output
        # The origin paper `LSTM: A Search Space Odyssey` use ``y`` to denote block output, but here use ``h`` instead.
        h = a * o

        tilde_a = T.dot(h, self.V.T)
        y = self.activation(tilde_a)[0]  # T.nnet.softmax returns a 2D matrix

        return (y, h, c)

    def forward(self, x):
        ((Y, H, C), _) = theano.scan(
            fn=self.time_step,
            outputs_info=[
                dict(initial=np.zeros(self.vocab_size)),
                dict(initial=np.zeros(self.hidden_size), taps=[-1]),
                dict(initial=np.zeros(self.hidden_size), taps=[-1])],
            sequences=[x]
        )

        return (Y, H, C)
