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


class Embedding(FullConnected):
    def __init__(self, vocab_size, output_size, activation='tanh'):
        super(Embedding, self).__init__(vocab_size, output_size, activation)

    def forward(self, X):
        return self.activation(self.W[:, X].T + self.b)


class TimeDistributed(FullConnected):
    def __init__(self, input_size, output_size, activation='softmax'):
        super(TimeDistributed, self).__init__(input_size, output_size, activation)


class LSTM(Layer):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        param_names = ['W', 'R', 'pi', 'pf', 'po', 'bz', 'bi', 'bf', 'bo']
        params = []

        for param_name in param_names:
            prefix = param_name[0]
            # Determine param shape
            if prefix == 'W':
                shape = (hidden_size * 4, input_size)
            elif prefix == 'R':
                shape = (hidden_size * 4, hidden_size)
            else:
                shape = (hidden_size,)

            # Init weights
            if (len(shape) == 2):
                param_value = np.asarray(rng.uniform(
                    low=-np.sqrt(1. / shape[1]),
                    high=np.sqrt(1. / shape[1]),
                    size=shape), dtype=theano.config.floatX)
            # Init biases
            elif param_name == 'bf':
                param_value = np.ones(shape)
            else:
                param_value = np.zeros(shape)

            param = shared(value=param_value, name=param_name, borrow=True)
            params.append(param)
            setattr(self, param_name, param)
        self.params = params

    def _slice(self, M, i):
        return M[:, i * self.hidden_size: (i + 1) * self.hidden_size]

    def step(self, batch, mask, prev_h, prev_c):
        """Forward a time step of minibatch.

        Reference: [1] LSTM: A Search Space Odyssey, Klaus Greff, Rupesh Kumar Srivastava, Jan Koutník,
                       Bas R. Steunebrink, Jürgen Schmidhuber, http://arxiv.org/abs/1503.04069
        """
        # Compute the input and recurrent signals
        Wx = T.dot(batch, self.W.T)
        Rh = T.dot(prev_h, self.R.T)

        # Block input
        z = T.tanh(self._slice(Wx, 0) + self._slice(Rh, 0) + self.bz)

        # Input gate
        i = T.nnet.sigmoid(self._slice(Wx, 1) + self._slice(Rh, 1) + self.pi * prev_c + self.bi)

        # Forget gate
        f = T.nnet.sigmoid(self._slice(Wx, 2) + self._slice(Rh, 2) + self.pf * prev_c + self.bf)

        # Cell
        c = z * i + prev_c * f

        # Output gate
        o = T.nnet.sigmoid(self._slice(Wx, 3) + self._slice(Rh, 3) + self.po * c + self.bo)

        # Block output
        h = T.tanh(c) * o

        c = mask[:, np.newaxis] * c + (1 - mask[:, np.newaxis]) * prev_c
        h = mask[:, np.newaxis] * h + (1 - mask[:, np.newaxis]) * prev_h

        return (h, c)

    def forward(self, batch, mask):
        (sens_size, batch_size, embedding_size) = T.shape(batch)

        ((H, C), _) = theano.scan(
            fn=self.step,
            outputs_info=[
                dict(initial=T.zeros((batch_size, self.hidden_size)), taps=[-1]),
                dict(initial=T.zeros((batch_size, self.hidden_size)), taps=[-1])],
            sequences=[batch, mask]
        )

        return (H, C)
