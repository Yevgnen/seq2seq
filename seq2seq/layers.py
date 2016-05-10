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


class Embedding(FullConnected):
    def __init__(self, vocab_size, output_size, activation='tanh'):
        super(Embedding, self).__init__(vocab_size, output_size, activation)

    def forward(self, X):
        return self.activation(self.W[:, X].T + self.b)
        # return T.as_tensor_variable([self.activation(self.W[:, x].T + self.b) for x in X])
        # sum = self.W[:, x].T + self.b
        # return self.activation(sum)


class TimeDistributed(FullConnected):
    def __init__(self, input_size, output_size, activation='softmax'):
        super(TimeDistributed, self).__init__(input_size, output_size, activation)


class LSTM(Layer):
    # def __init__(self, input_size, hidden_size, activation='softmax', return_sentences=True, feed_output=False):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        param_names = ['Wz', 'Wi', 'Wf', 'Wo', 'Rz', 'Ri', 'Rf', 'Ro', 'pi', 'pf', 'po', 'bz', 'bi', 'bf', 'bo', 'V', 'c']
        params = []

        for param_name in param_names:
            prefix = param_name[0]
            # Determine param shape
            if prefix == 'W':
                shape = (hidden_size, input_size)
            elif prefix == 'R':
                shape = (hidden_size, hidden_size)
            elif prefix == 'V':
                shape = (input_size, hidden_size)
            elif prefix == 'c':
                shape = (input_size,)
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

        # self.activation = ACTIVATION[activation]

        # self.retrun_sentences = return_sentences
        # self.feed_output = feed_output

    def _stack(self):
        return (T.concatenate([self.Wz, self.Wi, self.Wf, self.Wo]),
                T.concatenate([self.Rz, self.Ri, self.Rf, self.Ro]))

    def _slice(self, M):
        return np.asarray([M[:, i * self.hidden_size: (i + 1) * self.hidden_size] for i in range(4)])

    def lstm_step(self, batch, mask, prev_h, prev_c):
        """Forward a time step of minibatch.

        Reference: [1] LSTM: A Search Space Odyssey, Klaus Greff, Rupesh Kumar Srivastava, Jan Koutník,
                       Bas R. Steunebrink, Jürgen Schmidhuber, http://arxiv.org/abs/1503.04069
        """
        # Stack the weigth for fast matrix multiplication
        (W, R) = self._stack()
        Wx = T.dot(batch, W.T)
        Rh = T.dot(prev_h, R.T)

        Wxs = self._slice(Wx)
        Rhs = self._slice(Rh)

        # Block input
        z = T.tanh(Wxs[0] + Rhs[0] + self.bz)

        # Input gate
        i = T.nnet.sigmoid(Wxs[1] + Rhs[1] + self.pi * prev_c + self.bi)

        # Forget gate
        f = T.nnet.sigmoid(Wxs[2] + Rhs[2] + self.pf * prev_c + self.bf)

        # Cell
        c = z * i + prev_c * f

        # Output gate
        o = T.nnet.sigmoid(Wxs[3] + Rhs[3] + self.po * c + self.bo)

        # Block output
        h = T.tanh(c) * o

        c = mask[:, np.newaxis] * c + (1 - mask[:, np.newaxis]) * prev_c
        h = mask[:, np.newaxis] * h + (1 - mask[:, np.newaxis]) * prev_h

        return (h, c)

    def forward(self, batch, mask):
        (sens_size, batch_size, embedding_size) = T.shape(batch)

        ((H, C), _) = theano.scan(
            fn=self.lstm_step,
            outputs_info=[
                dict(initial=T.zeros((batch_size, self.hidden_size)), taps=[-1]),
                dict(initial=T.zeros((batch_size, self.hidden_size)), taps=[-1])],
            sequences=[batch, mask]       # ``X`` is a matrix whose row represents a word
        )

        return (H, C)

    def step(self, x, prev_h, prev_c):
        """Forward a time step.

        Reference: [1] LSTM: A Search Space Odyssey, Klaus Greff, Rupesh Kumar Srivastava, Jan Koutník,
                       Bas R. Steunebrink, Jürgen Schmidhuber, http://arxiv.org/abs/1503.04069
        """
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

        # tilde_a = T.dot(h, self.V.T)
        # y = self.activation(tilde_a)[0]  # T.nnet.softmax returns a 2D matrix

        return (h, c)

    def forward2(self, X):
        ((H, C), _) = theano.scan(
            fn=self.step,
            outputs_info=[
                dict(initial=np.zeros(self.hidden_size), taps=[-1]),
                dict(initial=np.zeros(self.hidden_size), taps=[-1])],
            sequences=[X]       # ``X`` is a matrix whose row represents a word
        )

        return (H, C)
