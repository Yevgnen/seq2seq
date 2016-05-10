#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.setrecursionlimit(40000)

import numpy as np
import theano
import theano.tensor as T
from numpy import random as rng
from theano import shared

from layers import LSTM, FullConnected, TimeDistributed
from models import Decoder, Encoder, Seq2seq, Sequential

# model = Sequential(
#     [
#         LSTM(4, 10)
#     ])

# X = np.array([[1, 2, 3, 2],
#               [1, 4, 7, 9],
#               [9, 8, 2, 7],
#               [9, 8, 2, 7],
#               [1, 2, 3, 2],
#               [1, 4, 7, 9],
#               [9, 8, 2, 7],
#               [9, 8, 2, 7]])
# T = np.array([1, 0, 2, 1, 1, 0, 2, 1])

# (H, C) = model.forward(X)

x = [0, 1, 4, 2, 5, 6, 1, 2, 4, 5]
y = [1, 4, 2, 5, 6, 1, 2, 4, 5, 9, 1, 2, 3, 4, 5]

X = rng.randint(low=0, high=10, size=((3000, 15)))
Y = rng.randint(low=0, high=10, size=((3000, 20)))

# encoder = Encoder(10, 6, 8)
# h = encoder.forward(x)

# decoder = Decoder(6, 12)
# Y = decoder.forward(h, len(x))

seq2seq = Seq2seq(10, 6, 8, 6, 12, 20)
# loss = seq2seq.each_loss(x, y)
loss = seq2seq.loss(X, Y)


import ipdb; ipdb.set_trace()
