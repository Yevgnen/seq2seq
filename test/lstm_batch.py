#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
import theano
import theano.tensor as T
from numpy import random as rng
from theano import shared

from layers import LSTM, FullConnected, TimeDistributed
from models import Decoder, Encoder, Seq2seq, Sequential
from utils import masking, padding

rng.seed(123)

# Preprocess data
x1 = [2, 1, 1, 1, 2, 4, 2]
x2 = [2, 1]
x3 = [2, 1, 4, 3, 1]
batch_value = np.asarray([x1, x2, x3])

vocab_size = 5
embedding_size = 4
encoder_hidden_size = 6

encoder = Encoder(vocab_size + 1, embedding_size, encoder_hidden_size)
mask_value = masking(batch_value)
padded_batch_value = padding(batch_value, 0)

mask = shared(mask_value, name='mask')
padded_batch = shared(padded_batch_value, name='padded_batch')
H, C = encoder.forward(padded_batch, mask)

(h1, c1) = encoder.forward2(x1)
(h2, c2) = encoder.forward2(x2)
(h3, c3) = encoder.forward2(x3)

print(T.isclose(H, T.as_tensor_variable([h1, h2, h3])).eval())
print(T.isclose(C, T.as_tensor_variable([c1, c2, c3])).eval())
