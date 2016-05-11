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

x1 = [2, 1, 1, 1, 2, 4, 2]
x2 = [2, 1]
x3 = [2, 1, 4, 3, 1]
x_value = np.asarray([x1, x2, x3])

y1 = [4, 3, 2, 1, 1]
y2 = [4, 3, 6]
y3 = [1, 2, 4, 3, 2, 1, 1]
y_value = np.asarray([y1, y2, y3])

mask_x_value = masking(x_value)
padded_x_value = padding(x_value, 0)
mask_x = shared(mask_x_value, name='mask_x')
padded_x = shared(padded_x_value, name='padded_x')

mask_y_value = masking(y_value)
padded_y_value = padding(y_value, 0)
mask_y = shared(mask_y_value, name='mask_y')
padded_y = shared(padded_y_value, name='padded_y')

encoder_vocab_size = 10
encoder_embedding_size = 4
encoder_hidden_size = 6

decoder_vocab_size = 8
decoder_embedding_size = 5
decoder_hidden_size = 6
decoder_output_size = 3

model = Seq2seq(encoder_vocab_size, encoder_embedding_size, encoder_hidden_size,
                decoder_vocab_size, decoder_embedding_size, decoder_hidden_size, decoder_output_size)

index2word = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six'}

# P = model.forward(padded_x, mask_x, padded_y, mask_y)
# loss = model.loss(padded_x, mask_x, padded_y, mask_y)
model.train(padded_x, mask_x, padded_y, mask_y, epoch=1000, batch_size=3, monitor=True)
predict = model.predict(padded_x, mask_x, padded_y, mask_y)
print(predict.eval())
