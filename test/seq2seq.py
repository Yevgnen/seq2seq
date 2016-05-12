#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy import random as rng
from theano import shared

from models import Seq2seq
from optimizer import SGD, RMSprop
from utils import masking, padding

vocab_size = 6
sample_size = 3

x1 = [2, 1, 1, 1, 2, 4, 2]
x2 = [2, 1]
x3 = [2, 1, 4, 3, 1]
x_value = np.asarray([x1, x2, x3])

y1 = [4, 3, 2, 1, 1]
y2 = [4, 3, 6]
y3 = [1, 2, 4, 3, 2, 1, 1]
y_value = np.asarray([y1, y2, y3])

# def make_sentences(n, min_len=8, max_len=15, vocab_size=100):
#     sens = []
#     for i in range(n):
#         l = rng.randint(min_len, max_len)
#         s = rng.randint(1, vocab_size + 1, size=(l,), dtype='int32')
#         sens.append(s)

#     return np.array(sens)

# x_value = make_sentences(sample_size, vocab_size=vocab_size)
# y_value = make_sentences(sample_size, vocab_size=vocab_size)

mask_x_value = masking(x_value)
padded_x_value = padding(x_value, 0)
mask_x = shared(mask_x_value, name='mask_x')
padded_x = shared(padded_x_value, name='padded_x')

mask_y_value = masking(y_value)
padded_y_value = padding(y_value, 0)
mask_y = shared(mask_y_value, name='mask_y')
padded_y = shared(padded_y_value, name='padded_y')

encoder_vocab_size = vocab_size + 1
encoder_embedding_size = 4
encoder_hidden_size = 6

decoder_vocab_size = vocab_size + 1
decoder_embedding_size = 5
decoder_hidden_size = 6
decoder_output_size = 3

model = Seq2seq(encoder_vocab_size, encoder_embedding_size, encoder_hidden_size,
                decoder_vocab_size, decoder_embedding_size, decoder_hidden_size, decoder_output_size,
                RMSprop(lr=0.1, gamma=0.9, eps=1e-6))

dest_index2word = dict((i, str(i)) for i in range(vocab_size + 1))

# P = model.forward(padded_x, mask_x, padded_y, mask_y)
# loss = model.loss(padded_x, mask_x, padded_y, mask_y)
model.train(padded_x, mask_x, padded_y, mask_y, epoch=200, batch_size=3, monitor=True)
predict = model.predict(padded_x, mask_x, padded_y, mask_y)

for (sens_predict, sens_true) in zip(predict.eval(), padded_y.get_value(borrow=True)):
    predict = [dest_index2word[w] for w in sens_predict]
    true = [dest_index2word[w] for w in sens_true]
    print('Predict: {0}'.format(' '.join(predict)))
    print('   True: {0}'.format(' '.join(true)))
    print('-----------------------------------------------------------------------------------------\n')
