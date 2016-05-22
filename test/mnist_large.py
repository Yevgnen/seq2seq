#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cross_validation import train_test_split
from theano import shared

from layers import FullConnected
from loader import mnist
from models import Sequential
from optimizer import RMSprop


def load_data(data_dir):
    (train_x, test_x, train_y, test_y) = mnist(data_dir)
    (train_x, valid_x, train_y, valid_y) = train_test_split(train_x, train_y, train_size=0.8,
                                                            random_state=np.random.randint(10e6))
    return (shared(train_x, borrow=True),
            shared(test_x, borrow=True),
            shared(valid_x, borrow=True),
            shared(train_y, borrow=True),
            shared(test_y, borrow=True),
            shared(valid_y, borrow=True))


(train_x, test_x, valid_x, train_y, test_y, valid_y) = load_data('../data/mnist')

# Shared params
epoch = 5
batch_size = 1000

# Without dropout
model = Sequential(
    [FullConnected(784, 625, activation='relu'),
     FullConnected(625, 625, activation='relu'),
     FullConnected(625, 10, activation='softmax')],
    optimizer=RMSprop()
)

model.train(train_x, train_y, epoch=epoch, batch_size=batch_size,
            validation_data=(valid_x, valid_y), valid_freq=20, monitor=True)

score = model.score(test_x, test_y)
print('test score: {0}'.format(score.eval()))
