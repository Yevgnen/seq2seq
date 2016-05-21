#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from theano import shared

from seq2seq.layers import FullConnected
from seq2seq.models import Sequential
from seq2seq.optimizer import SGD


def load_data(datasets, test=0.2, valid=0.0):
    (all_x, all_y) = (dataset.data, np.asarray(dataset.target, dtype=np.int32))
    (train_x, test_x, train_y, test_y) = train_test_split(all_x, all_y, train_size=1 - test)
    if valid > 0:
        (train_x, valid_x, train_y, valid_y) = train_test_split(all_x, all_y, train_size=1 - valid)
        return (shared(train_x, borrow=True),
                shared(test_x, borrow=True),
                shared(valid_x, borrow=True),
                shared(train_y, borrow=True),
                shared(test_y, borrow=True),
                shared(valid_y, borrow=True))
    else:
        return (shared(train_x, borrow=True),
                shared(test_x, borrow=True),
                shared(train_y, borrow=True),
                shared(test_y, borrow=True))

dataset = datasets.load_digits()
(train_x, test_x, valid_x, train_y, test_y, valid_y) = load_data(dataset, valid=0.2)

model = Sequential(
    [FullConnected(64, 128),
     FullConnected(128, 10, activation='softmax')],
    optimizer=SGD(lr=0.0001, decay=.001, momentum=0.9)
)

model.train(train_x, train_y, epoch=1000, batch_size=500,
            validation_data=(valid_x, valid_y), valid_freq=5)

score = model.score(test_x, test_y)
print('test score: {0}'.format(score.eval()))
