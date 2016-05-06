#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from theano import shared

from layers import FullConnected
from models import Sequential
from optimizer import SGD


def load_data(datasets):
    (all_x, all_y) = (dataset.data, np.asarray(dataset.target, dtype=np.int32))
    (train_x, test_x, train_y, test_y) = train_test_split(all_x, all_y, train_size=0.8)
    return (shared(train_x, borrow=True),
            shared(test_x, borrow=True),
            shared(train_y, borrow=True),
            shared(test_y, borrow=True))

dataset = datasets.load_digits()
(train_x, test_x, train_y, test_y) = load_data(dataset)

model = Sequential(
    [FullConnected(64, 128),
     FullConnected(128, 10, activation='softmax')],
    optimizer=SGD(lr=0.0001, decay=.001, momentum=0.9)
)

model.train(train_x, train_y, epoch=200, batch_size=train_x.shape.eval()[0])

score = model.score(test_x.get_value(borrow=True), test_y.get_value(borrow=True))
print('test score: {0}'.format(score))
