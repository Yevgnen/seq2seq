#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from theano import shared

from layers import Dropout, FullConnected
from loader import mnist
from models import Sequential
from optimizer import RMSprop


def load_data(data_dir):
    (train_x, test_x, train_y, test_y) = mnist(data_dir, onehot=False)
    return (shared(train_x, borrow=True),
            shared(test_x, borrow=True),
            shared(train_y, borrow=True),
            shared(test_y, borrow=True))


(train_x, test_x, train_y, test_y) = load_data('../data/mnist')

# Shared params
epoch = 2
batch_size = 128

# Without dropout
model = Sequential(
    [FullConnected(784, 625, activation='relu'),
     FullConnected(625, 625, activation='relu'),
     FullConnected(625, 10, activation='softmax')],
    optimizer=RMSprop()
)

model.train(train_x, train_y, epoch=epoch, batch_size=batch_size)
score1 = model.score(test_x.get_value(borrow=True), test_y.get_value(borrow=True))

# With dropout
model = Sequential(
    [Dropout(0.2),
     FullConnected(784, 625, activation='relu'),
     Dropout(0.5),
     FullConnected(625, 625, activation='relu'),
     Dropout(0.5),
     FullConnected(625, 10, activation='softmax')],
    optimizer=RMSprop()
)

model.train(train_x, train_y, epoch=epoch, batch_size=batch_size)
score2 = model.score(test_x.get_value(borrow=True), test_y.get_value(borrow=True))

print('Without dropout - test score: {0}'.format(score1))
print('   With dropout - test score: {0}'.format(score2))
