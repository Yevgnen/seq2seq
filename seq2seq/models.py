#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from datetime import datetime

import numpy as np
import theano
import theano.tensor as T
from theano import shared

from optimizer import SGD


class Sequential(object):
    def __init__(self, layers, loss='cross_entropy', optimizer=SGD()):
        self.layers = layers
        self.params = list(itertools.chain(*[layer.params for layer in layers]))
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, test_x):
        x = T.dmatrix('x')
        prediction = np.argmax(self.forward(x), axis=1)

        predict_fn = theano.function(
            inputs=[x],
            outputs=prediction
        )

        return predict_fn(test_x)

    def score(self, test_x, test_y):
        return np.mean(np.equal(self.predict(test_x), test_y))

    def loss(self, x, y):
        x = self.forward(x)
        return -T.mean(T.sum(T.log(x)[T.arange(y.shape[0]), y]))

    def train(self, train_set_x, train_set_y, epoch=10, batch_size=128, valid_x=None, valid_y=None):
        sample_num = train_set_x.get_value(borrow=True).shape[0]

        batch_index = T.iscalar('batch_index')
        batch_num = sample_num // batch_size

        x = T.dmatrix('x')
        y = T.ivector('y')

        loss = self.loss(x, y)

        updates = self.optimizer.get_updates(loss, self.params)

        train_model = theano.function(
            inputs=[batch_index],
            outputs=loss,
            updates=updates,
            givens=[
                (x, train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size]),
                (y, train_set_y[batch_index * batch_size: (batch_index + 1) * batch_size])]
        )

        for i in range(epoch):
            for j in range(batch_num):
                batch_loss = train_model(j)
                timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('{0} - epoch: {1:3d}, batch: {2:3d}, loss: {3}'.format(timestr, i + 1, j + 1, batch_loss))
        return
