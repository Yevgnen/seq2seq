#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from datetime import datetime

import numpy as np
import theano
import theano.tensor as T
from theano import shared

from layers import LSTM, Embedding, FullConnected, TimeDistributed
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


class Encoder(Sequential):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(embedding_size, hidden_size)
        self.layers = [self.embedding, self.lstm]
        self.params = [list(itertools.chain(*[layer.params for layer in self.layers]))]

    def forward(self, batch, mask):
        # ``batch`` is a matrix whose row ``x`` is a sentence, e.g. x = [1, 4, 5, 2, 0]
        emb = self.embedding.forward(batch)  # ``emb`` is a list of embedding matrix, e[i].shape = (sene_size, embedding_size)
        (H, C) = self.lstm.forward(emb, mask)
        return (H[-1], C[-1])

    def forward2(self, batch):
        # ``batch`` is a matrix whose row ``x`` is a sentence, e.g. x = [1, 4, 5, 2, 0]
        emb = self.embedding.forward(batch)  # ``emb`` is a list of embedding matrix, e[i].shape = (sene_size, embedding_size)
        (H, C) = self.lstm.forward2(emb)
        return (H[-1], C[-1])


class Decoder(Sequential):
    def __init__(self, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = T.nnet.softmax

        self.lstm = LSTM(output_size, hidden_size)
        self.output = TimeDistributed(hidden_size, output_size)
        self.layers = [self.lstm, self.output]
        self.params = [list(itertools.chain(*[layer.params for layer in self.layers]))]

    def forward(self, context, time):

        def step(y, h, c):
            """Forward a time step of the decoder."""
            (h, c) = self.lstm.step(y, h, c)
            y = self.output.forward(h)[0]  # T.nnet.softmax returns a 2D matrix
            return (y, h, c)

        results, updates = theano.scan(
            fn=step,
            outputs_info=[
                dict(initial=np.zeros(self.output_size), taps=[-1]),
                dict(initial=np.zeros(self.hidden_size), taps=[-1]),
                dict(initial=np.zeros(self.hidden_size), taps=[-1])],
            n_steps=time
        )

        return results[0]


class Seq2seq(object):
    def __init__(self, vocab_size, input_embedding_size, encoder_hidden_size,
                 decoder_hidden_size, output_size, output_embedding_size, optimizer=SGD()):
        self.vocab_size = vocab_size
        self.input_embedding_size = input_embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.output_embedding_size = output_embedding_size

        self.encoder = Encoder(vocab_size, input_embedding_size, encoder_hidden_size)
        self.decoder = Decoder(decoder_hidden_size, output_size)

        self.optimizer = optimizer

    def forward_sentence(self, x, y):
        # Encode
        (h, c) = self.encoder.forward(x)  # ``h`` is the context output by the encoder

        # Decode
        prediction = self.decoder.forward(h, len(y))  # Each row of ``prediction`` respects to a word
        return prediction

    def each_loss(self, x, y):
        prediction = self.forward_sentence(x, y)
        return -T.mean(T.sum(T.log(prediction)[T.arange(len(y)), y]))

    def loss(self, X, Y):
        loss = 0.0
        sens_num = X.shape[0]

        for (x, y) in zip(X, Y):
            loss += self.each_loss(x, y)

        return loss / sens_num
