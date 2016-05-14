#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from datetime import datetime

import numpy as np
import theano
import theano.tensor as T
from matplotlib import pyplot as plt

from layers import LSTM, Embedding, TimeDistributed
from optimizer import SGD, RMSprop


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
        self.params = list(itertools.chain(*[layer.params for layer in self.layers]))

    def forward(self, batch, mask):
        # ``batch`` is a matrix whose row ``x`` is a sentence, e.g. x = [1, 4, 5, 2, 0]
        emb = self.embedding.forward(batch)  # ``emb`` is a list of embedding matrix, e[i].shape = (sene_size, embedding_size)
        (H, C) = self.lstm.forward(emb, mask)
        return (H[-1], C[-1])


class Decoder(Sequential):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = LSTM(embedding_size, hidden_size)
        self.lstm_output = TimeDistributed(hidden_size, output_size, activation='tanh')
        self.softmax = TimeDistributed(output_size, vocab_size, activation='softmax')
        self.embedding = Embedding(vocab_size, embedding_size)

        self.layers = [self.lstm, self.lstm_output, self.softmax, self.embedding]
        self.params = list(itertools.chain(*[layer.params for layer in self.layers]))

    def forward(self, ec_H, ec_C, mask):
        (sens_size, batch_size) = T.shape(mask)

        def step(m, prev_Y, prev_H, prev_C):
            """Forward a time step of the decoder."""
            # LSTM forward time step
            (H, C) = self.lstm.step(prev_Y, m, prev_H, prev_C)
            # LSTM output
            O = self.lstm_output.forward(H)
            # Apply softmax to LSTM output
            P = self.softmax.forward(O)
            # Make prediction
            one_hot_Y = T.argmax(P, axis=1)
            # Feed the output to the next time step
            Y = self.embedding.forward(one_hot_Y)
            # FIXME: Deal with differ length ?
            return (P, Y, H, C)

        results, updates = theano.scan(
            fn=step,
            sequences=[mask],
            outputs_info=[
                None,
                dict(initial=T.zeros((batch_size, self.embedding_size)), taps=[-1]),
                dict(initial=ec_H, taps=[-1]),
                dict(initial=ec_C, taps=[-1])
            ]
        )

        # return np.swapaxes(results[0], 0, 1)       # returns the softmax probabilities
        return results[0]


class Seq2seq(object):
    def __init__(self, encoder_vocab_size, encoder_embedding_size, encoder_hidden_size,
                 decoder_vocab_size, decoder_embedding_size, decoder_hidden_size, decoder_output_size,
                 optimizer=RMSprop()):
        self.encoder_vocab_size = encoder_vocab_size
        self.encoder_embedding_size = encoder_embedding_size
        self.encoder_hidden_size = encoder_hidden_size

        self.decoder_embedding_size = decoder_embedding_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_outout_size = decoder_output_size

        self.encoder = Encoder(encoder_vocab_size, encoder_embedding_size, encoder_hidden_size)
        self.decoder = Decoder(decoder_vocab_size, decoder_embedding_size, decoder_hidden_size, decoder_output_size)

        self.optimizer = optimizer

        self.params = self.encoder.params + self.decoder.params

    def forward(self, batch_x, mask_x, batch_y, mask_y):
        # Encode
        (H, C) = self.encoder.forward(batch_x, mask_x)

        # Decode
        probs = self.decoder.forward(H, C, mask_y)
        return probs

    def predict(self, batch_x, mask_x, batch_y, mask_y):
        batch_size = T.shape(batch_x)[0]

        probs = self.forward(batch_x, mask_x, batch_y, mask_y)

        def predict(prob, mask):
            valid_index = T.nonzero(mask > 0)[0]
            prob = prob[valid_index]
            word_index = T.zeros((batch_size,), dtype='int32')
            word_index = T.set_subtensor(word_index[valid_index], T.argmax(prob, axis=1))  # +1?
            return word_index

        results, updates = theano.scan(
            fn=predict,
            sequences=[probs, mask_y]
        )
        # FIXME: Symbols or numbers ?
        return np.swapaxes(results, 0, 1)

    def loss(self, batch_x, mask_x, batch_y, mask_y):
        # The time steps should be the length of longest sentence of the batch.
        # time = mask_y.shape[0] will cause NaN.
        loss = 0.0
        time = T.max(T.sum(mask_y, axis=0))
        batch_size = T.shape(batch_x)[0]

        probs = self.forward(batch_x, mask_x, batch_y, mask_y)

        def loss_of_time(prob, y, mask):
            valid_index = T.nonzero(mask > 0)[0]
            # FIXME: why y log twice ?
            loss = -T.sum(T.log(prob[valid_index, y[valid_index]]))
            return loss

        results, updates = theano.scan(
            fn=loss_of_time,
            sequences=[probs, batch_y.T, mask_y],
            n_steps=time
        )

        loss = T.sum(results) / batch_size

        return loss

    def train(self, train_x, mask_train_x, train_y, mask_train_y, epoch=10, batch_size=128, monitor=False,
              epoch_end_callback=None):
        sample_num = train_x.get_value(borrow=True).shape[0]

        batch_index = T.iscalar('batch_index')
        batch_num = int(np.ceil(sample_num / batch_size))

        x = T.imatrix('x')
        y = T.imatrix('y')
        m_x = T.imatrix('m_x')
        m_y = T.imatrix('m_y')

        loss = self.loss(x, m_x, y, m_y)

        updates = self.optimizer.get_updates(loss, self.params)

        train_model = theano.function(
            inputs=[batch_index],
            outputs=loss,
            updates=updates,
            givens=[
                (x, train_x[batch_index * batch_size: (batch_index + 1) * batch_size]),
                (y, train_y[batch_index * batch_size: (batch_index + 1) * batch_size]),
                (m_x, mask_train_x[:, batch_index * batch_size: (batch_index + 1) * batch_size]),
                (m_y, mask_train_y[:, batch_index * batch_size: (batch_index + 1) * batch_size])
            ]
        )

        if monitor:
            plt.figure(figsize=(6, 4))
            plt.xlabel('Update count')
            plt.ylabel('Loss')
            plt.title('Monitoring')

        train_losses = []
        updates_count = 0
        for i in range(epoch):
            for j in range(batch_num):
                batch_loss = train_model(j)
                train_losses.append(batch_loss)
                timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('{0} - TRAINING - epoch: {1:3d}, batch: {2:3d}, loss: {3}'.format(timestr, i + 1, j + 1, batch_loss))

                if monitor:
                    if len(train_losses) > 1:
                        (x_min, x_max) = (0, i * batch_num + j + 1)
                        (y_min, y_max) = (0, np.max(train_losses) + 1)
                        plt.xlim(x_min, x_max)
                        plt.ylim(y_min, y_max)
                        this_x = i * batch_num + j
                        plt.plot([this_x - 1, this_x],
                                 [train_losses[-2], train_losses[-1]], 'g-', lw=1, label='train')
                    plt.pause(0.001)
                updates_count += 1

            if epoch_end_callback and callable(epoch_end_callback):
                epoch_end_callback()

        if monitor:
            plt.savefig('train.png')

        return np.asarray(train_losses).reshape(epoch, batch_num)
