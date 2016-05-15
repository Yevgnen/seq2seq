#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from numpy import random as rng

from loader import load_data
from seq2seq.models import Decoder, Encoder, Seq2seq
from seq2seq.optimizer import SGD, RMSprop
from seq2seq.utils import get_logger

# start
logger = get_logger()
logger.info('--------------- Encoder - Decoder ---------------')

# load data
logger.info('loading data...')
(train_x, test_x,
 train_y, test_y,
 mask_train_x, mask_test_x,
 mask_train_y, mask_test_y,
 src_index2word, src_word2index,
 dest_index2word, dest_word2index) = load_data('data/train2000.ja', 'data/train2000.en', train_size=0.8)

logger.info('loaded training source senstences: {0}, max length: {1}, vocabulary size: {2}'.format(
    len(train_x.get_value(borrow=True)), len(mask_train_x.get_value(borrow=True)), len(list(src_word2index))))
logger.info('loaded training target senstences: {0}, max length: {1}, vocabulary size: {2}'.format(
    len(train_y.get_value(borrow=True)), len(mask_train_y.get_value(borrow=True)), len(list(dest_word2index))))

# encoder
encoder_vocab_size = len(src_word2index)
encoder_embedding_size = 100
encoder_hidden_size = 50
encoder = Encoder(encoder_vocab_size, encoder_embedding_size, encoder_hidden_size)

# decoder
decoder_vocab_size = len(dest_word2index)
decoder_embedding_size = 100
decoder_hidden_size = 50
decoder_output_size = 100
decoder = Decoder(decoder_vocab_size, decoder_embedding_size, decoder_hidden_size, decoder_output_size)

# Sequential to sequential learning model
model = Seq2seq(encoder, decoder,
                # RMSprop(clip=5.0, lr=0.003, gamma=0.9, eps=1e-8),
                SGD(clip=0.5, lr=0.5),
                logger=logger)


# training
def epoch_end_callback():
    logger.info('sampling...')
    sample_size = 5
    sample_indices = rng.randint(0, train_x.get_value(borrow=True).shape[0], sample_size)
    predict = model.predict(train_x, mask_train_x, train_y, mask_train_y)
    sample_x = train_x.get_value(borrow=True)[sample_indices]
    sample_y = train_y.get_value(borrow=True)[sample_indices]
    predict_y = predict.eval()[sample_indices]
    for (source, predict, target) in zip(sample_x, predict_y, sample_y):
        source = [src_index2word[w] for w in source]
        predict = [dest_index2word[w] for w in predict]
        target = [dest_index2word[w] for w in target]
        logger.info(' Source: {0}'.format(' '.join(source)))
        logger.info(' Target: {0}'.format(' '.join(target)))
        logger.info('Predict: {0}'.format(' '.join(predict)))
        logger.info('-------------------------------------------------')

logger.info('training...')
epoch = 600
batch_size = 100
model.train(train_x, mask_train_x, train_y, mask_train_y,
            epoch=epoch, batch_size=batch_size, monitor=True,
            epoch_end_callback=epoch_end_callback)

# predicting
predict = model.predict(test_x, mask_test_x, test_y, mask_test_y)
logger.info('predicting...')
for (source, predict, target) in zip(test_x.get_value(borrow=True), predict.eval(), test_y.get_value(borrow=True)):
    source = [src_index2word[w] for w in source]
    predict = [dest_index2word[w] for w in predict]
    target = [dest_index2word[w] for w in target]
    logger.info(' Source: {0}'.format(' '.join(source)))
    logger.info(' Target: {0}'.format(' '.join(target)))
    logger.info('Predict: {0}'.format(' '.join(predict)))
    logger.info('-------------------------------------------------')
