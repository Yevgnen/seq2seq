#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import codecs
import itertools
import re

import nltk
from numpy import random as rng
from sklearn.cross_validation import train_test_split
from theano import shared

from seq2seq.utils import get_logger, masking, padding

logger = get_logger()


def read_file(file):
    logger.info('loading {0}...'.format(file))
    data_file = codecs.open(file, 'r', 'utf-8')
    sentences = []

    while 1:
        line = data_file.readline()

        if not line:
            break

        if line.startswith('title') or line == '\r\n':
            continue

        line = line.strip()
        sentences.append(line)

    return sentences


def _load_data(path, delimiters, word_limit, pad_char='_', unknown_token='<U>'):
    sens = read_file(path)
    sens = [re.split('|'.join(delimiters), s) for s in sens]

    word_freq = nltk.FreqDist(itertools.chain(*sens))
    words = word_freq.most_common(word_limit)
    words = [word[0] for word in words]
    words = [pad_char, unknown_token] + words

    index2word = dict((i, w) for (i, w) in enumerate(words))
    word2index = dict((w, i) for (i, w) in enumerate(words))

    sens_in_index = [[word2index[w] if w in word2index else word2index[unknown_token] for w in s] for s in sens]

    return (sens_in_index, index2word, word2index)


def load_data(src_path, dest_path, src_word_limit, dest_word_limit, train_size=0.8, valid_size=0):
    src_delimiters = [' ']
    dest_delimiters = [' ']

    (src_sens_in_index, src_index2word, src_word2index) = _load_data(
        src_path, src_delimiters, src_word_limit)
    (dest_sens_in_index, dest_index2word, dest_word2index) = _load_data(
        dest_path, dest_delimiters, dest_word_limit)

    (train_x, test_x, train_y, test_y) = train_test_split(
        src_sens_in_index, dest_sens_in_index, train_size=0.8, random_state=rng.randint(10000))

    if valid_size > 0:
        (train_x, valid_x, train_y, valid_y) = train_test_split(
            train_x, train_y, train_size=1 - valid_size, random_state=rng.randint(10000))

    mask_train_x = masking(train_x)
    mask_test_x = masking(test_x)
    mask_train_y = masking(train_y)
    mask_test_y = masking(test_y)

    train_x = padding(train_x)
    test_x = padding(test_x)
    train_y = padding(train_y)
    test_y = padding(test_y)

    train_x = shared(train_x, name='train_x', borrow=True)
    test_x = shared(test_x, name='test_x', borrow=True)
    train_y = shared(train_y, name='train_y', borrow=True)
    test_y = shared(test_y, name='test_y', borrow=True)

    mask_train_x = shared(mask_train_x, name='mask_train_x', borrow=True)
    mask_test_x = shared(mask_test_x, name='mask_test_x', borrow=True)
    mask_train_y = shared(mask_train_y, name='mask_train_y', borrow=True)
    mask_test_y = shared(mask_test_y, name='mask_test_y', borrow=True)

    if valid_size > 0:
        mask_valid_x = masking(valid_x)
        mask_valid_y = masking(valid_y)
        valid_x = padding(valid_x)
        valid_y = padding(valid_y)
        valid_x = shared(valid_x, name='valid_x', borrow=True)
        valid_y = shared(valid_y, name='valid_y', borrow=True)
        mask_valid_x = shared(mask_valid_x, name='mask_valid_x', borrow=True)
        mask_valid_y = shared(mask_valid_y, name='mask_valid_y', borrow=True)

        return (train_x, test_x, valid_x, train_y, test_y, valid_y,
                mask_train_x, mask_test_x, mask_valid_x, mask_train_y, mask_test_y, mask_valid_y,
                src_index2word, src_word2index,
                dest_index2word, dest_word2index)
    else:
        return (train_x, test_x, train_y, test_y,
                mask_train_x, mask_test_x, mask_train_y, mask_test_y,
                src_index2word, src_word2index,
                dest_index2word, dest_word2index)
