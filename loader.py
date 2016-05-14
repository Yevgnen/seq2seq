#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import codecs
import itertools
import re

from sklearn.cross_validation import train_test_split
from theano import shared

from seq2seq.utils import masking, padding


def read_file(file):
    print('loading {0}...'.format(file))
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


def _load_data(path, delimiters, pad_char='_'):
    sens = read_file(path)
    sens = [re.split('|'.join(delimiters), s) for s in sens]

    words = set(itertools.chain(*sens))

    index2word = dict((i + 1, w) for (i, w) in enumerate(list(words)))
    word2index = dict((w, i + 1) for (i, w) in enumerate(list(words)))
    index2word[0] = pad_char
    word2index[pad_char] = 0

    sens_in_index = [[word2index[w] for w in s] for s in sens]

    return (sens_in_index, index2word, word2index)


def load_data(src_path, dest_path, train_size=0.8, valid_size=0):
    src_delimiters = [' ']
    dest_delimiters = [' ']

    (src_sens_in_index, src_index2word, src_word2index) = _load_data(
        src_path, src_delimiters)
    (dest_sens_in_index, dest_index2word, dest_word2index) = _load_data(
        dest_path, dest_delimiters)

    (train_x, test_x, train_y, test_y) = train_test_split(
        src_sens_in_index, dest_sens_in_index, train_size=0.8)

    if valid_size > 0:
        (train_x, valid_x, train_y, valid_y) = train_test_split(
            train_x, train_y, train_size=1 - valid_size)

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
