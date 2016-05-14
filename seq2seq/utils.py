#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def padding(lst, digit=0):
    """Pad sentences of different lengths."""
    sens_num = len(lst)
    max_len = np.max([len(s) for s in lst])
    res = np.ones((sens_num, max_len), dtype='int32') * digit
    for (i, s) in enumerate(lst):
        res[i, 0: len(s)] = s
    return res


def masking(X):
    """Make masking matrix of a list of sentences."""
    max_len = np.max([len(x) for x in X])
    mask = np.zeros((max_len, len(X)), dtype='int32')
    for i, x in enumerate(X):
        mask[0: len(x), i] = np.ones_like(x)
    return mask
