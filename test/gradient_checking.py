#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from numpy import random as rng
from theano import shared

from layers import FullConnected
from models import Sequential
from optimizer import GradientChecking

sample_size = 100
feature_size = 10
classes_num = 3
X_value = rng.uniform(size=(sample_size, feature_size))
Y_value = np.array(rng.randint(low=0, high=classes_num, size=(sample_size,)), dtype=np.int32)

X = shared(value=X_value, name='X', borrow=True)
Y = shared(value=Y_value, name='Y', borrow=True)

model = Sequential(
    [FullConnected(feature_size, 64),
     FullConnected(64, classes_num, activation='softmax')])

gc = GradientChecking(model)
gc.check_gradient(X_value, Y_value)
