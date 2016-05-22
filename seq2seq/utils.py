#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
from matplotlib import pyplot as plt


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


def get_logger():
    logger = logging.getLogger('seq2seq')

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)

        # Log to file
        fh = logging.FileHandler('seq2seq.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Log to stdout
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


class Monitor(object):
    def __init__(self, monitor_acc=False):
        self.monitor_acc = monitor_acc

        fig_num = 1 + monitor_acc
        self.figure = plt.figure(figsize=(5.4 * fig_num, 3.6))

        self.loss_ax = self.figure.add_subplot(1, fig_num, 1)
        self.train_loss, = self.loss_ax.plot([], [], 'g-', label='training')
        self.valid_loss, = self.loss_ax.plot([], [], 'b-', label='validating')
        plt.title('Monitoring the training and validating lossess')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()

        if monitor_acc:
            self.acc_ax = self.figure.add_subplot(1, fig_num, 2)
            self.train_acc, = self.acc_ax.plot([], [], 'g-', label='training')
            self.valid_acc, = self.acc_ax.plot([], [], 'b-', label='validating')
            plt.title('Monitoring the training and validating accuracy')
            plt.xlabel('iteration')
            plt.ylabel('accuracy')
            plt.legend(loc=4)
            plt.grid()

    def update(self, train_losses, valid_losses=None, valid_freq=0, train_acces=None, valid_acces=None):
        self.train_loss.set_xdata(np.arange(len(train_losses)))
        self.train_loss.set_ydata(train_losses)

        if valid_losses and valid_freq > 0:
            self.valid_loss.set_xdata(np.arange(len(valid_losses)) * valid_freq)
            self.valid_loss.set_ydata(valid_losses)

        self.loss_ax.set_xlim(0, len(train_losses))
        self.loss_ax.set_ylim(0, np.max(train_losses) + 1)

        if self.monitor_acc and train_acces and valid_acces:
            self.train_acc.set_xdata(np.arange(len(train_acces)))
            self.train_acc.set_ydata(train_acces)

            self.valid_acc.set_xdata(np.arange(len(train_acces)))
            self.valid_acc.set_ydata(valid_acces)

            self.acc_ax.set_xlim(0, len(train_acces))
            self.acc_ax.set_ylim(0, 1)

        self.figure.canvas.draw()
        plt.pause(0.00001)

    def save(self):
        self.figure.savefig('figure.png')
