#!/usr/bin/env python3
"""RNN Cell"""
import numpy as np


class RNNCell:
    """RNN Cell class"""

    def __init__(self, i, h, o):
        """Constructor method"""
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Forward propagation method"""
        h_next = np.tanh(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                                   self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        return h_next, y
