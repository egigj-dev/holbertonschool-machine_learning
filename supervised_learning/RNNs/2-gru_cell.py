#!/usr/bin/env python3
"""GRU Cell"""
import numpy as np


class GRUCell:
    """GRU Cell class"""

    def __init__(self, i, h, o):
        """Constructor"""
        # Update gate
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        # Reset gate
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))

        # Candidate hidden state
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        # Output
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Softmax activation"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Forward propagation for one time step"""
        # Concatenate input and previous hidden state
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))

        # Reset gate
        r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

        # Candidate hidden state
        concat_reset = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, y
