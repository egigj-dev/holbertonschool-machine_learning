#!/usr/bin/env python3
"""Bidirectional RNN Cell"""
import numpy as np


class BidirectionalCell:
    """Bidirectional RNN Cell class"""

    def __init__(self, i, h, o):
        """
        i: input dimension
        h: hidden state dimension
        o: output dimension
        """

        # Forward direction weights
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))

        # Backward direction weights
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))

        # Output weights
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step (forward direction)

        h_prev: shape (m, h)
        x_t: shape (m, i)

        Returns:
        h_next: shape (m, h)
        """

        # Concatenate previous hidden state and current input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state (forward direction)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)

        return h_next
