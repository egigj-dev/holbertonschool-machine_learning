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

    def backward(self, h_next, x_t):
        """
        Perform backward propagation for one time step

        h_next: shape (m, h) → next hidden state
        x_t: shape (m, i) → current input

        Returns:
        h_prev: shape (m, h)
        """

        # Concatenate next hidden state with current input
        concat = np.concatenate((h_next, x_t), axis=1)

        # Compute previous hidden state (backward direction)
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)

        return h_prev


    def softmax(self, x):
        """Softmax activation function"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


    def output(self, H):
        """
        Calculate all outputs for the RNN

        H: shape (t, m, 2*h)

        Returns:
        Y: shape (t, m, o)
        """

        t, m, _ = H.shape
        o = self.by.shape[1]

        Y = np.zeros((t, m, o))

        for step in range(t):
            y_linear = np.matmul(H[step], self.Wy) + self.by
            Y[step] = self.softmax(y_linear)

        return Y
