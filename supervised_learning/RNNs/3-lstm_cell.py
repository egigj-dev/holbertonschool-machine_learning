#!/usr/bin/env python3
"""LSTM Cell"""
import numpy as np


class LSTMCell:
    """LSTM Cell class"""

    def __init__(self, i, h, o):
        """Constructor"""
        # Forget gate
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))

        # Update (input) gate
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))

        # Candidate cell state
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))

        # Output gate
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))

        # Output layer
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """Forward propagation for one time step"""
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Gates
        f = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)  # forget gate
        u = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)  # update gate
        o = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)  # output gate

        # Candidate cell state
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # Cell state update
        c_next = f * c_prev + u * c_tilde

        # Hidden state update
        h_next = o * np.tanh(c_next)

        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, c_next, y
