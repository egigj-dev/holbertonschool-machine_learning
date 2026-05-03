#!/usr/bin/env python3
"""Bidirectional RNN forward propagation"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation for a bidirectional RNN

    bi_cell: instance of BidirectionalCell
    X: shape (t, m, i)
    h_0: initial forward hidden state (m, h)
    h_t: initial backward hidden state (m, h)

    Returns:
    H: shape (t, m, 2*h)
    Y: shape (t, m, o)
    """

    t, m, _ = X.shape
    h = h_0.shape[1]

    # Forward hidden states
    Hf = np.zeros((t, m, h))
    h_prev = h_0

    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        Hf[step] = h_prev

    # Backward hidden states
    Hb = np.zeros((t, m, h))
    h_next = h_t

    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        Hb[step] = h_next

    # Concatenate forward and backward hidden states
    H = np.concatenate((Hf, Hb), axis=2)  # (t, m, 2*h)

    # Compute outputs
    Y = bi_cell.output(H)

    return H, Y
