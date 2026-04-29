#!/usr/bin/env python3
"""Deep RNN Forward Propagation"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN"""
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    # Initialize outputs
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    # Set initial hidden states
    H[0] = h_0

    for step in range(t):
        x = X[step]

        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]

            # Input is X for first layer, otherwise previous layer's output
            if layer == 0:
                h_next, y = cell.forward(h_prev, x)
            else:
                h_next, y = cell.forward(h_prev, H[step + 1, layer - 1])

            H[step + 1, layer] = h_next

        # Output comes from last layer
        Y[step] = y

    return H, Y
