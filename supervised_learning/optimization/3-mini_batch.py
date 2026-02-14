#!/usr/bin/env python3
import numpy as np
""" Script that creates mini-batches """

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for mini-batch gradient descent.
    """
    m = X.shape[0]

    # Shuffle data first
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    mini_batches = []

    # Create full mini-batches
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
