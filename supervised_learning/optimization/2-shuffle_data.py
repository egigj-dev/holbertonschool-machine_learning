#!/usr/bin/env python3
import numpy as np
""" Script that shuffles two matrices the same order """

def shuffle_data(X, Y):
    """
    Shuffles two matrices the same order along the first axis.
    """
    # Number of data points
    m = X.shape[0]

    # Generate a random permutation of indices
    perm = np.random.permutation(m)

    # Apply permutation to both matrices
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]

    return X_shuffled, Y_shuffled
