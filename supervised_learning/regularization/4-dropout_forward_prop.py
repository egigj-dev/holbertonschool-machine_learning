#!/usr/bin/env python3
import numpy as np
from scipy.special import softmax  	# possible
""" Script that conducts forward propagation using Dropout """

def softmax(Z):
    """Compute softmax activation for the output layer."""
    eZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return eZ / np.sum(eZ, axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Parameters:
    X: shape (nx, m) input data
    weights: Dictionary of weights and biases
    L: Number of layers
    keep_prob: Probability of keeping a node
    """
    cache = {}
    cache['A0'] = X

    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        A_prev = cache['A' + str(l - 1)]
        Z = np.dot(W, A_prev) + b

        # Apply activation
        if l == L:
            A = softmax(Z)
        else:
            A = np.tanh(Z)

            # Apply dropout
            D = (np.random.rand(*A.shape) < keep_prob).astype(float)
            A *= D
            A /= keep_prob  # scale to maintain expected value

            # Store dropout mask
            cache['D' + str(l)] = D

        cache['A' + str(l)] = A

    return cache
