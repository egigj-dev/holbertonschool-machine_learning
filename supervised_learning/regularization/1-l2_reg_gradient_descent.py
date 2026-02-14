#!/usr/bin/env python3
import numpy as np
"""Updates weights and biases of a neural network using L2 regularization"""


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Parameters:
    Y (numpy.ndarray): shape (classes, m), one-hot true labels
    weights (dict): dictionary of weights and biases
    cache (dict): dictionary of activations per layer
    alpha (float): learning rate
    lambtha (float): L2 regularization parameter
    L (int): number of layers in the network
    """
    m = Y.shape[1]
    # Gradient at the output layer (softmax)
    dZ = cache['A' + str(L)] - Y

    # Loop backward through all layers
    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]

        # Compute gradients with L2 regularization
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases in place
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        # Compute dZ for the next layer (if not input layer)
        if l > 1:
            A_prev_raw = A_prev  # tanh activation
            dZ = np.dot(W.T, dZ) * (1 - A_prev_raw ** 2)  # derivative of tanh
