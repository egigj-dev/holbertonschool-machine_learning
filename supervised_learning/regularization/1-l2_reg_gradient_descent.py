#!/usr/bin/env python3
import numpy as np
""" Calculates the cost of a neural network using L2 """


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases of a neural network using gradient descent
    with L2 regularization.

    Parameters:
    Y (numpy.ndarray): shape (classes, m)
        Correct labels (one-hot)
    weights (dict): Dictionary of current weights and biases
    cache (dict): Dictionary of layer outputs
    alpha (float): Learning rate
    lambtha (float): L2 regularization parameter
    L (int): Number of layers in the network
    """

    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y  # Gradient at output layer (softmax)

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)] if l > 1 else cache['A0']
        W = weights['W' + str(l)]

        # Compute gradients with L2 regularization
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        # Compute dZ for previous layer
        if l > 1:
            A_prev_raw = cache['A' + str(l - 1)]
            dZ = np.dot(W.T, dZ) * (1 - A_prev_raw ** 2)  # tanh derivative
