#!/usr/bin/env python3
import numpy as np
""" Script that updates the weights of a neural network """

def softmax(Z):
    """Compute softmax activation"""
    eZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return eZ / np.sum(eZ, axis=0, keepdims=True)


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout using gradient descent

    Parameters:
    Y: shape (classes, m), correct labels (one-hot)
    weights dictionary of weights and biases
    cache: dictionary of layer outputs and dropout masks
    alpha: learning rate
    keep_prob: probability of keeping a node
    L: number of layers in the network
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y  # Gradient for softmax output layer

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]

        # Compute gradients
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        # Backpropagate dZ for previous layer
        if l > 1:
            dA_prev = np.dot(W.T, dZ)

            # Apply dropout mask and scale
            D_prev = cache['D' + str(l - 1)]
            dA_prev *= D_prev
            dA_prev /= keep_prob

            # Derivative of tanh
            A_prev_raw = cache['A' + str(l - 1)]
            dZ = dA_prev * (1 - A_prev_raw ** 2)
