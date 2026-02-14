#!/usr/bin/env python3
"""Updates weights and biases using gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases of a neural network using gradient descent
    with L2 regularization.

    Parameters:
    -----------
    Y : numpy.ndarray of shape (classes, m)
        One-hot correct labels
    weights : dict
        Dictionary of current weights and biases
    cache : dict
        Dictionary of layer activations
    alpha : float
        Learning rate
    lambtha : float
        L2 regularization parameter
    L : int
        Number of layers in the network
    """
    m = Y.shape[1]
    
    # Start backpropagation from output layer
    # Gradient of softmax + cross-entropy loss
    dZ = cache['A' + str(L)] - Y
    
    # Backpropagate through all layers
    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        
        # Compute gradients with L2 regularization term
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * weights['W' + str(layer)]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Update weights and biases in place
        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
        
        # Compute dZ for previous layer (don't compute for layer 0)
        if layer > 1:
            # Derivative of tanh: 1 - tanh(x)^2
            dZ = np.matmul(weights['W' + str(layer)].T, dZ) * (1 - np.power(cache['A' + str(layer - 1)], 2))