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
    
    # Start with gradient at output layer (softmax)
    dZ = cache['A' + str(L)] - Y
    
    # Backpropagation through all layers
    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        
        # Compute gradients with L2 regularization
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Update weights and biases in place
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db
        
        # Compute dZ for previous layer (if not at input layer)
        if l > 1:
            # Derivative of tanh: 1 - tanh^2(x)
            dZ = np.matmul(W.T, dZ) * (1 - cache['A' + str(l - 1)] ** 2)