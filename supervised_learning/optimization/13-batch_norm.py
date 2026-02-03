#!/usr/bin/env python3
import numpy as np

def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization.

    Args:
        Z: numpy.ndarray of shape (m, n), unactivated outputs
        gamma: numpy.ndarray of shape (1, n), scale parameters
        beta: numpy.ndarray of shape (1, n), offset parameters
        epsilon: small float to avoid division by zero

    Returns:
        Normalized Z matrix
    """
    # Compute mean and variance across the batch
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    # Normalize
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    return gamma * Z_norm + beta
