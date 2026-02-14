#!/usr/bin/env python3
import numpy as np
"""Script that calculates the normalization constants of a matrix"""


def normalization_constants(X):
    """
    Function that finds the normalization constants of a matrix
    Args:
        X (numpy.ndarray): Shape (m, nx), data to normalize
    Returns:
        mean (numpy.ndarray): Shape (nx,), mean of each feature
        std (numpy.ndarray): Shape (nx,), standard deviation of each feature
    """
    mean = np.mean(X, axis=0)  # 1D array of shape (nx,)
    std = np.std(X, axis=0)    # 1D array of shape (nx,)
    return mean, std
