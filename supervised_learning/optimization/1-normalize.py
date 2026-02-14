#!/usr/bin/env python3
import numpy as np
"""Script that calculates the normalization constants of a matrix."""


def normalize(X, m, s):
    """
    Function that normalizes a matrix.
    Parameters:
    X : np.ndarray
        Input matrix to normalize
    m : np.ndarray
        Mean values for each feature
    s : np.ndarray
        Standard deviation values for each feature
    """
    # Avoid division by zero by replacing zeros in std with 1
    s_safe = np.where(s == 0, 1, s)

    # Standardization: (X - mean) / std
    X_norm = (X - m) / s_safe

    return X_norm
