#!/usr/bin/env python3
"""K-means clustering initialization module"""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means"""

    # # Input validation
    n, d = X.shape
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if isinstance(k, int) is False or k <= 0:
        return None
    
    # Check if k is reasonable (not more data points than we have)
    if k > n:
        return None
    
    # Find minimum and maximum values along each dimension
    low = X.min(axis=0)
    high = X.max(axis=0)

    # Randomly generate centroids
    return np.random.uniform(low, high, size=(k, d))
