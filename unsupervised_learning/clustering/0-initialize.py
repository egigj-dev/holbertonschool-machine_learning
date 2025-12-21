#!/usr/bin/env python3
"""K-means clustering initialization module"""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means"""
    
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    
    n, d = X.shape
    if k > n:
        return None
    
    # Find minimum and maximum values along each dimension
    low = X.min(axis=0)
    high = X.max(axis=0)
    
    # Randomly generate centroids
    return np.random.uniform(low, high, size=(k, d))
