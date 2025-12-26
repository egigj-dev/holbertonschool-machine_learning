#!/usr/bin/env python3
"""K-means variance calculation module"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance given a dataset
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    # Check dimensions match
    if X.shape[1] != C.shape[1]:
        return None
    n, d = X.shape
    k = C.shape[0]   
    # Calculate distances from each point to each centroid
    # X[:, np.newaxis] shape: (n, 1, d)
    # C shape: (k, d)
    # After broadcasting: (n, k, d)
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    # Find minimum distance (distance to nearest centroid)
    # Shape: (n,) - minimum distance
    min_distances = np.min(distances, axis=1)
    # Calculate total variance (sum of squared minimum distances)
    var = np.sum(min_distances ** 2)
    return var
