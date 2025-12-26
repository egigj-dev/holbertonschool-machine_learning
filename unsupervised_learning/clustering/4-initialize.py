#!/usr/bin/env python3
"""Gaussian Mixture Model initialization module"""
import numpy as np


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    n, d = X.shape
    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None

    # Initialize pi
    pi = np.ones(k) / k

    # Initialize m: use K-means to get centroids
    kmeans = __import__('1-kmeans').kmeans
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # Initialize S
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
