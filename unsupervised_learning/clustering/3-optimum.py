#!/usr/bin/env python3
"""Optimum K-means cluster finder module"""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Testing the optimum number of clusters by variance.
    """

    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmax >= n:
        return None, None
    if kmin >= kmax:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmax - kmin < 1:
        return None, None

    results = []
    d_vars = []
    base_var = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        results.append((C, clss))

        var = variance(X, C)
        if var is None:
            return None, None

        if base_var is None:      
            base_var = var

        # append difference relative to baseline
        d_vars.append(base_var - var)

    return results, d_vars
