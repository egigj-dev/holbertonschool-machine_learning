#!/usr/bin/env python3
"""Gaussian Mixture Model maximization step module"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    
    n, d = X.shape
    k = g.shape[0]
    
    if g.shape[1] != n:
        return None, None, None
    
    # Calculate Nk: sum of responsibilities of each cluster
    # Shape: (k,)
    Nk = np.sum(g, axis=1)
    
    # Check if any zero responsibilities
    if np.any(Nk == 0):
        return None, None, None
    
    # Calculate pi: Nk / n
    pi = Nk / n
    
    # Calculate m: weighted sum of X divided by Nk
    # m[k] = (1 / Nk[k]) * sum_n(g[k, n] * X[n])
    # Shape: (k, d)
    m = (g @ X) / Nk[:, np.newaxis]
    
    # Calculate S: covariance matrices
    # S[k] = (1 / Nk[k]) * sum_n(g[k, n] * (X[n] - m[k]) * (X[n] - m[k]).T)
    # Shape: (k, d, d)
    S = np.zeros((k, d, d))
    
    # Loop over clusters to calculate covariance (allowed 1 loop)
    for i in range(k):
        # Calculate differences: X - m[i]
        # Shape: (n, d)
        diff = X - m[i]
        
        # Calculate weighted outer product: sum_n(g[i, n] * diff[n] * diff[n].T)
        # Using: (g[i] * diff.T) @ diff
        # Shape: (d, d)
        S[i] = (g[i] * diff.T) @ diff / Nk[i]
    
    return pi, m, S
