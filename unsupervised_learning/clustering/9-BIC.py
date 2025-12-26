#!/usr/bin/env python3
"""Gaussian Mixture Model BIC model selection module"""
import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian Information Criterion.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    n, d = X.shape
    if not isinstance(kmin, int) or kmin <= 0 or kmin > n:
        return None, None, None, None
    # Set kmax to n if not provided
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmax > n:
        return None, None, None, None
    if kmin > kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None 
    if not isinstance(verbose, bool):
        return None, None, None, None
    expectation_maximization = __import__('8-EM').expectation_maximization
    # Arrays to store results
    num_k_values = kmax - kmin + 1
    l = np.zeros(num_k_values)
    b = np.zeros(num_k_values)
    results = []
    # Test each k value (LOOP - allowed to have at most 1)
    for i, k in enumerate(range(kmin, kmax + 1)):
        # Run EM algorithm
        pi, m, S, g, log_likelihood = expectation_maximization(X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None:
            return None, None, None, None
        # Store log likelihood
        l[i] = log_likelihood
        # Calculate number of parameters: p
        # p = k * d + k * d * (d + 1) / 2 + (k - 1)
        # This is: k priors (but sum to 1, so k-1 free) + k*d means + k*d*(d+1)/2 covariance elements
        p = k * d + k * d * (d + 1) // 2 + (k - 1)
        # Calculate BIC: p * ln(n) - 2 * l
        b[i] = p * np.log(n) - 2 * log_likelihood
        # Store result
        results.append((pi, m, S))
    # Find the best k (minimum BIC)
    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]
    return best_k, best_result, l, b
