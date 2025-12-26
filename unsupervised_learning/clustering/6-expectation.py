#!/usr/bin/env python3
"""Gaussian Mixture Model expectation step module"""
import numpy as np


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    
    n, d = X.shape
    k = pi.shape[0]
    
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    
    if not np.isclose(np.sum(pi), 1):
        return None, None
    
    pdf = __import__('5-pdf').pdf
    
    # Initialize numerator: pi[k] * P(X|mu[k], Sigma[k])
    # Shape: (k, n)
    numerator = np.zeros((k, n))
    
    # Calculate PDF of each cluster
    for i in range(k):
        P = pdf(X, m[i], S[i])
        
        if P is None:
            return None, None
        
        numerator[i] = pi[i] * P
    
    # Calculate denominator: sum of all numerators across clusters
    # Shape: (n,)
    denominator = np.sum(numerator, axis=0)
    
    # Avoid division by zero
    if np.any(denominator == 0):
        return None, None
    
    # Calculate posterior probabilities: g[k, n] = numerator[k, n] / denominator[n]
    g = numerator / denominator
    
    # Calculate log likelihood: sum(log(denominator))
    l = np.sum(np.log(denominator))
    
    return g, l
