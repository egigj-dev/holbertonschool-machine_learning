#!/usr/bin/env python3
"""Optimum K-means cluster finder module"""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.
    """
    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance
    
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    
    n, d = X.shape
    
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None
    
    # Set kmax to n if not provided
    if kmax is None:
        kmax = n
    
    if not isinstance(kmax, int) or kmax <= 0 or kmax >= n:
        return None, None
    
    if kmin >= kmax:
        return None, None
    
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    
    # Must analyze at least 2 different cluster sizes
    if kmax - kmin < 1:
        return None, None
    
    results = []
    variances = []
    
    # Loop through each k value (LOOP #1)
    for k in range(kmin, kmax + 1):
        # Run K-means for this k
        C, clss = kmeans(X, k, iterations)
        
        if C is None or clss is None:
            return None, None
        
        # Store the result (centroids and cluster assignments)
        results.append((C, clss))
        
        # Calculate variance for this clustering
        var = variance(X, C)
        
        if var is None:
            return None, None
        
        variances.append(var)
    
    # Calculate difference in variance from smallest cluster size (k=kmin)
    # The smallest variance is at index 0 (k=kmin)
    smallest_var = variances[0]
    
    # Calculate differences (LOOP #2 - but can be vectorized)
    d_vars = [smallest_var - var for var in variances]
    
    return results, d_vars
