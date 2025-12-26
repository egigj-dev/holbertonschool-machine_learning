#!/usr/bin/env python3
"""Gaussian Mixture Model EM algorithm module"""
import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    
    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization
    
    # Initialize parameters
    pi, m, S = initialize(X, k)
    
    if pi is None or m is None or S is None:
        return None, None, None, None, None
    
    # Perform EM algorithm (LOOP - allowed to have at most 1)
    prev_l = -np.inf
    
    for i in range(iterations):
        # E-step
        g, l = expectation(X, pi, m, S)
        
        if g is None or l is None:
            return None, None, None, None, None
        
        # Print log likelihood if verbose
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {l:.5f}")
        
        # Check convergence
        if abs(l - prev_l) <= tol:
            if verbose and i % 10 != 0 and i != iterations - 1:
                print(f"Log Likelihood after {i} iterations: {l:.5f}")
            break
        
        # M-step
        pi, m, S = maximization(X, g)
        
        if pi is None or m is None or S is None:
            return None, None, None, None, None
        
        prev_l = l
    
    # Final E-step to get final g and l
    g, l = expectation(X, pi, m, S)
    
    if g is None or l is None:
        return None, None, None, None, None
    
    return pi, m, S, g, l
