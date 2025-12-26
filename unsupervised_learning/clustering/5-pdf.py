#!/usr/bin/env python3
"""Gaussian Mixture Model PDF calculation module"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None   
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None    
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None   
    n, d = X.shape   
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None   
    # Calculate (X - m)
    diff = X - m  # shape (n, d)   
    # Calculate the determinant of S
    det_S = np.linalg.det(S)   
    if det_S <= 0:
        return None   
    # Calculate the inverse of S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None    
    # Calculate (X - m) @ S_inv @ (X - m).T of each point
    # Using matrix operations: sum((X - m) * (S_inv @ (X - m).T).T, axis=1)
    # Equivalent to: sum((X - m) * ((X - m) @ S_inv.T), axis=1)
    S_inv_diff = diff @ S_inv.T  # shape (n, d)    
    # Element-wise multiply and sum across dimensions
    mahalanobis_sq = np.sum(diff * S_inv_diff, axis=1)  # shape (n,)
    # Calculate normalization factor: 1 / sqrt((2*pi)^d * det(S))
    norm_factor = 1.0 / np.sqrt((2 * np.pi) ** d * det_S)    
    # Calculate PDF: norm_factor * exp(-0.5 * mahalanobis_sq)
    P = norm_factor * np.exp(-0.5 * mahalanobis_sq)    
    # Ensure minimum value of 1e-300
    P = np.maximum(P, 1e-300)    
    return P
