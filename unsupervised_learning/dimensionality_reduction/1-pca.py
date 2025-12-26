#!/usr/bin/env python3
"""Principal Component Analysis (PCA) transformation module"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    
    n, d = X.shape
    
    if not isinstance(ndim, int) or ndim <= 0 or ndim > d:
        return None
    
    # Center the data by subtracting the mean
    X_m = X - np.mean(X, axis=0)
    
    # Compute the Singular Value Decomposition (SVD)
    U, S, VT = np.linalg.svd(X_m, full_matrices=False)
    
    # Extract the first ndim right singular vectors (principal components)
    W = VT[:ndim].T
    
    # Transform X by projecting onto the principal components
    T = np.dot(X_m, W)
    
    return T