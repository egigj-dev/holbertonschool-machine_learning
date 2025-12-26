#!/usr/bin/env python3
"""Principal Component Analysis (PCA) module"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    
    if not isinstance(var, (int, float)) or var <= 0 or var > 1:
        return None
    
    n, d = X.shape
    
    # Compute the covariance matrix
    C = np.dot(X.T, X) / n
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(C)
    
    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate cumulative explained variance
    total_variance = np.sum(eigenvalues)
    cumsum_variance = np.cumsum(eigenvalues) / total_variance
    
    # Find the number of components needed to maintain var fraction
    nd = np.argmax(cumsum_variance >= var) + 1
    
    # Select the first nd eigenvectors (principal components)
    W = eigenvectors[:, :nd]
    
    # Ensure W is real-valued (in case of numerical errors with complex types)
    W = np.real(W)
    
    return W