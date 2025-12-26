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
    
    # Use SVD for better numerical stability
    # X = U * S * V^T where S contains singular values
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Eigenvalues of the covariance matrix are related to singular values by:
    # eigenvalue = (singular_value^2) / (n - 1)
    # But we use n for consistency with the covariance definition
    eigenvalues = (S ** 2) / n
    
    # The eigenvectors are the columns of V (from SVD)
    # V = Vt.T, so eigenvectors are columns of Vt.T
    eigenvectors = Vt.T
    
    # Calculate cumulative explained variance
    total_variance = np.sum(eigenvalues)
    cumsum_variance = np.cumsum(eigenvalues) / total_variance
    
    # Find the number of components needed to maintain var fraction
    mask = cumsum_variance >= var
    indices = np.where(mask)[0]
    
    if len(indices) > 0:
        nd = indices[0] + 1
    else:
        # If no components meet the threshold, use all
        nd = len(eigenvalues)
    
    # Select the first nd eigenvectors (principal components)
    W = eigenvectors[:, :nd]
    
    # Ensure W is real-valued
    W = np.real(W)
    
    # Enforce sign convention: make the largest absolute value in each column positive
    for i in range(W.shape[1]):
        max_idx = np.abs(W[:, i]).argmax()
        if W[max_idx, i] < 0:
            W[:, i] *= -1
    
    return W