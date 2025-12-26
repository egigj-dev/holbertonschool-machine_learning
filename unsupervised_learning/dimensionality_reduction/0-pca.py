#!/usr/bin/env python3
"""Principal Component Analysis (PCA) module"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    
    Args:
        X: numpy.ndarray of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each point
            - all dimensions have a mean of 0 across all data points
        var: the fraction of the variance that the PCA transformation should maintain
    
    Returns:
        W: numpy.ndarray of shape (d, nd) containing the weights matrix that maintains
           var fraction of X's original variance
           - nd is the new dimensionality of the transformed X
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    
    if not isinstance(var, (int, float)) or var <= 0 or var > 1:
        return None
    
    n, d = X.shape
    
    # Compute the covariance matrix
    # Covariance: C = (1/n) * X^T * X (since X is already centered)
    C = np.dot(X.T, X) / n
    
    # Compute eigenvalues and eigenvectors
    # eigenvalues are in ascending order
    eigenvalues, eigenvectors = np.linalg.eig(C)
    
    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate cumulative explained variance
    # Total variance is the sum of all eigenvalues
    total_variance = np.sum(eigenvalues)
    cumsum_variance = np.cumsum(eigenvalues) / total_variance
    
    # Find the number of components needed to maintain var fraction
    # Find the first index where cumulative variance >= var
    mask = cumsum_variance >= var
    nd = np.where(mask)[0]
    
    if len(nd) > 0:
        nd = nd[0] + 1
    else:
        # If no components meet the threshold, use all
        nd = len(eigenvalues)
    
    # Select the first nd eigenvectors (principal components)
    W = eigenvectors[:, :nd]
    
    # Ensure W is real-valued (in case of numerical errors with complex types)
    W = np.real(W)
    
    # Enforce sign convention: make the largest absolute value in each column positive
    for i in range(W.shape[1]):
        if np.abs(W[:, i]).argmax() > 0:
            # Find the element with largest absolute value
            max_idx = np.abs(W[:, i]).argmax()
            # If that element is negative, flip the entire column
            if W[max_idx, i] < 0:
                W[:, i] *= -1
    
    return W