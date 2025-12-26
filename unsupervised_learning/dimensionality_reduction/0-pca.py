#!/usr/bin/env python3
"""Principal Component Analysis (PCA) module"""
import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset and returns the transformed data and principal components.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(var, (int, float)) or var <= 0 or var > 1:
        return None

    n, d = X.shape

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigenvalues = (S ** 2) / n
    eigenvectors = Vt.T

    # Cumulative explained variance
    cumsum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    nd = np.searchsorted(cumsum_var, var) + 1

    # Select principal components
    W = eigenvectors[:, :nd]

    # Sign convention: largest absolute value positive
    for i in range(W.shape[1]):
        max_idx = np.abs(W[:, i]).argmax()
        if W[max_idx, i] < 0:
            W[:, i] *= -1

    # Project X onto principal components
    X_transformed, W = pca(X)
    print(X_transformed)
    print(X_transformed.shape)
    print(W)
    print(W.shape)

    return X_transformed, W
