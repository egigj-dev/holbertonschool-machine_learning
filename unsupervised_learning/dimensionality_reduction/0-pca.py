#!/usr/bin/env python3
"""Principal Component Analysis (PCA) module"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(var, (int, float)) or var <= 0 or var > 1:
        return None

    # Compute SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Eigenvalues of covariance matrix
    eigenvalues = (S ** 2) / X.shape[0]

    # Principal components
    W = Vt.T

    # Determine how many components to keep
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    nd = np.searchsorted(cum_var, var) + 1
    W = W[:, :nd]

    # Optional: enforce sign convention
    for i in range(W.shape[1]):
        max_idx = np.abs(W[:, i]).argmax()
        if W[max_idx, i] < 0:
            W[:, i] *= -1

    return W
