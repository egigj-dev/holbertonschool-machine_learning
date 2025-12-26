#!/usr/bin/env python3
"""PCA dimensionality reduction"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on X and reduces it to ndim dimensions.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    n, d = X.shape
    if not isinstance(ndim, int) or ndim <= 0 or ndim > d:
        return None

    # Center the data (zero mean)
    X_centered = X - np.mean(X, axis=0)

    # Compute SVD
    U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
    V = Vh.T  # Principal components

    # Select first ndim components
    W = V[:, :ndim]

    # Transform the datas
    T = X_centered @ W
    return T
