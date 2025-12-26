#!/usr/bin/env python3
"""PCA Module"""
import numpy as np

def pca(X, var=0.95):
    """
    Performs Principal Component Analysis (PCA) on a dataset.
    """
    # Compute SVD of X
    U, S, Vh = np.linalg.svd(X)
    V = Vh.T  # Columns are principal components

    # Compute cumulative variance fraction
    cum_var = np.cumsum(S) / np.sum(S)

    # Determine number of components to keep
    num_components = np.where(cum_var >= var)[0][0] + 1

    # Return the first 'num_components' principal components
    W = V[:, :num_components]
    return W
