#!/usr/bin/env python3
"""K-means clustering module using scikit-learn"""
import sklearn.cluster
import numpy as np


def kmeans(X, k):
    """
    Performs K-means on a dataset.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    
    n, d = X.shape
    
    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None
    
    # Create and fit K-means model
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_model.fit(X)
    
    # Extract results
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_
    
    return C, clss
