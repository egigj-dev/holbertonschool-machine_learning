#!/usr/bin/env python3
"""K-means clustering module"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    
    n, d = X.shape
    if k > n:
        return None, None
    
    # Initialize centroids using multivariate uniform distribution
    # This is the FIRST use of np.random.uniform
    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, d))
    
    # Main K-means loop
    for i in range(iterations):
        # Store old centroids to check for convergence
        C_old = C.copy()
        
        # Assignment step: assign each point to nearest centroid
        # Calculate distances from each point to each centroid
        # Shape: (n, k) - distance from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        
        # Assign each point to the closest centroid
        # clss shape: (n,) - cluster index for each point
        clss = np.argmin(distances, axis=1)
        
        # Update step: recalculate centroids
        for j in range(k):
            # Find all points assigned to cluster j
            cluster_points = X[clss == j]
            
            # If cluster is empty, reinitialize its centroid
            # This is the SECOND use of np.random.uniform
            if len(cluster_points) == 0:
                C[j] = np.random.uniform(low, high, size=(d,))
            else:
                # Update centroid to mean of assigned points
                C[j] = cluster_points.mean(axis=0)
        
        # Check for convergence (no change in centroids)
        if np.all(C == C_old):
            break
    
    # Final assignment with final centroids
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    
    return C, clss