#!/usr/bin/env python3
"""K-means clustering module"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering"""
    # Validation 
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialization
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))  # FIRST use

    # Main loop
    for _ in range(iterations):  # LOOP 1
        C_prev = C.copy()

        # distances: (n, k)
        distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):  # LOOP 2
            points = X[clss == j]

            if points.shape[0] == 0:
                # Reinitialize empty cluster
                C[j] = np.random.uniform(low, high, size=d)  # SECOND use
            else:
                C[j] = np.mean(points, axis=0)

        # Convergence check
        if np.allclose(C, C_prev):
            break

    return C, clss
