#!/usr/bin/env python3
"""Gaussian Mixture Model maximization step module"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n2 = g.shape
    if n2 != n or k <= 0:
        return None, None, None

    # g must be non-negative
    if np.any(g < 0):
        return None, None, None

    # Columns of g must sum to 1
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    # Responsibilities per cluster
    Nk = np.sum(g, axis=1)
    if np.any(Nk == 0):
        return None, None, None

    # Priors
    pi = Nk / n

    # Means
    m = (g @ X) / Nk[:, None]

    # Covariances (ONLY ONE LOOP)
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted = g[i][:, None] * diff
        S[i] = (weighted.T @ diff) / Nk[i]

    return pi, m, S
