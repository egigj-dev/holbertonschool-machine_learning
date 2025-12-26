#!/usr/bin/env python3
"""Gaussian Mixture Model clustering module using scikit-learn"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset.
    """
    n, d = X.shape
    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None, None, None
    # Create and fit GMM model
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_model.fit(X)
    # Extract results
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)
    return pi, m, S, clss, bic
