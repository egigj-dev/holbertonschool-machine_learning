#!/usr/bin/env python3
"""Convert labels to one-hot matrix."""
import numpy as np


def one_hot(labels, classes=None):
    """Convert label vector to one-hot matrix.
    """
    if classes is None:
        classes = np.max(labels) + 1
    m = labels.shape[0]
    one_hot_matrix = np.zeros((m, classes))
    one_hot_matrix[np.arange(m), labels] = 1
    return one_hot_matrix
