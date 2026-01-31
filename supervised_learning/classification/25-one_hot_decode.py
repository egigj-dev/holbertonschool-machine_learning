#!/usr/bin/env python3
"""One-hot decoding module."""
import numpy as np


def one_hot_decode(one_hot):
    """
    Convert a one-hot matrix into a vector of labels.
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    if one_hot.shape[0] == 0 or one_hot.shape[1] == 0:
        return None

    labels = np.argmax(one_hot, axis=0)
    return labels
