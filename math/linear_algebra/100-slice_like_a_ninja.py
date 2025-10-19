#!/usr/bin/env python3
"""Script to slice a numpy.ndarray along specific axes."""
import numpy as np


def np_slice(matrix, axes={}):
    """Script to slice a numpy.ndarray along specific axes."""
    slices = []
    for i in range(matrix.ndim):
        if i in axes:
            s = axes[i]
            slices.append(slice(*s))
        else:
            slices.append(slice(None))
    return matrix[tuple(slices)].copy()
