#!/usr/bin/env python3
"""Script to slice a numpy.ndarray along specific axes."""


def np_slice(matrix, axes={}):
    """Slices a numpy.ndarray along specific axes."""
    slices = []
    for i in range(matrix.ndim):
        if i in axes:
            slices.append(slice(*axes[i]))
        else:
            slices.append(slice(None))
    return matrix[tuple(slices)]
