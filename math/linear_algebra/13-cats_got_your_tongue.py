#!/usr/bin/env python3
"""Script that concatenates two numpy matrices along a specific axis."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Script that concatenates two numpy matrices along a specific axis."""
    return np.concatenate((mat1, mat2), axis=axis)
