#!/usr/bin/env python3
"""Adds two matrices (n-dimensional lists) element-wise."""


def add_matrices(mat1, mat2):
    """Adds two matrices (n-dimensional lists) element-wise."""
    # If both are lists, recurse
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None
        return [add_matrices(m1, m2) for m1, m2 in zip(mat1, mat2)]
    else:
        return mat1 + mat2
