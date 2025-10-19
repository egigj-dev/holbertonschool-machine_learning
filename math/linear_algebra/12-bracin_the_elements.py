#!/usr/bin/env python3
"""Script that performs elementwise +, -, *, and / on two numpy matrices."""


def np_elementwise(mat1, mat2):
    """Script that performs elementwise +, -, *, / on mat1, mat2."""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
