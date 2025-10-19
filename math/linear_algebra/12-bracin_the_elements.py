#!/usr/bin/env python3
"""Script that returns the result elementwise addition, subtraction, multiplication, and division of two numpy matrices."""


def np_elementwise(mat1, mat2):
    """Script that returns the result elementwise addition, subtraction, multiplication, and division of two numpy matrices."""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
