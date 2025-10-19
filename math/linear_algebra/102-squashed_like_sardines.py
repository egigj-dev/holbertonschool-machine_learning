#!/usr/bin/env python3
"""Script that concatenates two matrices along a specific axis."""
matrix_shape = __import__('2-size_me_please').matrix_shape


def cat_matrices(mat1, mat2, axis=0):
    """Script that concatenates two matrices along a specific axis."""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None
    if axis == 0:
        return mat1 + mat2
    else:
        result = []
        for sub1, sub2 in zip(mat1, mat2):
            merged = cat_matrices(sub1, sub2, axis - 1)
            result.append(merged)
        return result
