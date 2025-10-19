#!/usr/bin/env python3
"""Script that adds two matrices element-wise."""
matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """Script that adds two matrices element-wise."""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    sum_matrix = [[0] * len(mat1[0]) for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            sum_matrix[i][j] = mat1[i][j] + mat2[i][j]
    return sum_matrix
