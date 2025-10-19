#!/usr/bin/env python3
"""Script that adds two matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """Script that adds two matrices element-wise."""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    sum_matrix = [[0] * len(mat1[0]) for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            sum_matrix[i][j] = mat1[i][j] + mat2[i][j]
    return sum_matrix
