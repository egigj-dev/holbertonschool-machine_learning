#!/usr/bin/env python3
"""Script that returns the transpose of a 2D matrix."""


def matrix_transpose(matrix):
    """Script that returns the transpose of a 2D matrix."""
    transposed_matrix = []
    for i in range(len(matrix[0])):
        new_row = []
        for row in matrix:
            new_row.append(row[i])
        transposed_matrix.append(new_row)
    return transposed_matrix
