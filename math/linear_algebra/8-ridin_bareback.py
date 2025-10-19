#!/usr/bin/env python3
"""Performs matrix multiplication between mat1 and mat2."""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication between mat1 and mat2."""
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat2[0])):
            sum_val = 0
            for k in range(len(mat1[0])):
                sum_val += mat1[i][k] * mat2[k][j]
            new_row.append(sum_val)
        result.append(new_row)
    return result
