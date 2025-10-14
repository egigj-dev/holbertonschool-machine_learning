#!/usr/bin/env python3
"""Script that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Script that calculates the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
