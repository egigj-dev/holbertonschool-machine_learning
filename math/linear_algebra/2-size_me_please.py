#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Module that defines a function to compute the shape of a matrix.

    Functions:
        matrix_shape(matrix): Returns the dimensions of a matrix as a list.

    Parameters: matrix that needs shape defined
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
