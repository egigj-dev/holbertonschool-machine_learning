#!/usr/bin/env python3
"""Script that returns the transpose of a 2D matrix"""

def matrix_transpose(matrix):
    """
    matrix = [[1, 2, 3]
              [4, 5, 6]]
    
    transposed_matrix = [[1, 4]
                         [2, 5]
                         [3, 6]]
    """
    transposed_matrix = []
    for i in range(len(matrix[0])):
        new_row = []
        for row in matrix:
            new_row.append(row[i])
        transposed_matrix.append(new_row)
    
    return transposed_matrix
    #return [[row[i] for row in matrix] for i in range(len(matrix[0]))]