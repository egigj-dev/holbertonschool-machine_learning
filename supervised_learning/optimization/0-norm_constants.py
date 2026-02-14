#!/usr/bin/env python3
import numpy as np
"""Script that calculates the normalization constants of a matrix"""


def normalization_constants(X):
	"""
	Function that finds the normalization constants of a matrix
	"""

	mean = np.mean(X, axis=0, keepdims=True)
	std = np.std(X, axis=0, keepdims=True)

	return mean, std
