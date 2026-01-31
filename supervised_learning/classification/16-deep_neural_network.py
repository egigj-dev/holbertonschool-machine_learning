#!/usr/bin/env python3
"""Deep Neural Network module."""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""
    def __init__(self, nx, layers):
        """Initialize the deep neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        layer_sizes = [nx] + layers
        for l in range(1, self.L + 1):
            if not isinstance(layers[l - 1], int) or layers[l - 1] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights[f'W{l}'] = np.random.randn(layers[l - 1], layer_sizes[l - 1]) * np.sqrt(2 / layer_sizes[l - 1])
            self.weights[f'b{l}'] = np.zeros((layers[l - 1], 1))
