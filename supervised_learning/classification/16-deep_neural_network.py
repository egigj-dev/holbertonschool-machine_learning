#!/usr/bin/env python3
"""Deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        # --- Input validation ---
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        # --- Public attributes ---
        self.L = len(layers)           
        self.cache = {}                
        self.weights = {}              

        # --- Initialize weights and biases using He initialization ---
        for l in range(self.L):
            nodes = layers[l]
            if l == 0:
                self.weights["W1"] = np.random.randn(nodes, nx) * np.sqrt(2 / nx)
            else:
                prev_nodes = layers[l - 1]
                self.weights[f"W{l + 1}"] = np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            self.weights[f"b{l + 1}"] = np.zeros((nodes, 1))
