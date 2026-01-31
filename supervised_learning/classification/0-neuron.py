#!/usr/bin/env python3
""" A script that defines a single neuron """
import numpy as np


class Neuron:
    """Defines a single neuron."""

    def __init__(self, nx):
        """Initialize the neuron."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
