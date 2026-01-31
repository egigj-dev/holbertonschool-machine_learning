#!/usr/bin/env python3
"""A script that defines a single neuron."""
import numpy as np


class Neuron:
    """Defines a single neuron."""

    def __init__(self, nx):
        """Initialize the neuron with private attributes."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
