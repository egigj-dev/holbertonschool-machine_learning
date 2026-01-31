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

    def forward_prop(self, X):
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    
    def cost(self, Y, A):
        m = Y.shape[1]
        log_loss = -1 / m * np.sum(Y * np.log(A) + (1 - Y)*(np.log(1.000001 - A)))
        return log_loss
