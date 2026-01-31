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
        """Get the weights vector."""
        return self.__W

    @property
    def b(self):
        """Get the bias."""
        return self.__b

    @property
    def A(self):
        """Get the activated output."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of loss function.
        """
        m = Y.shape[1]
        log_loss = -1 / m * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return log_loss

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.
        """
        self.__A = self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates gradient descent.
        """
        m = Y.shape[1]
        dW = np.matmul(X, (A - Y).T) / m
        db = np.sum(A - Y) / m
        self.__W -= alpha * dW.T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)

                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

                if graph:
                    costs.append(cost)
                    steps.append(i)

            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
