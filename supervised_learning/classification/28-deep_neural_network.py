#!/usr/bin/env python3
"""Deep Neural Network module for multiclass classification."""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification."""
    
    def __init__(self, nx, layers, activation='sig'):
        """Initialize the deep neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        
        layer_sizes = [nx] + layers
        for l in range(1, self.__L + 1):
            if not isinstance(layers[l - 1], int) or layers[l - 1] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.__weights[f'W{l}'] = (np.random.randn(
                layers[l - 1], layer_sizes[l - 1]) *
                np.sqrt(2 / layer_sizes[l - 1]))
            self.__weights[f'b{l}'] = np.zeros((layers[l - 1], 1))
    
    @property
    def L(self):
        """Getter for L."""
        return self.__L
    
    @property
    def cache(self):
        """Getter for cache."""
        return self.__cache
    
    @property
    def weights(self):
        """Getter for weights."""
        return self.__weights
    
    @property
    def activation(self):
        """Getter for activation."""
        return self.__activation
    
    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache['A0'] = X
        
        for l in range(1, self.__L + 1):
            Z = (np.matmul(self.__weights[f'W{l}'],
                 self.__cache[f'A{l-1}']) + self.__weights[f'b{l}'])
            
            if l == self.__L:
                # Softmax activation for output layer
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache[f'A{l}'] = exp_Z / np.sum(
                    exp_Z, axis=0, keepdims=True)
            else:
                # Hidden layer activation
                if self.__activation == 'sig':
                    self.__cache[f'A{l}'] = 1 / (1 + np.exp(-Z))
                else:  # tanh
                    self.__cache[f'A{l}'] = np.tanh(Z)
        
        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using cross-entropy."""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        A, _ = self.forward_prop(X)
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network."""
        m = Y.shape[1]
        dZ = cache[f'A{self.__L}'] - Y
        
        for l in range(self.__L, 0, -1):
            dW = np.matmul(dZ, cache[f'A{l-1}'].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                if self.__activation == 'sig':
                    dA = (cache[f'A{l-1}'] * (1 - cache[f'A{l-1}']))
                else:  # tanh
                    dA = 1 - np.square(cache[f'A{l-1}'])
                
                dZ = np.matmul(self.__weights[f'W{l}'].T, dZ) * dA
            
            self.__weights[f'W{l}'] -= alpha * dW
            self.__weights[f'b{l}'] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        
        costs = []
        steps_list = []
        
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            
            if verbose or graph:
                cost = self.cost(Y, A)
                
                if (i % step == 0) or (i == iterations):
                    if verbose:
                        print(f"Cost after {i} iterations: {cost}")
                    if graph:
                        costs.append(cost)
                        steps_list.append(i)
            
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
        
        if graph:
            plt.plot(steps_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        
        return self.evaluate(X, Y)

    def save(self, filename):
        """Save the instance object to a file in pickle format."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object."""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
