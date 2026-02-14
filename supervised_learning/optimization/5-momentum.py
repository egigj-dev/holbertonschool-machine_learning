#!/usr/bin/env python3
import numpy as np
""" Script that uses gradient descent to update a var """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum
    """

    # Update velocity (first moment estimate)
    v = beta1 * v + (1 - beta1) * grad

    # Update variable
    var = var - alpha * v

    return var, v
