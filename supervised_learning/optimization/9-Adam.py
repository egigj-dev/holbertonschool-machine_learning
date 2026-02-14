#!/usr/bin/env python3
import numpy as np
""" Script that updates a variable using Adam """


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm

    Parameters:
    alpha: Learning rate
    beta1: Weight for first moment
    beta2: Weight for second moment
    epsilon: Small constant to avoid division by zero
    var: Variable to update
    grad: Gradient of the variable
    v: Previous first moment
    s: Previous second moment
    t: Time step (for bias correction)
    """

    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # Update biased second moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Bias correction
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    # Update variable
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
