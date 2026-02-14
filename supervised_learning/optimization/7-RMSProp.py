#!/usr/bin/env python3
import numpy as np
""" Script that updates a variable using RMSProp """


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.
    """

    # Update second moment (running average of squared gradients)
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update variable
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
