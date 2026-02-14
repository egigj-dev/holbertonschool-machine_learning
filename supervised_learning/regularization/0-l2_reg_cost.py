#!/usr/bin/env python3
import numpy as np
""" Calculates the cost of a neural network using L2 """


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    """
    l2_sum = 0

    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        l2_sum += np.sum(np.square(W))

    # L2 regularization term
    l2_term = (lambtha / (2 * m)) * l2_sum

    # Total cost
    cost_l2 = cost + l2_term

    return cost_l2
