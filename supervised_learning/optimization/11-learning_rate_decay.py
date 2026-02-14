#!/usr/bin/env python3
import numpy as np
""" Updates the learning rate using stepwise inverse time decay """


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using stepwise inverse time decay

    Parameters:
    alpha (float): Original learning rate
    decay_rate (float): Decay rate
    global_step (int): Number of gradient descent steps elapsed
    decay_step (int): Number of steps before applying further decay
    """

    # Compute how many decay intervals have passed
    step = np.floor(global_step / decay_step)

    # Apply stepwise inverse time decay
    alpha_updated = alpha / (1 + decay_rate * step)

    return alpha_updated
