#!/usr/bin/env python3
import numpy as np
""" Script that calculates the weighted moving average """

shuffle_data = __import__('2-shuffle_data').shuffle_data


def moving_average(data, beta):
    """
    Calculates the bias-corrected weighted moving average
    """
    moving_averages = []
    v = 0
    t = 0

    for value in data:
        t += 1

        # Exponentially weighted moving average
        v = beta * v + (1 - beta) * value

        # Bias correction
        v_corrected = v / (1 - beta ** t)

        moving_averages.append(v_corrected)

    return moving_averages
