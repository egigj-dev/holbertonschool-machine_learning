#!/usr/bin/env python3
"""Make predictions with neural network."""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ Make predictions using a neural network. """
    predictions = network.predict(data, verbose=verbose)
    return predictions
