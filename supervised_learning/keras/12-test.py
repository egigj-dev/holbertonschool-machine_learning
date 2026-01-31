#!/usr/bin/env python3
"""Test neural network model."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ Test a neural network. """
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return [loss, accuracy]
