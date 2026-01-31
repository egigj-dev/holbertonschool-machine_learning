#!/usr/bin/env python3
"""Save and load Keras models."""
import tensorflow.keras as K


def save_model(network, filename):
    """ Save an entire model. """
    network.save(filename)


def load_model(filename):
    """ Load an entire model. """
    return K.models.load_model(filename)
