#!/usr/bin/env python3
"""Save and load Keras models."""
import tensorflow.keras as keras


def save_model(network, filename):
    """ Save an entire model. """
    network.save(filename)


def load_model(filename):
    """ Load an entire model. """
    return keras.models.load_model(filename)
