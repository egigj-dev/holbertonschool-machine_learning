#!/usr/bin/env python3
"""Save and load model config."""
import tensorflow.keras as keras


def save_config(network, filename):
    """ Save model configuration in JSON. """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """ Load model from JSON config. """
    with open(filename, 'r') as f:
        config = f.read()
    model = keras.models.model_from_json(config)
    return model
