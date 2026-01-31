#!/usr/bin/env python3
"""Convert labels to one-hot matrix."""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Convert label vector to one-hot matrix. """
    return K.utils.to_categorical(labels, num_classes=classes)
