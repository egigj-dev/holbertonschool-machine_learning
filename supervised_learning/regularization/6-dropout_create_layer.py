#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers
""" Script that creates a layer of a neural network using dropout """


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a dense layer with dropout regularization

    Parameters:
    prev: Output of the previous layer.
    n: Number of neurons in the new layer.
    activation: Activation function.
    keep_prob: Probability of keeping a node (0 < keep_prob <= 1)
    training: If True, dropout is applied. If False, dropout is skipped
    """
    # Create dense layer
    layer = layers.Dense(units=n, activation=activation)(prev)

    # Apply dropout if in training mode
    if training and keep_prob < 1.0:
        layer = layers.Dropout(rate=1 - keep_prob)(layer, training=training)

    return layer
