#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, regularizers
""" Creates a TensorFlow dense layer with L2 regularization """


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow dense layer with L2 regularization

    Parameters:
    prev: Output of the previous layer
    n: Number of neurons in the new layer
    activation: Activation function for the layer
    lambtha: L2 regularization parameter
    """

    layer = layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizers.L2(lambtha)
    )(prev)

    return layer
