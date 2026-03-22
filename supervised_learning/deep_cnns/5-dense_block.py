#!/usr/bin/env python3
"""Dense block module"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block

    Args:
        X: output from previous layer
        nb_filters: number of filters in X
        growth_rate: growth rate of the dense block
        layers: number of layers in the block

    Returns:
        concatenated output and updated number of filters
    """
    initializer = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        # Bottleneck layer
        bn1 = K.layers.BatchNormalization(axis=-1)(X)
        act1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(act1)

        # 3x3 convolution
        bn2 = K.layers.BatchNormalization(axis=-1)(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=initializer
        )(act2)

        # Concatenate with input
        X = K.layers.concatenate([X, conv2], axis=-1)

        # Update number of filters
        nb_filters += growth_rate

    return X, nb_filters
