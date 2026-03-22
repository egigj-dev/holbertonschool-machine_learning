#!/usr/bin/env python3
"""Transition layer module"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer

    Args:
        X: output from previous layer
        nb_filters: number of filters in X
        compression: compression factor

    Returns:
        output of the transition layer and updated number of filters
    """
    initializer = K.initializers.he_normal(seed=0)

    # Compute compressed number of filters
    filters = int(nb_filters * compression)

    # BatchNorm -> ReLU
    bn = K.layers.BatchNormalization(axis=-1)(X)
    act = K.layers.Activation('relu')(bn)

    # 1x1 convolution (channel reduction)
    conv = K.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act)

    # Average pooling (spatial reduction)
    output = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv)

    return output, filters
