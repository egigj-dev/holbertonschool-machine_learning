#!/usr/bin/env python3
"""Projection block module"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block

    Args:
        A_prev: output from previous layer
        filters: tuple/list containing (F11, F3, F12)
        s: stride for the first convolution

    Returns:
        activated output of the projection block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    # Save shortcut
    shortcut = A_prev

    # First component (with stride s)
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=-1)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Second component
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=-1)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # Third component
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # Shortcut path (projection)
    shortcut_conv = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(shortcut)
    shortcut_bn = K.layers.BatchNormalization(axis=-1)(shortcut_conv)

    # Add main path and shortcut
    add = K.layers.Add()([bn3, shortcut_bn])
    output = K.layers.Activation('relu')(add)

    return output
