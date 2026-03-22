#!/usr/bin/env python3
"""Identity block module"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block

    Args:
        A_prev: output from previous layer
        filters: tuple/list containing (F11, F3, F12)

    Returns:
        activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = A_prev

    # First component of main path
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                        kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Second component of main path
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer=init)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # Third component of main path
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                        kernel_initializer=init)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)
    act3 = K.layers.Activation('relu')(bn3)

    # Add shortcut value to main path
    add = K.layers.Add()([act3, X_shortcut])
    output = K.layers.Activation('relu')(add)
    return output
