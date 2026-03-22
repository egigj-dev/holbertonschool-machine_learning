#!/usr/bin/env python3
"""Identity block module"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block with ReLU layers (not generic Activation)."""
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    X_shortcut = A_prev

    # First component
    conv1 = K.layers.Conv2D(F11, (1, 1), strides=(1, 1), padding='valid',
                            kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.ReLU()(bn1)  # <-- ReLU layer

    # Second component
    conv2 = K.layers.Conv2D(F3, (3, 3), strides=(1, 1), padding='same',
                            kernel_initializer=init)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.ReLU()(bn2)  # <-- ReLU layer

    # Third component
    conv3 = K.layers.Conv2D(F12, (1, 1), strides=(1, 1), padding='valid',
                            kernel_initializer=init)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Add shortcut
    add = K.layers.Add()([bn3, X_shortcut])
    output = K.layers.ReLU()(add)  # <-- ReLU layer after addition

    return output
