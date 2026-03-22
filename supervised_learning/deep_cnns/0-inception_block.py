#!/usr/bin/env python3
"""Inception block module"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block

    Args:
        A_prev: output tensor from previous layer
        filters: tuple/list containing (F1, F3R, F3, F5R, F5, FPP)

    Returns:
        concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch
    conv1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    # 1x1 -> 3x3 convolution branch
    conv3_reduce = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(conv3_reduce)

    # 1x1 -> 5x5 convolution branch
    conv5_reduce = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(conv5_reduce)

    # maxpool -> 1x1 convolution branch
    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    pool_proj = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(pool)

    # concatenate filters
    output = K.layers.concatenate(
        [conv1, conv3, conv5, pool_proj],
        axis=-1
    )

    return output
