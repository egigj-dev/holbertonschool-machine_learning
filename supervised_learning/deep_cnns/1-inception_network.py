#!/usr/bin/env python3
"""Inception Network as described in 'Going Deeper with Convolutions' (2014)"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the Inception network (GoogLeNet) with explicit ReLU layers.

    Input shape: (224, 224, 3)
    All convolutions use ReLU activation via K.layers.ReLU().

    Returns:
        keras.Model: the Inception (GoogLeNet) Keras model
    """
    X = K.Input(shape=(224, 224, 3))

    # Stage 1: Conv 7x7/2
    C1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(X)
    # Stage 2: MaxPool 3x3/2
    P1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(C1)
    # Stage 2: Conv 1x1/1
    C2 = K.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(P1)
    # Stage 2: Conv 3x3/1
    C3 = K.layers.Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu')(C2)
    # Stage 2: MaxPool 3x3/2
    P2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(C3)

    I3a = inception_block(P2, [64, 96, 128, 16, 32, 32])
    I3b = inception_block(I3a, [128, 128, 192, 32, 96, 64])
    P3 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(I3b)

    I4a = inception_block(P3, [192, 96, 208, 16, 48, 64])
    I4b = inception_block(I4a, [160, 112, 224, 24, 64, 64])
    I4c = inception_block(I4b, [128, 128, 256, 24, 64, 64])
    I4d = inception_block(I4c, [112, 144, 288, 32, 64, 64])
    I4e = inception_block(I4d, [256, 160, 320, 32, 128, 128])
    P4 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(I4e)

    I5a = inception_block(P4, [256, 160, 320, 32, 128, 128])
    I5b = inception_block(I5a, [384, 192, 384, 48, 128, 128])

    AP = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(I5b)
    D = K.layers.Dropout(0.4)(AP)
    F = K.layers.Flatten()(D)
    output = K.layers.Dense(1000, activation='softmax')(F)

    return K.Model(inputs=X, outputs=output)
