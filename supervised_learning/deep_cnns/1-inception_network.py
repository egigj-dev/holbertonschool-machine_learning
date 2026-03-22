#!/usr/bin/env python3
"""Inception Network as described in 'Going Deeper with Convolutions' (2014)"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the Inception network (GoogLeNet).

    Input shape: (224, 224, 3)
    All convolutions use ReLU activation.

    Returns:
        keras.Model: the Inception (GoogLeNet) Keras model
    """
    X = K.Input(shape=(224, 224, 3))

    # Stage 1: Conv 7x7/2
    C1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                         activation='relu')(X)

    # Stage 2: MaxPool 3x3/2
    P1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(C1)

    # Stage 2: Conv 1x1/1 (dimension reduction)
    C2 = K.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                         activation='relu')(P1)

    # Stage 2: Conv 3x3/1
    C3 = K.layers.Conv2D(192, (3, 3), strides=(1, 1), padding='same',
                         activation='relu')(C2)

    # Stage 2: MaxPool 3x3/2
    P2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(C3)

    # Inception 3a: filters = [64, 96, 128, 16, 32, 32]
    I3a = inception_block(P2, [64, 96, 128, 16, 32, 32])

    # Inception 3b: filters = [128, 128, 192, 32, 96, 64]
    I3b = inception_block(I3a, [128, 128, 192, 32, 96, 64])

    # MaxPool 3x3/2
    P3 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(I3b)

    # Inception 4a: filters = [192, 96, 208, 16, 48, 64]
    I4a = inception_block(P3, [192, 96, 208, 16, 48, 64])

    # Inception 4b: filters = [160, 112, 224, 24, 64, 64]
    I4b = inception_block(I4a, [160, 112, 224, 24, 64, 64])

    # Inception 4c: filters = [128, 128, 256, 24, 64, 64]
    I4c = inception_block(I4b, [128, 128, 256, 24, 64, 64])

    # Inception 4d: filters = [112, 144, 288, 32, 64, 64]
    I4d = inception_block(I4c, [112, 144, 288, 32, 64, 64])

    # Inception 4e: filters = [256, 160, 320, 32, 128, 128]
    I4e = inception_block(I4d, [256, 160, 320, 32, 128, 128])

    # MaxPool 3x3/2
    P4 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(I4e)

    # Inception 5a: filters = [256, 160, 320, 32, 128, 128]
    I5a = inception_block(P4, [256, 160, 320, 32, 128, 128])

    # Inception 5b: filters = [384, 192, 384, 48, 128, 128]
    I5b = inception_block(I5a, [384, 192, 384, 48, 128, 128])

    # AvgPool 7x7/1
    AP = K.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(I5b)

    # Dropout 40%
    D = K.layers.Dropout(0.4)(AP)

    # Flatten
    F = K.layers.Flatten()(D)

    # Fully connected (linear) → Softmax output for 1000 classes
    output = K.layers.Dense(1000, activation='softmax')(F)

    model = K.Model(inputs=X, outputs=output)

    return model
