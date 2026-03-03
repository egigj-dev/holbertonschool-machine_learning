#!/usr/bin/env python3
"""Back propagation over a convolutional layer"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture using Keras
    """
    initializer = K.initializers.HeNormal(seed=0)

    # First convolutional layer
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # First max pooling
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Second convolutional layer
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # Second max pooling
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten for fully connected layers
    flat = K.layers.Flatten()(pool2)

    # Fully connected layers
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flat)

    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # Output layer with softmax
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    # Build and compile the model
    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
