#!/usr/bin/env python3
"""DenseNet-121 model with explicit ReLU layers"""
from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture

    Args:
        growth_rate: growth rate of the network
        compression: compression factor

    Returns:
        keras.Model
    """
    initializer = K.initializers.he_normal(seed=0)

    # Input
    X_input = K.Input(shape=(224, 224, 3))

    # Initial convolution
    X = K.layers.BatchNormalization(axis=-1)(X_input)
    X = K.layers.ReLU()(X)  # Explicit ReLU
    X = K.layers.Conv2D(
        filters=2 * growth_rate,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(X)

    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    nb_filters = 2 * growth_rate

    # Dense Block 1
    X, nb_filters = dense_block(X, nb_filters, growth_rate, layers=6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, layers=12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, layers=24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, layers=16)

    # Final BN + ReLU
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)  

    # Global average pooling
    X = K.layers.GlobalAveragePooling2D()(X)

    # Output layer (ImageNet)
    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(X)

    # Create model
    model = K.models.Model(inputs=X_input, outputs=X)
    return model