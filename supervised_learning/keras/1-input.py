#!/usr/bin/env python3
"""Build neural network with Keras Functional API."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build a neural network with Functional API.
    """
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i, num_nodes in enumerate(layers):
        x = K.layers.Dense(
            num_nodes,
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    model = K.Model(inputs=inputs, outputs=x)
    return model
