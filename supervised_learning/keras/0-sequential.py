#!/usr/bin/env python3
"""Build neural network with Keras Sequential API."""
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build a neural network with Sequential API.
    """
    model = keras.Sequential()
    for i, num_nodes in enumerate(layers):
        if i == 0:
            model.add(keras.layers.Dense(
                num_nodes,
                activation=activations[i],
                input_shape=(nx,),
                kernel_regularizer=keras.regularizers.l2(lambtha)
            ))
        else:
            model.add(keras.layers.Dense(
                num_nodes,
                activation=activations[i],
                kernel_regularizer=keras.regularizers.l2(lambtha)
            ))
        if i < len(layers) - 1:
            model.add(keras.layers.Dropout(1 - keep_prob))
    return model
