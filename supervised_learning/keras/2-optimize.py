#!/usr/bin/env python3
"""Optimize Keras model with Adam optimizer."""
import tensorflow.keras as keras


def optimize_model(network, alpha, beta1, beta2):
    """
    Set up Adam optimization for a Keras model.
    """
    optimizer = keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
