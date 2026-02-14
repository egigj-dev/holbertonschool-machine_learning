#!/usr/bin/env python3
import tensorflow as tf
""" Calculates the total cost of a Keras model """


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a Keras model including L2 regularization

    Parameters:
    cost (tf.Tensor): Cost without L2 regularization
    model (tf.keras.Model): Keras model with layers that include L2 regularization
    """
    # Initialize L2 regularization term
    l2_term = 0

    # Loop through each layer
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            # Add regularization penalty for this layer
            l2_term += layer.kernel_regularizer(layer.kernel)

    # Total cost
    total_cost = cost + l2_term

    return total_cost
