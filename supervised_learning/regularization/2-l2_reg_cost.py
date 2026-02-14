#!/usr/bin/env python3
"""Calculates the cost of a neural network with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    
    Parameters:
    -----------
    cost : tensor
        Cost of the network without L2 regularization
    model : Keras model
        Model that includes layers with L2 regularization
    
    Returns:
    --------
    tensor containing the cost for each layer of the network,
    accounting for L2 regularization
    """
    # Collect L2 regularization losses from each layer
    l2_losses = []
    
    for layer in model.layers:
        if hasattr(layer, 'losses') and layer.losses:
            # Add the regularization loss for this layer
            l2_losses.append(tf.reduce_sum(layer.losses))
    
    # Return as a tensor
    return tf.convert_to_tensor(l2_losses)
