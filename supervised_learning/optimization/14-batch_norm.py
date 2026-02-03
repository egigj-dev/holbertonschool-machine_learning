#!/usr/bin/env python3
import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function to apply

    Returns:
        Tensor of the activated output for the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Dense layer without activation
    dense = tf.keras.layers.Dense(
        n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
        use_bias=False
    )(prev)

    # Trainable batch norm parameters
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    # Batch statistics
    mean, variance = tf.nn.moments(dense, axes=[0])

    # Batch normalization
    batch_norm = tf.nn.batch_normalization(
        dense,
        mean,
        variance,
        beta,
        gamma,
        variance_epsilon=1e-7
    )

    # Activation
    return activation(batch_norm)
