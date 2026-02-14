#!/usr/bin/env python3
import tensorflow as tf
""" Script that creates a TF Adam optimizer """


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates a TensorFlow Adam optimizer.

    Parameters:
    alpha: Learning rate
    beta1: Exponential decay rate for the first moment
    beta2: Exponential decay rate for the second moment
    epsilon: Small constant to avoid division by zero
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )

    return optimizer
