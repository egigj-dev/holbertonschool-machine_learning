#!/usr/bin/env python3
import tensorflow as tf
""" Script that creates a TF RMSProp optimizer """


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates a TensorFlow RMSProp optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )

    return optimizer
