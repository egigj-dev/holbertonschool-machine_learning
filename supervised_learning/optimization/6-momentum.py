#!/usr/bin/env python3
import tensorflow as tf
""" Script that creates a TF optimizer """


def create_momentum_op(alpha, beta1):
    """
    Creates a TensorFlow optimizer for gradient descent with momentum.
    """

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer
