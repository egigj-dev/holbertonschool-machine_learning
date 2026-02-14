#!/usr/bin/env python3
import tensorflow as tf
""" Creates a TF inverse time decay learning rate schedule """


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a TensorFlow inverse time decay learning rate schedule
    (stepwise)

    Parameters:
    alpha (float): Initial learning rate
    decay_rate (float): Decay rate
    decay_step (int): Number of steps before applying further decay
    """

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    return lr_schedule
