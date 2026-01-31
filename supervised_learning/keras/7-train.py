#!/usr/bin/env python3
"""Train with learning rate decay."""
import tensorflow.keras as keras


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Train a model with learning rate decay.
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
        )
    if learning_rate_decay and validation_data is not None:
        def decay_lr(epoch):
            return alpha / (1 + decay_rate * epoch)
        callbacks.append(
            keras.callbacks.LearningRateScheduler(decay_lr, verbose=1)
        )
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
