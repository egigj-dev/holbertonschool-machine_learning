#!/usr/bin/env python3
"""Converts a gensim Word2Vec model to a Keras Embedding layer."""
import tensorflow as tf


def gensim_to_keras(model):
    """Converts a trained gensim Word2Vec to a Keras Embedding."""
    weights = model.wv.vectors
    return tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
