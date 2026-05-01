#!/usr/bin/env python3
import numpy as np
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model into a Keras Embedding layer.

    Parameters:
    - model: trained gensim Word2Vec model

    Returns:
    - Keras Embedding layer with pretrained weights
    """

    # vocabulary size
    vocab_size = len(model.wv.index_to_key)

    # embedding dimension
    embedding_dim = model.vector_size

    # initialize embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # map words to indices in gensim vocab
    for i, word in enumerate(model.wv.index_to_key):
        embedding_matrix[i] = model.wv[word]

    # create Keras embedding layer (trainable=True by default)
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True
    )

    return embedding_layer
