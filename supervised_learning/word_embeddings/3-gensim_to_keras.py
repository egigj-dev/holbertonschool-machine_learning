#!/usr/bin/env python3
import tensorflow as tf


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
    #embedding_matrix = np.zeros((vocab_size, embedding_dim))
    embedding_matrix = [[0.0] * embedding_dim for _ in range(vocab_size)]  # list of lists to avoid numpy dependency

    # map words to indices in gensim vocab
    for i, word in enumerate(model.wv.index_to_key):
        embedding_matrix[i] = model.wv[word]

    # create Keras embedding layer (trainable=True by default)
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True
    )

    return embedding_layer
