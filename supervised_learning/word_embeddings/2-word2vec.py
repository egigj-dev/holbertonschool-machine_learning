#!/usr/bin/env python3
from pyexpat import model

import gensim


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """
    Creates, builds, and trains a Word2Vec model using gensim.

    Parameters:
    - sentences: list of sentences (strings)
    - vector_size: embedding dimension
    - min_count: minimum word frequency
    - window: context window size
    - negative: negative sampling size
    - cbow: True = CBOW, False = Skip-gram
    - epochs: training iterations
    - seed: random seed
    - workers: number of threads

    Returns:
    - trained Word2Vec model
    """

    model = gensim.models.Word2Vec(
        sentences=sentences,      # pass as-is, no re-tokenization
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0 if cbow else 1,
        negative=negative,
        seed=seed,
        workers=workers,
        epochs=epochs
    )
    return model
