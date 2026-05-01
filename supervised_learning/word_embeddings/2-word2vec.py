#!/usr/bin/env python3
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

    # simple tokenization
    tokenized_sentences = [
        sentence.lower().split() for sentence in sentences
    ]

    model = gensim.models.Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0 if cbow else 1,   # 0 = CBOW, 1 = Skip-gram
        negative=negative,
        seed=seed,
        workers=workers
    )

    # train explicitly (gensim modern versions require this)
    model.train(
        tokenized_sentences,
        total_examples=len(tokenized_sentences),
        epochs=epochs
    )

    return model
