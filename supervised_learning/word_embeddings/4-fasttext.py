#!/usr/bin/env python3
import gensim


def fasttext_model(
    sentences,
    vector_size=100,
    min_count=5,
    negative=5,
    window=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """
    Creates, builds, and trains a FastText model using gensim.
    """

    # tokenization (gensim expects list of token lists)
    tokenized_sentences = [
        sentence.lower().split() for sentence in sentences
    ]

    # build FastText model
    model = gensim.models.FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0 if cbow else 1,
        negative=negative,
        seed=seed,
        workers=workers
    )

    # build vocabulary
    model.build_vocab(tokenized_sentences)

    # train model
    model.train(
        tokenized_sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
