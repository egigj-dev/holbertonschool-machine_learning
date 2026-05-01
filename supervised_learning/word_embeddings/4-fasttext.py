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

    # sentences are already tokenized → DO NOT split or lowercase
    tokenized_sentences = sentences

    model = gensim.models.FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0 if cbow else 1,
        negative=negative,
        seed=seed,
        workers=workers
    )

    model.build_vocab(tokenized_sentences)

    model.train(
        tokenized_sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
