#!/usr/bin/env python3
import numpy as np
import re
import math
from collections import Counter, defaultdict


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Parameters:
    - sentences: list of strings
    - vocab: list of vocabulary words (optional)

    Returns:
    - embeddings: numpy.ndarray of shape (s, f)
    - features: list of vocabulary words used
    """

    # --- tokenizer ---
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    tokenized_sentences = [tokenize(s) for s in sentences]

    s = len(sentences)

    # --- build vocab if not provided ---
    if vocab is None:
        vocab = sorted(set(word for sent in tokenized_sentences for word in sent))

    f = len(vocab)

    word_to_index = {word: i for i, word in enumerate(vocab)}

    # --- compute document frequency (DF) ---
    df = defaultdict(int)
    for sent in tokenized_sentences:
        unique_words = set(sent)
        for word in unique_words:
            if word in word_to_index:
                df[word] += 1

    # --- compute IDF ---
    idf = {}
    for word in vocab:
        # smoothing to avoid division by zero
        idf[word] = math.log((s + 1) / (df[word] + 1)) + 1

    # initialize FIRST
    embeddings = np.zeros((s, f), dtype=float)

    # fill SECOND
    for i, sent in enumerate(tokenized_sentences):
        tf = Counter(sent)
        total_terms = len(sent)
        for word, count in tf.items():
            if word in word_to_index:
                tf_val = count / total_terms
                embeddings[i, word_to_index[word]] = tf_val * idf[word]

    # normalize THIRD
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, np.array(vocab)
