#!/usr/bin/env python3
"""Creates a TF-IDF embedding matrix."""
import re
import math
from collections import Counter, defaultdict
import numpy as np


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding matrix."""
    def tokenize(text):
        """Tokenizes text to lowercase words, removing possessives."""
        text = re.sub(r"'s\b", "", text.lower())
        return re.findall(r"[a-z]+", text)

    tokenized_sentences = [tokenize(s) for s in sentences]
    s = len(sentences)

    if vocab is None:
        vocab = sorted(
            set(word for sent in tokenized_sentences for word in sent)
        )

    f = len(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}

    df = defaultdict(int)
    for sent in tokenized_sentences:
        for word in set(sent):
            if word in word_to_index:
                df[word] += 1

    idf = {
        word: math.log((s + 1) / (df[word] + 1)) + 1
        for word in vocab
    }

    embeddings = np.zeros((s, f), dtype=float)

    for i, sent in enumerate(tokenized_sentences):
        tf = Counter(sent)
        total_terms = len(sent)
        for word, count in tf.items():
            if word in word_to_index:
                tf_val = count / total_terms
                embeddings[i, word_to_index[word]] = tf_val * idf[word]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, np.array(vocab)
