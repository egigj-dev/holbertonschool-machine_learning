#!/usr/bin/env python3
"""Creates a Bag-of-Words embedding matrix."""
import re
from collections import Counter
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Creates a Bag-of-Words embedding matrix."""
    def tokenize(text):
        """Tokenizes text to lowercase words, removing possessives."""
        text = re.sub(r"'s\b", "", text.lower())
        return re.findall(r"[a-z]+", text)

    tokenized_sentences = [tokenize(s) for s in sentences]

    if vocab is None:
        vocab = sorted(
            set(word for sent in tokenized_sentences for word in sent)
        )

    word_to_index = {word: i for i, word in enumerate(vocab)}

    s = len(sentences)
    f = len(vocab)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sent in enumerate(tokenized_sentences):
        counts = Counter(sent)
        for word, count in counts.items():
            if word in word_to_index:
                embeddings[i, word_to_index[word]] = count

    return embeddings, np.array(vocab)
