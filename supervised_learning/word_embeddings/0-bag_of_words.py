#!/usr/bin/env python3
import numpy as np
import re
from collections import Counter


def bag_of_words(sentences, vocab=None):
    """
    Creates a Bag-of-Words embedding matrix.
    """

    # Simple tokenizer (lowercase + keep words only)
    def tokenize(text):
        text = re.sub(r"'s\b", "", text.lower())
        return re.findall(r"[a-z]+", text)

    # Tokenize all sentences
    tokenized_sentences = [tokenize(s) for s in sentences]

    # Build vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(word for sent in tokenized_sentences for word in sent))

    # Map word → index
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Initialize embedding matrix
    s = len(sentences)
    f = len(vocab)
    embeddings = np.zeros((s, f), dtype=int)

    # Fill bag-of-words counts
    for i, sent in enumerate(tokenized_sentences):
        counts = Counter(sent)
        for word, count in counts.items():
            if word in word_to_index:
                embeddings[i, word_to_index[word]] = count

    return embeddings, np.array(vocab)
