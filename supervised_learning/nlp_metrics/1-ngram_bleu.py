#!/usr/bin/env python3
import math
from collections import Counter
"""Calculates n-gram BLEU score"""


def get_ngrams(tokens, n):
    """Returns tokens n-grams as tuples"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def ngram_bleu(references, sentence, n):
    """Calculates n-gram BLEU score"""

    sentence_len = len(sentence)

    sent_ngrams = get_ngrams(sentence, n)
    sent_counts = Counter(sent_ngrams)

    max_ref_counts = {}
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        ref_counts = Counter(ref_ngrams)
        for ng in sent_counts:
            max_ref_counts[ng] = max(
                max_ref_counts.get(ng, 0),
                ref_counts.get(ng, 0)
            )

    clipped = sum(
        min(count, max_ref_counts.get(ng, 0))
        for ng, count in sent_counts.items()
    )

    total = len(sent_ngrams)
    precision = clipped / total if total > 0 else 0

    # Brevity penalty
    ref_lens = [len(r) for r in references]
    clsest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - sentence_len), rl))

    if sentence_len > clsest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - clsest_ref_len / sentence_len)

    return bp * precision
