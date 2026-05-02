#!/usr/bin/env python3
import math
from collections import Counter
"""Calculates unigram BLEU score"""


def uni_bleu(references, sentence):
    """Calculates unigram BLEU score"""

    sentence_len = len(sentence)

    # Count unigrams in sentence
    sent_counts = Counter(sentence)

    # Max reference counts
    max_ref_counts = {}
    for ref in references:
        ref_counts = Counter(ref)
        for word in sent_counts:
            max_ref_counts[word] = max(
                max_ref_counts.get(word, 0),
                ref_counts.get(word, 0)
            )

    # Clipped count
    clipped = sum(
        min(count, max_ref_counts.get(word, 0))
        for word, count in sent_counts.items()
    )

    precision = clipped / sentence_len if sentence_len > 0 else 0

    # Brevity penalty
    ref_lens = [len(r) for r in references]
    closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - sentence_len), rl))

    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / sentence_len)

    return bp * precision
