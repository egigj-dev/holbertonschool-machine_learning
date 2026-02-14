#!/usr/bin/env python3
"""Script that calculates F1 score"""
import numpy as np

# Import previously defined functions
sensitivity = __import__('1-sensitivity').sensitivity 
precision = __import__('2-precision').precision

def f1_score(confusion):
    """
    Function that calculates F1 score
    """
    recall_scores = sensitivity(confusion)
    precision_scores = precision(confusion)

    f1_scores = np.divide(
        2 * precision_scores * recall_scores,
        precision_scores + recall_scores,
        out=np.zeros_like(precision_scores, dtype=float),
        where=(precision_scores + recall_scores) != 0
    )

    return f1_scores
