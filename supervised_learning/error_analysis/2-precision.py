#!/usr/bin/env python3
import numpy as np
"""Script that calculates the precision"""


def precision(confusion):
    """Precision = TP/(TP+FP)"""
    
    true_positives = np.diag(confusion)                 # TP
    predicted_positives = np.sum(confusion, axis=0)    # TP+FP

    # TP/(TP+FP)
    precision_scores = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(predicted_positives != 0)
    )

    return precision_scores
