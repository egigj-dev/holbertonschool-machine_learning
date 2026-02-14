#!/usr/bin/env python3
"""Script that calculates the precision for each class"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    """
    # True positives for each class (diagonal)
    true_positives = np.diag(confusion)

    # Predicted positives = TP + FP (sum of column)
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP)
    precision_scores = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(predicted_positives != 0)
    )

    return precision_scores
