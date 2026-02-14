#!/usr/bin/env python3
import numpy as np
"""Script that calculates the specificity"""


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    Specificity = TN / (TN + FP)
    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   Rows = true classes, Columns = predicted classes
    Returns:
        numpy.ndarray: Specificity for each class, shape (classes,)
    """
    # True positives for each class (diagonal)
    true_positives = np.diag(confusion)

    # False positives: sum of column minus TP
    false_positives = np.sum(confusion, axis=0) - true_positives

    # False negatives: sum of row minus TP
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # Total samples
    total = np.sum(confusion)

    # True negatives = everything else
    true_negatives = total - true_positives - false_positives - false_negatives

    # Specificity = TN / (TN + FP)
    specificity_scores = np.divide(
        true_negatives,
        true_negatives + false_positives,
        out=np.zeros_like(true_negatives, dtype=float),
        where=(true_negatives + false_positives) != 0
    )

    return specificity_scores
