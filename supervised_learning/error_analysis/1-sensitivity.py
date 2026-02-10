#!/usr/bin/env python3
import numpy as np


def sensitivity(confusion_matrix):
    """
    Calculate sensitivity (recall) for each class.
    Sensitivity = TP/(TP+FN)
    """
    cm = confusion_matrix.astype(float)

    # Sensitivity = TP/(TP+FN)
    true_positives = np.diag(cm)
    false_negatives = cm.sum(axis=1) - true_positives

    # Avoid division by zero
    sensitivity = np.divide(
        true_positives,
        true_positives + false_negatives,
        out=np.zeros_like(true_positives),
        where=(true_positives + false_negatives) != 0
    )

    return sensitivity
