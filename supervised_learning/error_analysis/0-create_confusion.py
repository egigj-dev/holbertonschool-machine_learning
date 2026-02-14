#!/usr/bin/env python3
"""
Module that creates a confusion matrix from one-hot encoded
true labels and predicted labels for classification tasks.
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded true labels
    and predicted labels
    """
    # Convert one-hot to class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=float)

    # Count occurrences for each true-predicted pair
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1.0

    return confusion
