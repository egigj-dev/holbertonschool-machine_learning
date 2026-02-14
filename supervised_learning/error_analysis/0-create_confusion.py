#!/usr/bin/env python3
import numpy as np
""" Script that creates confusion matrix """

def create_confusion_matrix(labels, logits):
    """
    Creates confusion matrix from one-hot encoded true labels and predictions.
    """
    # Convert one-hot to class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)

    # Count occurrences for each true-predicted pair
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
