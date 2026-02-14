#!/usr/bin/env python3
import numpy as np
"""Script that calculates the specificity"""


def specificity(confusion):
	"""Specificity = TN/(TN+FP)"""

	# TN = total - TP - FP - FN
	true_positives = np.diag(confusion)
	false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    total = np.sum(confusion)

    true_negatives = total - true_positives - false_positives - false_negatives


	# TN/(TN+FP)
	specificity_scores = np.divide(
        true_negatives,
        true_negatives + false_positives,
        out=np.zeros_like(true_negatives, dtype=float),
        where=(true_negatives + false_positives) != 0
    )

    return specificity_scores
