#!/usr/bin/env python3
""" Determines whether training should stop early based on validation cost """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Early stopping occurs if the validation cost has not decreased
    relative to the optimal cost by more than `threshold` for `patience` steps

    Parameters:
    cost: Current validation cost
    opt_cost: Lowest recorded validation cost
    threshold: Minimum improvement required to reset count
    patience: Number of consecutive steps to wait before stopping
    count: Current count of steps without sufficient improvement
    """

    # Check if cost improved sufficiently
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    stop = count >= patience

    return stop, count
