#!/usr/bin/env python3
"""Summation of the first n integers."""


def summation_i_squared(n):
    """Summation of the first n integers."""
    if isinstance(n, int) and n > 0:
        squares = [i**2 for i in range(1, n + 1)]
        return sum(squares)
    else:
        return None
