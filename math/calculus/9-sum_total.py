#!/usr/bin/env python3
"""Summation of the first n integers."""


def summation_i_squared(n):
    """Summation of the first n integers."""
    if isinstance(n, int) and n > 0:
        return n*(n+1)*(2*n+1)//6
    else:
        return None
