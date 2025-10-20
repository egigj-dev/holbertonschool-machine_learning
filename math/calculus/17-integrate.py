#!/usr/bin/env python3
"""Integrate a polynomial function."""


def poly_integral(poly, C=0):
    """Integrate a polynomial function."""
    if (not isinstance(poly, list) or
        not all(isinstance(c, (int, float)) for c in poly) or
        not isinstance(C, (int, float))):
        return None
    result = [C] + [
        int(c / (i + 1)) if (c / (i + 1)).is_integer() else c / (i + 1)
        for i, c in enumerate(poly)
    ]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result
