#!/usr/bin/env python3
"""Integrate a polynomial function."""


def poly_integral(poly, C=0):
    """Integrate a polynomial function."""
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    return [C] + [poly[i] / (i + 1) for i in range(len(poly))]
