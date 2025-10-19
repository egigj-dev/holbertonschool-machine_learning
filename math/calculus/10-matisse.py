#!/usr/bin/env python3
"""Derivative of a polynomial function."""


def poly_derivative(poly):
    """Derivative of a polynomial function."""
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    if len(poly) == 1:
        return [0]
    return [poly[i] * i for i in range(len(poly)) if i > 0]
