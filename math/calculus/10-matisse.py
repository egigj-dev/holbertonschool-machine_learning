#!/usr/bin/env python3
"""Derivative of a polynomial function."""


def poly_derivative(poly):
    """Derivative of a polynomial function."""
    return [poly[i] * i for i in range(len(poly)) if i > 0]
