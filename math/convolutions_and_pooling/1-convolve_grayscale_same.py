#!/usr/bin/env python3
"""Script for performing same convolution on grayscale images."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute padding size
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad images with zeros
    padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    # Prepare output
    output = np.zeros((m, h, w))

    # Perform convolution (only two loops)
    for i in range(h):
        for j in range(w):
            region = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
