#!/usr/bin/env python3
""" Script that performs a valid convolution on grayscale images """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Module that performs a valid convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions for valid convolution
    oh = h - kh + 1
    ow = w - kw + 1

    # Initialize output array
    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            # Extract the region from all images at once using slicing
            region = images[:, i:i+kh, j:j+kw]  # shape: (m, kh, kw)
 
            # Element-wise multiply with kernel and sum
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
