#!/usr/bin/env python3
"""Module for performing convolution on images with multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        ph = kh // 2
        pw = kw // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, nc))

    # Convolution (three loops: over kernels, height, width)
    for k in range(nc):
        kernel = kernels[:, :, :, k]
        for i in range(out_h):
            vert_start = i * sh
            vert_end = vert_start + kh
            for j in range(out_w):
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                region = padded[:, vert_start:vert_end,
                                horiz_start:horiz_end, :]

                output[:, i, j, k] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
