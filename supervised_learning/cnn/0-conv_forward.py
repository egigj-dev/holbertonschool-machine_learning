#!/usr/bin/env python3
"""Forward propagation over a convolutional layer"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Determine padding
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + \
            (((h_prev - 1) * sh + kh - h_prev) % 2 != 0)
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + \
            (((w_prev - 1) * sw + kw - w_prev) % 2 != 0)
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # Pad the input
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Compute output dimensions
    h_out = (h_prev + 2 * ph - kh) // sh + 1
    w_out = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize output
    Z = np.zeros((m, h_out, w_out, c_new))

    # Perform convolution
    for i in range(h_out):
        for j in range(w_out):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw
            A_slice = A_prev_pad[:, vert_start:vert_end,
                                 horiz_start:horiz_end, :]
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(
                    A_slice * W[:, :, :, k], axis=(1, 2, 3))
    Z = Z + b
    return activation(Z)
