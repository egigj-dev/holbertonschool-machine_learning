#!/usr/bin/env python3
"""Back propagation over a pooling layer"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    _, h_new, w_new, _ = dA.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            if mode == 'max':
                # Create mask of max elements
                mask = (A_slice == np.max(A_slice, axis=(1, 2), keepdims=True))
                dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, :] += \
                    mask * dA[:, i, j, :][:, None, None, :]
            elif mode == 'avg':
                # Distribute gradient evenly
                da = dA[:, i, j, :][:, None, None, :]
                shape = (m, kh, kw, c)
                dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, :] += \
                    da / (kh * kw)
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return dA_prev
