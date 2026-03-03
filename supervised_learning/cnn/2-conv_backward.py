#!/usr/bin/env python3
"""Back propagation over a convolutional layer"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer
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

    # Pad A_prev and dA_prev
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    m, h_new, w_new, _ = dZ.shape

    # Compute gradients
    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev_pad[:, vert_start:vert_end,
                                 horiz_start:horiz_end, :]

            for k in range(c_new):
                # Update gradients for the slice
                dW[:, :, :, k] += np.sum(
                    A_slice * dZ[:, i, j, k][:, None, None, None],
                    axis=0
                )
            # Update dA_prev_pad
            for n in range(m):
                dA_prev_pad[n, vert_start:vert_end,
                            horiz_start:horiz_end, :] += np.sum(
                    W[:, :, :, :] * dZ[n, i, j, :][None, None, None, :],
                    axis=3
                )

    # Unpad dA_prev
    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:-ph if ph != 0 else None,
                               pw:-pw if pw != 0 else None, :]
    else:
        dA_prev = dA_prev_pad

    # Gradient for biases
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    return dA_prev, dW, db
