#!/usr/bin/env python3
"""Rotate image 90 degrees counter-clockwise"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise
    """
    return tf.image.rot90(image, k=1)
