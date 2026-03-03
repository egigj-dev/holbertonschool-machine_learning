#!/usr/bin/env python3
"""PCA color augmentation (AlexNet style)"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as in the AlexNet paper
    """
    # Flatten image to [H*W, 3]
    orig_shape = tf.shape(image)
    flat_image = tf.reshape(tf.cast(image, tf.float32), [-1, 3])
    
    # Compute covariance
    mean = tf.reduce_mean(flat_image, axis=0, keepdims=True)
    centered = flat_image - mean
    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flat_image)[0], tf.float32)
    
    # Eigen decomposition
    s, u, _ = tf.linalg.svd(cov)
    
    # Add noise along principal components
    alpha = tf.constant(alphas, dtype=tf.float32)
    delta = tf.matmul(u, tf.multiply(s, alpha)[:, tf.newaxis])
    
    augmented = flat_image + delta[:, 0]
    
    # Reshape back to original image
    return tf.reshape(tf.clip_by_value(augmented, 0, 255), orig_shape)
