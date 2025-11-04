#!/usr/bin/env python3
"""Script that plots scattered graph"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """Function that plots scattered graph"""
    np.random.seed(5)
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    # Scatter plot with colormap based on elevation z
    scatter = plt.scatter(x, y, c=z, cmap='viridis')

    # Axis labels
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")

    # Title
    plt.title("Mountain Elevation")

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("elevation (m)")
    plt.show()
