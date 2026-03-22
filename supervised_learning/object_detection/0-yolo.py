#!/usr/bin/env python3
"""Yolo v3 object detection class"""
import numpy as np
from tensorflow.keras.models import load_model


class Yolo:
    """
    Yolo v3 object detection class
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor for Yolo class

        Parameters:
        - model_path: str, path to Darknet Keras model
        - classes_path: str, path to text file with class names
        - class_t: float, box score threshold for filtering
        - nms_t: float, IOU threshold for non-max suppression
        - anchors: np.ndarray, shape (outputs, anchor_boxes, 2)
        """
        # Load the Darknet Keras model
        self.model = load_model(model_path)

        # Load class names from file
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Store thresholds and anchors
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
