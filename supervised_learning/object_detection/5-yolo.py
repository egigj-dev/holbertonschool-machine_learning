#!/usr/bin/env python3
"""Yolo v3 object detection class"""
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class Yolo:
    """
    Yolo v3 object detection class
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor for Yolo class
        """
        self.model = load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        # ... (implementation from 2-yolo.py) ...
        pass

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        # ... (implementation from 2-yolo.py) ...
        pass

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        # ... (implementation from 2-yolo.py) ...
        pass

    @staticmethod
    def load_images(folder_path):
        # ... (implementation from 3-yolo.py) ...
        pass

    def preprocess_images(self, images):
        """
        Preprocess images for Darknet model

        Parameters:
        - images: list of images as np.ndarrays

        Returns:
        - pimages: np.ndarray of shape (ni, input_h, input_w, 3)
        - image_shapes: np.ndarray of shape (ni, 2) with original (height, width)
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            original_shape = img.shape[:2]  # (height, width)
            image_shapes.append(original_shape)

            # Resize image using cubic interpolation
            resized_img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            # Scale pixel values to [0, 1]
            normalized_img = resized_img / 255.0
            pimages.append(normalized_img)

        pimages = np.array(pimages, dtype=np.float32)
        image_shapes = np.array(image_shapes, dtype=np.int)

        return pimages, image_shapes
