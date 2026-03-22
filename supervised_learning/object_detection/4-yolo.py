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
        """
        Loads all images from a folder

        Parameters:
        - folder_path: str, path to folder with images

        Returns:
        - images: list of images as np.ndarrays
        - image_paths: list of file paths corresponding to images
        """
        images = []
        image_paths = []

        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            # Check if it is a file and has an image extension
            if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img = cv2.imread(filepath)
                if img is not None:
                    images.append(img)
                    image_paths.append(filepath)

        return images, image_paths
