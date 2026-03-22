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
        self.model = load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        # ... implementation from 2-yolo.py ...
        pass

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        # ... implementation from 2-yolo.py ...
        pass

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        # ... implementation from 2-yolo.py ...
        pass

    @staticmethod
    def load_images(folder_path):
        # ... implementation from 3-yolo.py ...
        pass

    def preprocess_images(self, images):
        # ... implementation from 4-yolo.py ...
        pass

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        # ... implementation from 5-yolo.py ...
        pass

    def predict(self, folder_path):
        """
        Performs object detection on all images in a folder

        Parameters:
        - folder_path: str, path to folder with images

        Returns:
        - predictions: list of tuples (boxes, box_classes, box_scores) per image
        - image_paths: list of image file paths
        """
        # Load images and their paths
        images, image_paths = self.load_images(folder_path)

        # Preprocess images
        pimages, image_shapes = self.preprocess_images(images)

        predictions = []

        # Iterate through each image
        for i, image in enumerate(images):
            # Model prediction
            outputs = self.model.predict(np.expand_dims(pimages[i], axis=0))
            
            # Ensure outputs are a list of arrays (YOLOv3 has 3 outputs)
            if not isinstance(outputs, list):
                outputs = [outputs]

            # Process outputs
            boxes, box_confidences, box_class_probs = self.process_outputs(outputs, image_shapes[i])

            # Filter boxes
            filtered_boxes, box_classes, box_scores = self.filter_boxes(boxes, box_confidences, box_class_probs)

            # Apply Non-max suppression
            pred_boxes, pred_classes, pred_scores = self.non_max_suppression(filtered_boxes, box_classes, box_scores)

            # Show boxes on image (window named by file name without path)
            file_name = os.path.basename(image_paths[i])
            self.show_boxes(image, pred_boxes, pred_classes, pred_scores, file_name)

            predictions.append((pred_boxes, pred_classes, pred_scores))

        return predictions, image_paths
