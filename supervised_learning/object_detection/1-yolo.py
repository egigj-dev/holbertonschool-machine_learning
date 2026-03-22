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
        """
        self.model = load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs for a single image
        Parameters:
        - outputs: list of numpy.ndarrays, predictions from the Darknet model
        - image_size: np.ndarray, original image size [height, width]

        Returns:
        - boxes: list of np.ndarrays of shape (grid_h, grid_w, anchor_boxes, 4) containing
                 (x1, y1, x2, y2) relative to original image
        - box_confidences: list of np.ndarrays of shape (grid_h, grid_w, anchor_boxes, 1)
        - box_class_probs: list of np.ndarrays of shape (grid_h, grid_w, anchor_boxes, classes)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract t_x, t_y, t_w, t_h, box_confidence, class_probs
            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_conf = output[..., 4:5]
            class_probs = output[..., 5:]

            # Sigmoid for xy and box confidence
            sigmoid_xy = 1 / (1 + np.exp(-t_xy))
            box_confidence = 1 / (1 + np.exp(-box_conf))
            class_prob = 1 / (1 + np.exp(-class_probs))

            # Create grid
            grid_x = np.arange(grid_w)
            grid_y = np.arange(grid_h)
            cx, cy = np.meshgrid(grid_x, grid_y)
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            # bx, by: center coordinates relative to image (0-1)
            bx = (sigmoid_xy[..., 0] + cx) / grid_w
            by = (sigmoid_xy[..., 1] + cy) / grid_h

            # bw, bh: width and height relative to image
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            bw = (anchor_w * np.exp(t_wh[..., 0])) / input_w
            bh = (anchor_h * np.exp(t_wh[..., 1])) / input_h

            # Convert to corner coordinates in original image scale
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_prob)

        return boxes, box_confidences, box_class_probs

