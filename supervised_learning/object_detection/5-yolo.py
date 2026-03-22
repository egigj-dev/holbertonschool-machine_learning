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
        
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_conf = output[..., 4:5]
            class_probs = output[..., 5:]

            sigmoid_xy = 1 / (1 + np.exp(-t_xy))
            box_confidence = 1 / (1 + np.exp(-box_conf))
            class_prob = 1 / (1 + np.exp(-class_probs))

            grid_x = np.arange(grid_w)
            grid_y = np.arange(grid_h)
            cx, cy = np.meshgrid(grid_x, grid_y)
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            bx = (sigmoid_xy[..., 0] + cx) / grid_w
            by = (sigmoid_xy[..., 1] + cy) / grid_h

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            bw = (anchor_w * np.exp(t_wh[..., 0])) / input_w
            bh = (anchor_h * np.exp(t_wh[..., 1])) / input_h

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on class score threshold
        Returns filtered_boxes, box_classes, box_scores
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, class_prob in zip(boxes, box_confidences, box_class_probs):
            # Compute box scores
            scores = conf * class_prob
            class_indices = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            # Mask boxes with score above threshold
            mask = class_scores >= self.class_t

            # Select boxes, classes, scores
            filtered_boxes.append(box[mask])
            box_classes.append(class_indices[mask])
            box_scores.append(class_scores[mask])

        # Concatenate all outputs into single arrays
        if filtered_boxes:
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)
            box_classes = np.concatenate(box_classes, axis=0)
            box_scores = np.concatenate(box_scores, axis=0)
        else:
            filtered_boxes = np.array([])
            box_classes = np.array([])
            box_scores = np.array([])

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to avoid overlapping boxes
        Returns box_predictions, predicted_box_classes, predicted_box_scores
        """
        if len(filtered_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        unique_classes = np.unique(box_classes)
        final_boxes = []
        final_classes = []
        final_scores = []

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            # Sort boxes by scores descending
            idxs = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[idxs]
            cls_scores = cls_scores[idxs]

            while len(cls_boxes) > 0:
                # Pick the box with highest score
                final_boxes.append(cls_boxes[0])
                final_classes.append(cls)
                final_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                # Compute IOU of the remaining boxes with the first box
                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                union = ((cls_boxes[0, 2] - cls_boxes[0, 0]) *
                         (cls_boxes[0, 3] - cls_boxes[0, 1]) +
                         (cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                         (cls_boxes[1:, 3] - cls_boxes[1:, 1]) - intersection)

                iou = intersection / union

                # Keep boxes with IOU <= nms_t
                keep_idxs = np.where(iou <= self.nms_t)[0] + 1
                cls_boxes = cls_boxes[keep_idxs]
                cls_scores = cls_scores[keep_idxs]

        return (np.array(final_boxes),
                np.array(final_classes),
                np.array(final_scores))

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
            if os.path.isfile(filepath): 
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img = cv2.imread(filepath)
                    if img is not None:
                        images.append(img)
                        image_paths.append(filepath)

        return images, image_paths

    def preprocess_images(self, images):
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2] 

        pimages = []
        image_shapes = []

        for img in images:
            original_shape = img.shape[:2]
            image_shapes.append(original_shape)
            resized_img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            normalized_img = resized_img / 255.0
            pimages.append(normalized_img)

        pimages = np.array(pimages, dtype=np.float32)
        image_shapes = np.array(image_shapes, dtype=int)  # fixed

        return pimages, image_shapes
