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
        # ... (implementation from 4-yolo.py) ...
        pass

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Draws boxes, class names, and scores on an image and displays it
        Saves image if 's' key is pressed
        """
        img_copy = image.copy()

        for box, cls_idx, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box.astype(int)
            class_name = self.class_names[cls_idx]
            label = f"{class_name} {score:.2f}"

            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

            # Draw text above rectangle
            cv2.putText(img_copy, label, (x1, max(y1 - 5, 0)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        # Display image
        window_name = file_name
        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(0) & 0xFF

        # Save image if 's' pressed
        if key == ord('s'):
            save_dir = "detections"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, os.path.basename(file_name))
            cv2.imwrite(save_path, img_copy)

        cv2.destroyAllWindows()
