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
            scores = conf * class_prob  # shape: (grid_h, grid_w, anchor_boxes, classes)
            class_indices = np.argmax(scores, axis=-1)  # best class per box
            class_scores = np.max(scores, axis=-1)  # best score per box

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
                keep_idxs = np.where(iou <= self.nms_t)[0] + 1  # +1 because iou excludes first box
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
            if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img = cv2.imread(filepath)
                if img is not None:
                    images.append(img)
                    image_paths.append(filepath)

        return images, image_paths

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
