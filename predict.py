#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 + DeepSORT Vehicle Tracking 🚀
GPL-3.0 License

This script integrates Ultralytics YOLOv8 (object detection) with DeepSORT
(object tracking). It detects vehicles in a video feed, tracks them across
frames, and counts objects crossing a virtual line (for "entering" vs "leaving").

Features:
  - Robust YOLOv8 inference via Ultralytics API
  - DeepSORT tracker for unique IDs across frames
  - Custom drawing utilities (bounding boxes, labels, trails)
  - Vehicle counting by direction across a predefined line
  - Works with webcam, video files, or images

Author: Adapted for clarity and maintainability
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import hydra
import argparse
import time
from pathlib import Path
from collections import deque

import cv2
import torch
import numpy as np

from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


# ---------------------------------------------------------------------------
# Global state & constants
# ---------------------------------------------------------------------------

# Color palette for random/consistent color generation
PALETTE = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# Object trajectory memory (each ID gets a deque of past positions)
data_deque: dict[int, deque] = {}

# Counters for objects entering and leaving
object_counter = {}
object_counter1 = {}

# DeepSORT tracker instance
deepsort = None

# Virtual line for counting (x1, y1) → (x2, y2)
COUNT_LINE = [(100, 500), (1050, 500)]


# ---------------------------------------------------------------------------
# Tracker initialization
# ---------------------------------------------------------------------------

def init_tracker():
    """
    Initialize the DeepSORT tracker using its YAML config.
    Global variable `deepsort` will hold the tracker instance.
    """
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=True,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def xyxy_to_xywh(*xyxy):
    """Convert [x1, y1, x2, y2] box format into [center_x, center_y, width, height]."""
    x1, y1, x2, y2 = [val.item() for val in xyxy]
    bbox_w, bbox_h = abs(x2 - x1), abs(y2 - y1)
    return x1 + bbox_w / 2, y1 + bbox_h / 2, bbox_w, bbox_h


def compute_color_for_labels(label: int):
    """
    Assign a fixed color for each class label.
    Specific colors are chosen for common vehicle classes, otherwise random.
    """
    if label == 0:   # person
        return (85, 45, 255)
    elif label == 2: # car
        return (222, 82, 175)
    elif label == 3: # motorbike
        return (0, 204, 255)
    elif label == 5: # bus
        return (0, 149, 255)
    else:
        return tuple(int((p * (label ** 2 - label + 1)) % 255) for p in PALETTE)


def ccw(A, B, C):
    """Check if points A, B, C are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    """Check if line AB intersects line CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def get_direction(pt1, pt2) -> str:
    """
    Determine movement direction (North/South/East/West) between two points.
    Used to decide whether objects are "entering" or "leaving".
    """
    direction = ""
    if pt1[1] > pt2[1]:
        direction += "South"
    elif pt1[1] < pt2[1]:
        direction += "North"
    if pt1[0] > pt2[0]:
        direction += "East"
    elif pt1[0] < pt2[0]:
        direction += "West"
    return direction


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

def UI_box(box, img, color, label, line_thickness=2):
    """Draw a bounding box with label and a custom styled border."""
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    if label:
        tf = max(line_thickness - 1, 1)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 0.5, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    """
    Draw tracked bounding boxes and update counts when crossing the line.
    """
    # Draw counting line
    cv2.line(img, COUNT_LINE[0], COUNT_LINE[1], (46, 162, 112), 3)

    # Remove lost objects from memory
    for key in list(data_deque):
        if identities is None or key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(c) for c in box]
        id = int(identities[i]) if identities is not None else 0

        # Create trajectory memory for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)

        # Object info
        obj_name = names[object_id[i]]
        color = compute_color_for_labels(object_id[i])
        label = f"{id}:{obj_name}"

        # Append trajectory point
        center = (int((x1 + x2) / 2), int(y2))
        data_deque[id].appendleft(center)

        # Check if trajectory crosses the counting line
        if len(data_deque[id]) >= 2:
            if intersect(data_deque[id][0], data_deque[id][1], COUNT_LINE[0], COUNT_LINE[1]):
                direction = get_direction(data_deque[id][0], data_deque[id][1])
                if "South" in direction:  # Leaving
                    object_counter[obj_name] = object_counter.get(obj_name, 0) + 1
                elif "North" in direction:  # Entering
                    object_counter1[obj_name] = object_counter1.get(obj_name, 0) + 1

        # Draw bounding box and trail
        UI_box(box, img, color=color, label=label)
        for j in range(1, len(data_deque[id])):
            if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
            cv2.line(img, data_deque[id][j - 1], data_deque[id][j], color, thickness)

    # Display counters
    h, w, _ = img.shape
    for idx, (cls, cnt) in enumerate(object_counter1.items()):
        cv2.putText(img, f"{cls} Entering: {cnt}", (w - 300, 50 + idx * 30),
                    0, 0.8, (85, 45, 255), 2, lineType=cv2.LINE_AA)
    for idx, (cls, cnt) in enumerate(object_counter.items()):
        cv2.putText(img, f"{cls} Leaving: {cnt}", (30, 50 + idx * 30),
                    0, 0.8, (85, 45, 255), 2, lineType=cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class DetectionPredictor(BasePredictor):
    """YOLOv8 predictor extended with DeepSORT-based multi-object tracking."""

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        """Convert numpy image to torch tensor, normalize, and move to device."""
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        return img / 255.0

    def postprocess(self, preds, img, orig_img):
        """Apply NMS and rescale boxes back to original image size."""
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou,
                                        agnostic=self.args.agnostic_nms, max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def write_results(self, idx, preds, batch):
        """Run DeepSORT on YOLO predictions, update trackers, and draw results."""
        p, im, im0 = batch
        if len(im.shape) == 3:
            im = im[None]

        det = preds[idx]
        if len(det) == 0:
            return ""

        xywhs, confs, oids = [], [], []
        for *xyxy, conf, cls in reversed(det):
            xywhs.append(xyxy_to_xywh(*xyxy))
            confs.append([conf.item()])
            oids.append(int(cls))

        outputs = deepsort.update(torch.Tensor(xywhs), torch.Tensor(confs), oids, im0)
        if len(outputs) > 0:
            bbox_xyxy, identities, object_id = outputs[:, :4], outputs[:, -2], outputs[:, -1]
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)
        return "Processed frame"


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    """Hydra entrypoint: initialize tracker, build predictor, and run inference."""
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
