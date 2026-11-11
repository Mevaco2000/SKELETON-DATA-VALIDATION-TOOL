"""Definitions and registry for keypoint dataset formats.

This module provides a lightweight extension point: add a new format by
subclassing ``KeypointDatasetFormat`` and registering it in ``FORMAT_REGISTRY``.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .yolo_pose_dataset import _candidate_label_paths


def bbox_from_keypoints(keypoints: np.ndarray) -> Tuple[float, float, float, float]:
    """Return a normalized YOLO bbox (cx, cy, w, h) around the visible keypoints.

    Uses all keypoints when none is visible (or there is no visibility column).
    """
    keypoints = np.asarray(keypoints, dtype=float)
    if keypoints.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    coords = keypoints[:, :2]
    if keypoints.shape[1] >= 3 and np.any(keypoints[:, 2] > 0):
        coords = coords[keypoints[:, 2] > 0]
    x_min, y_min = (float(value) for value in coords.min(axis=0))
    x_max, y_max = (float(value) for value in coords.max(axis=0))
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, max(0.0, x_max - x_min), max(0.0, y_max - y_min)


@dataclass
class KeypointAnnotation:
    """One person/object keypoint annotation for a single image."""

    class_id: int
    keypoints: np.ndarray  # (N, 2) or (N, 3) in normalized coordinates
    bbox_xywh: Optional[Tuple[float, float, float, float]] = None  # normalized YOLO bbox


@dataclass
class KeypointSample:
    """One image and all keypoint annotations associated with it."""

    image_path: str
    relative_image_path: str
    annotations: List[KeypointAnnotation]


class KeypointDatasetFormat(ABC):
    """Base class for custom keypoint dataset formats."""

    name: str

    @abstractmethod
    def iter_samples(self, input_root: str) -> Iterable[KeypointSample]:
        """Yield normalized samples from ``input_root``."""


class YoloPoseFormat(KeypointDatasetFormat):
    """Read an existing YOLO Pose dataset as normalized keypoint samples."""

    name = "yolo_pose"

    def __init__(self, split_file: str = "train.txt", labels_dir: str = "labels") -> None:
        self.split_file = split_file
        self.labels_dir = labels_dir

    @staticmethod
    def _parse_label_line(line: str) -> Optional[KeypointAnnotation]:
        parts = line.strip().split()
        if len(parts) < 6:
            return None

        class_id = int(float(parts[0]))
        bbox = tuple(float(v) for v in parts[1:5])
        values = [float(v) for v in parts[5:]]

        if not values:
            keypoints = np.empty((0, 3), dtype=float)
            return KeypointAnnotation(class_id=class_id, keypoints=keypoints, bbox_xywh=bbox)

        if len(values) % 3 == 0:
            keypoints = np.asarray(values, dtype=float).reshape(-1, 3)
        elif len(values) % 2 == 0:
            keypoints_xy = np.asarray(values, dtype=float).reshape(-1, 2)
            visibility = np.full((keypoints_xy.shape[0], 1), 2.0, dtype=float)
            keypoints = np.hstack([keypoints_xy, visibility])
        else:
            return None

        return KeypointAnnotation(class_id=class_id, keypoints=keypoints, bbox_xywh=bbox)

    def iter_samples(self, input_root: str) -> Iterable[KeypointSample]:
        root_abs = os.path.abspath(input_root)
        split_path = os.path.join(root_abs, self.split_file)
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r", encoding="utf-8-sig") as split_handle:
            for raw_image_rel_path in split_handle:
                image_rel_path = raw_image_rel_path.strip().replace('\\', '/')
                if not image_rel_path:
                    continue

                image_path = os.path.abspath(os.path.join(root_abs, image_rel_path))
                # Same lookup as YOLOPoseDataset.
                label_path = next(
                    (
                        candidate
                        for candidate in _candidate_label_paths(
                            image_rel_path, os.path.join(root_abs, self.labels_dir), root_abs
                        )
                        if os.path.isfile(candidate)
                    ),
                    None,
                )
                if label_path is None:
                    continue

                annotations: List[KeypointAnnotation] = []
                with open(label_path, "r", encoding="utf-8-sig") as label_handle:
                    for line in label_handle:
                        parsed = self._parse_label_line(line)
                        if parsed is not None:
                            annotations.append(parsed)

                yield KeypointSample(
                    image_path=image_path,
                    relative_image_path=image_rel_path,
                    annotations=annotations,
                )


class CocoKeypointsFormat(KeypointDatasetFormat):
    """Read COCO-style keypoints JSON as normalized keypoint samples."""

    name = "coco_keypoints"

    def __init__(self, annotation_file: str = "annotations/person_keypoints_train2017.json") -> None:
        self.annotation_file = annotation_file

    @staticmethod
    def _bbox_from_coco(
        bbox_xywh_pixels: Optional[List[float]],
        width: float,
        height: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Convert a COCO [x, y, w, h] pixel box to a normalized YOLO box."""
        if not bbox_xywh_pixels or len(bbox_xywh_pixels) != 4:
            return None
        x, y, box_w, box_h = (float(value) for value in bbox_xywh_pixels)
        if box_w <= 0 or box_h <= 0:
            return None
        x0 = min(max(x / width, 0.0), 1.0)
        y0 = min(max(y / height, 0.0), 1.0)
        x1 = min(max((x + box_w) / width, 0.0), 1.0)
        y1 = min(max((y + box_h) / height, 0.0), 1.0)
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0

    def iter_samples(self, input_root: str) -> Iterable[KeypointSample]:
        root_abs = os.path.abspath(input_root)
        annotation_path = os.path.join(root_abs, self.annotation_file)
        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

        with open(annotation_path, "r", encoding="utf-8") as handle:
            coco = json.load(handle)

        images_by_id: Dict[int, Dict[str, object]] = {int(img["id"]): img for img in coco.get("images", [])}

        categories = sorted({int(cat["id"]) for cat in coco.get("categories", [])})
        class_map = {cat_id: idx for idx, cat_id in enumerate(categories)}

        grouped_annotations: Dict[int, List[KeypointAnnotation]] = {}
        for ann in coco.get("annotations", []):
            if "keypoints" not in ann:
                continue
            # Crowd regions are not individual persons (Ultralytics skips them too).
            if ann.get("iscrowd", 0):
                continue

            image_id = int(ann["image_id"])
            image_meta = images_by_id.get(image_id)
            if image_meta is None:
                continue

            width = float(image_meta.get("width", 0.0))
            height = float(image_meta.get("height", 0.0))
            if width <= 0 or height <= 0:
                continue

            kp_values = ann.get("keypoints", [])
            if not kp_values or len(kp_values) % 3 != 0:
                continue

            keypoints_xyv = np.asarray(kp_values, dtype=float).reshape(-1, 3)
            keypoints_xyv[:, 0] = keypoints_xyv[:, 0] / width
            keypoints_xyv[:, 1] = keypoints_xyv[:, 1] / height
            keypoints_xyv[:, :2] = np.clip(keypoints_xyv[:, :2], 0.0, 1.0)

            class_id = class_map.get(int(ann.get("category_id", -1)), 0)
            bbox = self._bbox_from_coco(ann.get("bbox"), width, height)
            if bbox is None:
                bbox = bbox_from_keypoints(keypoints_xyv)
            grouped_annotations.setdefault(image_id, []).append(
                KeypointAnnotation(class_id=class_id, keypoints=keypoints_xyv, bbox_xywh=bbox)
            )

        for image_id, image_meta in images_by_id.items():
            file_name = str(image_meta.get("file_name", "")).replace('\\', '/')
            if not file_name:
                continue

            if file_name.startswith("images/"):
                rel_path = file_name
            else:
                rel_path = os.path.join("images", file_name).replace('\\', '/')

            image_path = os.path.abspath(os.path.join(root_abs, rel_path))
            annotations = grouped_annotations.get(image_id, [])
            yield KeypointSample(
                image_path=image_path,
                relative_image_path=rel_path,
                annotations=annotations,
            )


class KeypointFormatRegistry:
    """Registry for discoverable keypoint dataset formats."""

    def __init__(self) -> None:
        self._formats: Dict[str, KeypointDatasetFormat] = {}

    def register(self, format_impl: KeypointDatasetFormat) -> None:
        self._formats[format_impl.name.lower()] = format_impl

    def get(self, name: str) -> KeypointDatasetFormat:
        key = name.lower().strip()
        if key not in self._formats:
            available = ", ".join(sorted(self._formats))
            raise KeyError(f"Unknown dataset format '{name}'. Available formats: {available}")
        return self._formats[key]

    def list_formats(self) -> List[str]:
        return sorted(self._formats.keys())


FORMAT_REGISTRY = KeypointFormatRegistry()
FORMAT_REGISTRY.register(YoloPoseFormat())
FORMAT_REGISTRY.register(CocoKeypointsFormat())
