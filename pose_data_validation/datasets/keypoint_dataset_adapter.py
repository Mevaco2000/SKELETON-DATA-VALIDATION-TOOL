"""Adapter that exposes non-YOLO keypoint datasets as YOLOPoseDataset.

Main goal: keep the existing project workflow unchanged.
You can instantiate this adapter from COCO/custom formats and then use all
methods available on YOLOPoseDataset and validation classes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import weakref
from typing import Any, Dict, Optional

from .format_converters import (
    convert_coco_keypoints_with_ultralytics,
    convert_registered_format_to_yolo_pose,
)
from .yolo_pose_dataset import YOLOPoseDataset


class KeypointDatasetAdapter:
    """Expose converted keypoint data through YOLOPoseDataset API.

    The adapter converts source data to YOLO Pose once and stores conversion
    metadata. After that it proxies attribute access to an internal
    ``YOLOPoseDataset`` instance.
    """

    def __init__(
        self,
        source_format: str,
        input_root: str,
        output_root: Optional[str] = None,
        split_name: str = "train",
        copy_images: bool = True,
        use_ultralytics_for_coco: bool = True,
        coco_annotations_dir: Optional[str] = None,
        coco_images_dir: Optional[str] = None,
    ) -> None:
        self.source_format = str(source_format).strip().lower()
        self.input_root = os.path.abspath(input_root)
        self.split_name = str(split_name)
        self.copy_images = bool(copy_images)
        self._owns_output_root = output_root is None
        self._closed = False

        if output_root is None:
            self.output_root = tempfile.mkdtemp(prefix="keypoint_adapter_")
        else:
            self.output_root = os.path.abspath(output_root)
            os.makedirs(self.output_root, exist_ok=True)

        try:
            if self.source_format == "coco_keypoints" and use_ultralytics_for_coco:
                annotations_dir = (
                    os.path.abspath(coco_annotations_dir)
                    if coco_annotations_dir is not None
                    else self.input_root
                )
                images_dir = (
                    os.path.abspath(coco_images_dir)
                    if coco_images_dir is not None
                    else None
                )
                self.conversion_info = convert_coco_keypoints_with_ultralytics(
                    coco_annotations_dir=annotations_dir,
                    output_root=self.output_root,
                    copy_images_from=images_dir,
                    split_name=self.split_name,
                )
            else:
                self.conversion_info = convert_registered_format_to_yolo_pose(
                    format_name=self.source_format,
                    input_root=self.input_root,
                    output_root=self.output_root,
                    split_name=self.split_name,
                    copy_images=self.copy_images,
                    generate_split_file=True,
                )

            split_file = self.conversion_info.get("split_file")
            if not split_file:
                raise ValueError("Conversion did not produce split file path")

            self.dataset = YOLOPoseDataset(
                dataset_file=str(split_file),
                labels_dir=os.path.join(self.output_root, "labels"),
            )
        except BaseException:
            if self._owns_output_root:
                shutil.rmtree(self.output_root, ignore_errors=True)
            raise

        # The temporary folder is removed when the dataset object is garbage
        # collected (or on close()/exit), not when the adapter is: the dataset
        # returned by as_yolo_dataset() reads files from that folder and may
        # outlive the adapter.
        self._cleanup = None
        if self._owns_output_root:
            self._cleanup = weakref.finalize(
                self.dataset, shutil.rmtree, self.output_root, ignore_errors=True
            )

    @classmethod
    def from_coco(
        cls,
        annotations_dir: str,
        images_dir: Optional[str] = None,
        output_root: Optional[str] = None,
        split_name: str = "train",
        use_ultralytics: bool = True,
    ) -> "KeypointDatasetAdapter":
        """Create adapter directly from COCO keypoints data."""
        return cls(
            source_format="coco_keypoints",
            input_root=annotations_dir,
            output_root=output_root,
            split_name=split_name,
            copy_images=images_dir is not None,
            use_ultralytics_for_coco=use_ultralytics,
            coco_annotations_dir=annotations_dir,
            coco_images_dir=images_dir,
        )

    @classmethod
    def from_registered_format(
        cls,
        format_name: str,
        input_root: str,
        output_root: Optional[str] = None,
        split_name: str = "train",
        copy_images: bool = True,
    ) -> "KeypointDatasetAdapter":
        """Create adapter from any format available in FORMAT_REGISTRY."""
        return cls(
            source_format=format_name,
            input_root=input_root,
            output_root=output_root,
            split_name=split_name,
            copy_images=copy_images,
            use_ultralytics_for_coco=False,
        )

    def as_yolo_dataset(self) -> YOLOPoseDataset:
        """Return internal YOLOPoseDataset instance."""
        return self.dataset

    def info(self) -> Dict[str, Any]:
        """Return adapter metadata useful in notebooks/reports."""
        return {
            "source_format": self.source_format,
            "input_root": self.input_root,
            "output_root": self.output_root,
            "split_name": self.split_name,
            "owns_output_root": self._owns_output_root,
            "conversion_info": dict(self.conversion_info),
            "converted_dataset_file": self.dataset.dataset_file,
            "converted_labels_dir": self.dataset.labels_dir,
        }

    def close(self) -> None:
        """Remove the temporary conversion output now, if the adapter owns it.

        After this the dataset returned by ``as_yolo_dataset()`` can no longer
        load images or labels.
        """
        if self._closed:
            return
        self._closed = True

        if self._cleanup is not None:
            self._cleanup()

    def __enter__(self) -> "KeypointDatasetAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __getattr__(self, item: str):
        """Proxy unknown attributes/methods to YOLOPoseDataset.

        This lets existing code call dataset methods directly on adapter.
        """
        # Avoid infinite recursion when __init__ failed before self.dataset was set.
        if item == "dataset" or item.startswith("__"):
            raise AttributeError(item)
        return getattr(self.dataset, item)
