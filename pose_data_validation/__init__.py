"""
Validation of keypoint (skeleton) annotations in YOLO Pose datasets.

Provides tools for:
- keypoint anomaly ranking (inter-keypoint distances, LBP texture, segmentation masks),
- the merged ranking of these signals (MergedRankingEvaluator),
- near-duplicate detection with CLIP embeddings,
- model-in-the-loop validation with YOLO Pose models,
- dataset operations (flatten, merge, split, convert, synthetic keypoint noise).
"""

from importlib import import_module
from typing import Any, Dict, Tuple

__version__ = "0.1.0"
__author__ = "Rafał Wysocki, Poznań University of Technology"

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    
    "YPImageValidation": (".validation", "YPImageValidation"),
    "YPSetValidation": (".validation", "YPSetValidation"),
    "YPValidation_Test": (".validation", "YPValidation_Test"),
    "MergedRankingEvaluator": (".validation", "MergedRankingEvaluator"),
    "AnglePreservingKeypointReplacement": (".datasets", "AnglePreservingKeypointReplacement"),
    "AxisAlignedKeypointReplacement": (".datasets", "AxisAlignedKeypointReplacement"),
    "RigidKeypointReplacement": (".datasets", "RigidKeypointReplacement"),
    "YOLOPoseImage": (".datasets", "YOLOPoseImage"),
    "YOLOPoseDataset": (".datasets", "YOLOPoseDataset"),
    "KeypointDatasetAdapter": (".datasets", "KeypointDatasetAdapter"),
    "flatten_cvat_yolo_pose": (".datasets", "flatten_cvat_yolo_pose"),
    "replace_angle_preserving_keypoints": (".datasets", "replace_angle_preserving_keypoints"),
    "replace_axis_aligned_keypoints": (".datasets", "replace_axis_aligned_keypoints"),
    "replace_rigid_keypoints": (".datasets", "replace_rigid_keypoints"),
    "merge_yolo_pose_datasets": (".datasets", "merge_yolo_pose_datasets"),
    "split_yolo_pose_dataset": (".datasets", "split_yolo_pose_dataset"),
    "convert_yolo_pose_to_cvat": (".datasets", "convert_yolo_pose_to_cvat"),
    "export_dataset_partitions": (".datasets", "export_dataset_partitions"),
    "remove_duplicate_samples_from_dataset": (".datasets", "remove_duplicate_samples_from_dataset"),
    "extract_image_subset": (".datasets", "extract_image_subset"),
    "MODEL_REGISTRY": (".config", "MODEL_REGISTRY"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))

__all__ = [
    # Validation - classes
    "YPImageValidation",
    "YPSetValidation",
    "YPValidation_Test",
    "MergedRankingEvaluator",
    # Datasets - classes
    "AnglePreservingKeypointReplacement",
    "AxisAlignedKeypointReplacement",
    "RigidKeypointReplacement",
    "YOLOPoseImage",
    "YOLOPoseDataset",
    "KeypointDatasetAdapter",
    # Datasets - operations
    "flatten_cvat_yolo_pose",
    "replace_angle_preserving_keypoints",
    "replace_axis_aligned_keypoints",
    "replace_rigid_keypoints",
    "merge_yolo_pose_datasets",
    "split_yolo_pose_dataset",
    "convert_yolo_pose_to_cvat",
    "export_dataset_partitions",
    "remove_duplicate_samples_from_dataset",
    "extract_image_subset",
    # Config
    "MODEL_REGISTRY",
]
