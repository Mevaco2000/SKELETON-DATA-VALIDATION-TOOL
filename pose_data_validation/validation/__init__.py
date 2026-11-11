"""Validation module for image analysis, duplicate detection, and model evaluation."""

from importlib import import_module
from typing import Any, Dict, Tuple


_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    
    "parse_results_txt": (".merged_ranking_evaluator", "parse_results_txt"),
    "parse_distance_anomaly_report": (".merged_ranking_evaluator", "parse_distance_anomaly_report"),
    "compare_results_txt_to_distance_anomalies": (".merged_ranking_evaluator", "compare_results_txt_to_distance_anomalies"),
    "compare_results_txt_to_distance_anomalies_with_outside_keypoints": (".merged_ranking_evaluator", "compare_results_txt_to_distance_anomalies_with_outside_keypoints"),
    "save_distance_anomaly_report": (".merged_ranking_evaluator", "save_distance_anomaly_report"),
    "YPValidation_Test": (".merged_ranking_evaluator", "YPValidation_Test"),
    "MergedRankingEvaluator": (".merged_ranking_evaluator", "MergedRankingEvaluator"),
    "YOLOPoseImage": ("..datasets.yolo_pose_dataset", "YOLOPoseImage"),
    "YOLOPoseDataset": ("..datasets.yolo_pose_dataset", "YOLOPoseDataset"),
    "YPImageValidation": (".image_validation", "YPImageValidation"),
    "YPSetValidation": (".set_validation", "YPSetValidation"),
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
    # report parsing and comparison
    "parse_results_txt",
    "parse_distance_anomaly_report",
    "compare_results_txt_to_distance_anomalies",
    "compare_results_txt_to_distance_anomalies_with_outside_keypoints",
    "save_distance_anomaly_report",
    # dataset classes
    "YOLOPoseImage",
    "YOLOPoseDataset",
    # validation classes
    "YPImageValidation",
    "YPSetValidation",
    "YPValidation_Test",
    "MergedRankingEvaluator",
]
