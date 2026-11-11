"""Smoke tests for validation utilities.

Run directly:
    python test_validation_smoke.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from pose_data_validation.validation import (
    MergedRankingEvaluator,
    YPImageValidation,
    YPSetValidation,
    YPValidation_Test,
    compare_results_txt_to_distance_anomalies,
    parse_results_txt,
    save_distance_anomaly_report,
)


NOTEBOOKS_DIR = Path(__file__).resolve().parent / "scores_from_publication"
SCORES_NOTEBOOKS = [
    "Dataset_axis_out_evaluation.ipynb",
    "Dataset_preserving_angles_evaluation.ipynb",
    "Dataset_rigid_out_evaluation.ipynb",
    "Detected_not_detected_comp.ipynb",
    "Dirty_clean_models_evaluation.ipynb",
    "Distances_ML_models_benchmark.ipynb",
    "LBP_analysis.ipynb",
    "Preparing_datasets_for_training.ipynb",
    "Random_reference.ipynb",
    "Segmentation_models_benchmark.ipynb",
    "Training_dirty_and_clean_models.ipynb",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _iter_notebook_code_lines(notebook_path: Path):
    with notebook_path.open("r", encoding="utf-8") as file_handle:
        notebook = json.load(file_handle)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, str):
            for line in source.splitlines():
                yield line
        else:
            for line in source:
                yield str(line)


def _parse_imported_symbols(line: str) -> list[str]:
    match = re.search(r"^\s*from\s+pose_data_validation\.validation\s+import\s+(.+)$", line.strip())
    if not match:
        return []

    imported = []
    for item in match.group(1).split(","):
        token = item.strip()
        if not token:
            continue
        if " as " in token:
            token = token.split(" as ", 1)[0].strip()
        imported.append(token)
    return imported


def test_image_validation_core() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    keypoints = np.array(
        [
            [0.1, 0.1, 2.0],
            [0.8, 0.1, 2.0],
            [0.8, 0.8, 1.0],
            [0.1, 0.8, 2.0],
        ],
        dtype=float,
    )

    validator = YPImageValidation(image, keypoints)

    distances_default = validator.sequential_distances(visibility_threshold=2.0)
    _assert(distances_default.dtype.names == ("distance_index", "distance"), "Invalid distances dtype")
    _assert(len(distances_default) == 1, "Expected only one visible consecutive segment")

    distances_custom = validator.sequential_distances(
        visibility_threshold=2.0,
        distance_connections=[(0, 1), (1, 3)],
    )
    _assert(len(distances_custom) == 2, "Expected two custom visible distances")

    angles = validator.sequential_angles(visibility_threshold=2.0)
    _assert(angles.dtype.names == ("angle_index", "angle"), "Invalid angles dtype")

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:9, 1:9] = 255
    metrics = validator.evaluate_keypoints_against_mask(mask=mask, visibility_threshold=2.0)
    _assert(metrics.dtype.names == ("keypoint_index", "is_visible", "inside_mask", "left_distance", "right_distance"), "Invalid mask metrics dtype")


def test_keypoint_loader() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        label_path = Path(tmp_dir) / "sample.txt"
        # class, bbox(x,y,w,h), then 2 keypoints in xyv format.
        label_path.write_text("0 0.5 0.5 1.0 1.0 0.1 0.2 2 0.3 0.4 1\n", encoding="utf-8")

        first_person = YPImageValidation.load_keypoints(str(label_path), include_visibility=True)
        _assert(first_person.shape == (2, 3), "Expected (2, 3) keypoint array")

        all_people = YPImageValidation.load_all_keypoints_from_file(str(label_path), include_visibility=False)
        _assert(len(all_people) == 1 and all_people[0].shape == (2, 2), "Expected one person with xy keypoints")


class _TinyDataset:
    def __init__(self, samples: list[dict]):
        self._samples = samples
        self.dataset_dir = os.getcwd()
        self.dataset_file = os.path.join(self.dataset_dir, "train.txt")
        self._valid_pairs = [(s["image_path"], "") for s in samples]
        self.stats = {
            "total_images": len(samples),
            "total_keypoints": int(sum(np.asarray(s["keypoints"]).shape[0] for s in samples)),
            "avg_keypoints_per_image": 0 if not samples else float(sum(np.asarray(s["keypoints"]).shape[0] for s in samples) / len(samples)),
            "dataset_file": self.dataset_file,
            "labels_dir": self.dataset_dir,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)


class _EvaluatorStubValidator:
    def __init__(self, dataset: _TinyDataset):
        self.dataset = dataset

    @staticmethod
    def _get_sample_person_keypoints(sample):
        keypoints = np.asarray(sample["keypoints"], dtype=float)
        return [keypoints]

    def train_distance_models(self, *args, **kwargs):
        return [object()], np.array([0], dtype=np.int64)

    def predict_distance_anomalies(self, *args, **kwargs):
        records = []
        for idx, sample in enumerate(self.dataset):
            records.append(
                {
                    "image_path": sample["image_path"],
                    "person_id": 0,
                    "rank": idx + 1,
                    "error": float(idx + 1),
                    "sort_score": float(idx + 1),
                    "is_anomaly": True,
                }
            )
        return {"images": records}

    def predict_lbp_anomalies_by_embedding_groups(self, *args, **kwargs):
        records = []
        for idx, sample in enumerate(self.dataset):
            records.append(
                {
                    "image_path": sample["image_path"],
                    "person_id": 0,
                    "rank": idx + 1,
                    "error": float(idx + 1),
                    "sort_score": float(idx + 1),
                    "is_anomaly": True,
                }
            )
        return {"images": records}


def test_set_validation_mask_pipeline() -> None:
    image = np.zeros((12, 12), dtype=np.uint8)
    samples = [
        {
            "image_path": "img_a.jpg",
            "image": image,
            "keypoints": np.array([[0.2, 0.2, 2.0], [0.4, 0.2, 2.0], [0.6, 0.2, 2.0]], dtype=float),
        },
        {
            "image_path": "img_b.jpg",
            "image": image,
            "keypoints": np.array([[0.2, 0.7, 2.0], [0.4, 0.7, 2.0], [0.6, 0.7, 2.0]], dtype=float),
        },
    ]
    dataset = _TinyDataset(samples)
    validator = YPSetValidation(dataset)

    masks = [np.ones((12, 12), dtype=np.uint8) * 255 for _ in samples]
    eval_results = validator.evaluate_dataset_keypoints_against_masks(masks, verbose=False)
    _assert(len(eval_results["images"]) == 2, "Expected 2 evaluated images")

    mask_distance_results = validator.build_keypoint_mask_distance_datasets(
        eval_results,
        variant=0,
        drop_nan=True,
        contamination=0.5,
        n_estimators=10,
    )
    _assert("records" in mask_distance_results, "Missing records in mask distance results")


def test_results_parsing_and_comparison() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        reference_path = os.path.join(tmp_dir, "results.txt")
        report_path = os.path.join(tmp_dir, "distance_report.txt")

        with open(reference_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("{'image_path': 'img_a.jpg', 'person_id': 0} | score: 1.0\n")
            file_handle.write("{'image_path': 'img_b.jpg', 'person_id': 0} | score: 0.5\n")

        parsed = parse_results_txt(reference_path)
        _assert(len(parsed) == 2, "Expected two parsed reference rows")

        anomaly_results = {
            "images": [
                {"rank": 1, "image_path": "img_a.jpg", "person_id": 0, "error": 2.0, "is_anomaly": True},
                {"rank": 2, "image_path": "img_c.jpg", "person_id": 0, "error": 1.0, "is_anomaly": False},
            ]
        }
        save_distance_anomaly_report(anomaly_results, report_path)

        comparison = compare_results_txt_to_distance_anomalies(reference_path, report_path, top_k=2)
        _assert(comparison["matched_top_k_count"] == 1, "Expected one top-k match")


def test_merged_ranking_evaluator_run() -> None:
    image = np.zeros((8, 8), dtype=np.uint8)
    samples = [
        {
            "image_path": "img_a.jpg",
            "image": image,
            "keypoints": np.array([[0.2, 0.2, 2.0], [0.7, 0.2, 2.0]], dtype=float),
        },
        {
            "image_path": "img_b.jpg",
            "image": image,
            "keypoints": np.array([[0.2, 0.7, 2.0], [0.7, 0.7, 2.0]], dtype=float),
        },
    ]
    dataset = _TinyDataset(samples)
    stub_validator = _EvaluatorStubValidator(dataset)

    with tempfile.TemporaryDirectory() as tmp_dir:
        reference_path = os.path.join(tmp_dir, "reference.txt")
        with open(reference_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("{'image_path': 'img_a.jpg', 'person_id': 0} | score: 1.0\n")
            file_handle.write("{'image_path': 'img_b.jpg', 'person_id': 0} | score: 0.8\n")

        evaluator = MergedRankingEvaluator(
            validator=stub_validator,
            report_paths={"merged": reference_path},
            top_k=10,
            segmentation_model_name=None,
        )
        result = evaluator.run(1.0, 1.0, 0.0)

        _assert(isinstance(result, dict), "Evaluator should return a dictionary")
        _assert(len(result.get("combined_ranking", [])) == 2, "Expected ranking for both samples")


def test_scores_notebooks_validation_api_contract() -> None:
    _assert(NOTEBOOKS_DIR.exists(), f"Missing notebooks directory: {NOTEBOOKS_DIR}")

    missing_notebooks = [name for name in SCORES_NOTEBOOKS if not (NOTEBOOKS_DIR / name).exists()]
    _assert(not missing_notebooks, f"Missing notebooks: {missing_notebooks}")

    exported_symbols = set(__import__("pose_data_validation.validation", fromlist=["__all__"]).__all__)

    expected_yp_set_methods = {
        "train_distance_models",
        "predict_distance_anomalies",
        "predict_lbp_anomalies_by_embedding_groups",
        "predict_lbp_anomalies_by_embedding_groups_zscore",
        "predict_lbp_anomalies_by_embedding_groups_ocsvm",
        "model_in_the_loop_validation",
        "segment_dataset_with_model",
        "evaluate_dataset_keypoints_against_masks",
        "build_keypoint_mask_distance_datasets",
        "collect_visible_keypoints_outside_masks",
    }
    for method_name in expected_yp_set_methods:
        _assert(
            hasattr(YPSetValidation, method_name),
            f"YPSetValidation is missing method required by notebooks: {method_name}",
        )

    _assert(hasattr(MergedRankingEvaluator, "run"), "MergedRankingEvaluator must expose run()")
    _assert(hasattr(YPValidation_Test, "parse_results_txt"), "YPValidation_Test must expose parse_results_txt()")

    for notebook_name in SCORES_NOTEBOOKS:
        notebook_path = NOTEBOOKS_DIR / notebook_name
        imported_from_validation: set[str] = set()
        all_lines = list(_iter_notebook_code_lines(notebook_path))

        for line in all_lines:
            _assert(
                "pose_data_validation.validation.weryfikacje" not in line,
                f"Notebook {notebook_name} uses removed path: {line.strip()}",
            )

            for symbol in _parse_imported_symbols(line):
                imported_from_validation.add(symbol)

        for symbol in imported_from_validation:
            _assert(
                symbol in exported_symbols,
                f"Notebook {notebook_name} imports unknown symbol from pose_data_validation.validation: {symbol}",
            )

        joined_source = "\n".join(all_lines)
        if "MergedRankingEvaluator(" in joined_source:
            _assert("MergedRankingEvaluator" in imported_from_validation, f"Notebook {notebook_name} uses MergedRankingEvaluator without importing it")
        if "YPSetValidation(" in joined_source:
            _assert("YPSetValidation" in imported_from_validation, f"Notebook {notebook_name} uses YPSetValidation without importing it")
        if "YPValidation_Test" in joined_source:
            _assert("YPValidation_Test" in imported_from_validation, f"Notebook {notebook_name} uses YPValidation_Test without importing it")


def run_all_smoke_tests() -> None:
    test_image_validation_core()
    test_keypoint_loader()
    test_set_validation_mask_pipeline()
    test_results_parsing_and_comparison()
    test_merged_ranking_evaluator_run()
    test_scores_notebooks_validation_api_contract()
    print("All smoke tests passed.")


if __name__ == "__main__":
    run_all_smoke_tests()
