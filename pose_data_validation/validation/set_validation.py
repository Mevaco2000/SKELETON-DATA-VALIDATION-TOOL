"""Dataset-level validation methods: LBP features and duplicate detection."""

import os
import math
import shutil
import tempfile
import warnings
from typing import List, Tuple, Optional, Union, Callable, Dict, Any, Sequence
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
import faiss
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
    RANSACRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    IsolationForest,
)
from sklearn.svm import SVR, OneClassSVM
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from .image_validation import YPImageValidation, labelled_keypoint_mask, normalize_distance_connections
from ..config import (
    CLIP_MODEL_NAME,
    CLIP_MODEL_PRETRAINED,
    DEFAULT_K_NEIGHBORS,
)


# Mapping of string keys to sklearn regressor constructors.
# train_distance_models filters invalid rows per target, so regressors do not
# need native NaN support here.
DEFAULT_REGRESSORS = {
    'linear': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'elasticnet': ElasticNet,
    'bayesian_ridge': BayesianRidge,
    'huber': HuberRegressor,
    'ransac': RANSACRegressor,
    'svr': SVR,
    'decision_tree': lambda: DecisionTreeRegressor(random_state=42),
    'random_forest': RandomForestRegressor,
    'gradient_boosting': lambda: GradientBoostingRegressor(random_state=42),
    'hist_gradient_boosting': HistGradientBoostingRegressor,
    'knn': KNeighborsRegressor,
    'mlp': lambda: MLPRegressor(max_iter=1000, random_state=42),
}


# The tuples below must list every class in the model's label order, because
# the position in the tuple is the class id the model outputs. Do not trim them.

# torchvision Mask R-CNN (COCO_V1 weights): 91 ids, including "N/A" gaps.
COCO_DETECTION_CATEGORIES = (
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

# torchvision DeepLabV3 / LRASPP / FCN (COCO_WITH_VOC_LABELS_V1 weights): 21 ids.
VOC_SEGMENTATION_CLASSES = (
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

# Ultralytics YOLO COCO models: 80 contiguous ids.
COCO_YOLO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

assert len(COCO_DETECTION_CATEGORIES) == 91
assert len(VOC_SEGMENTATION_CLASSES) == 21
assert len(COCO_YOLO_CLASSES) == 80


def _normalize_class_name(name: str) -> str:
    return "".join(char for char in str(name).lower() if char.isalnum())


def _build_class_lookup(class_names: Sequence[str]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx, class_name in enumerate(class_names):
        if class_name in {"N/A", "__background__"}:
            continue
        lookup[_normalize_class_name(class_name)] = idx
    return lookup


COCO_DETECTION_CLASS_LOOKUP = _build_class_lookup(COCO_DETECTION_CATEGORIES)
VOC_SEGMENTATION_CLASS_LOOKUP = _build_class_lookup(VOC_SEGMENTATION_CLASSES)
COCO_YOLO_CLASS_LOOKUP = _build_class_lookup(COCO_YOLO_CLASSES)

_CLIP_MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}


# ==================== CLIP-based Embedding Functions ====================

def _load_clip_model(device: str) -> Tuple:
    """Load CLIP model for image embedding.
    
    Args:
        device: Device to load model on ("cuda" or "cpu")
        
    Returns:
        Tuple of (model, preprocess) for CLIP
    """
    model_name = CLIP_MODEL_NAME
    pretrained_tag = CLIP_MODEL_PRETRAINED

    # OpenAI checkpoints are trained with QuickGELU; prefer the matching arch
    # to avoid mismatch warnings and extra compatibility overhead.
    candidate_model_names = [model_name]
    if pretrained_tag == "openai" and not model_name.endswith("-quickgelu"):
        candidate_model_names = [f"{model_name}-quickgelu", model_name]

    for candidate_name in candidate_model_names:
        cache_key = (device, candidate_name, pretrained_tag)
        cached = _CLIP_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*QuickGELU mismatch.*",
                    category=UserWarning,
                )
                model, _, preprocess = open_clip.create_model_and_transforms(
                    candidate_name,
                    pretrained=pretrained_tag,
                )
            model = model.to(device)
            model.eval()
            _CLIP_MODEL_CACHE[cache_key] = (model, preprocess)
            return model, preprocess
        except Exception:
            continue

    # Keep a clear error if both preferred and fallback variants fail.
    raise RuntimeError(
        f"Failed to load OpenCLIP model '{CLIP_MODEL_NAME}' with pretrained='{CLIP_MODEL_PRETRAINED}'."
    )


 


class YPSetValidation:
    
    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def _load_yolo_pose_model(model):
        """Return an Ultralytics YOLO model from an existing object or weights path."""
        if hasattr(model, "train") and hasattr(model, "predict"):
            resolved_model = model
            overrides = getattr(resolved_model, "overrides", None)
            if isinstance(overrides, dict) and "model" not in overrides:
                fallback_model = getattr(resolved_model, "ckpt_path", None) or getattr(resolved_model, "model_name", None)
                if fallback_model:
                    overrides["model"] = fallback_model
            return resolved_model

        from ultralytics import YOLO

        resolved_model = YOLO(model)
        overrides = getattr(resolved_model, "overrides", None)
        if isinstance(overrides, dict) and "model" not in overrides:
            overrides["model"] = str(model)
        return resolved_model

    @staticmethod
    def _ensure_ultralytics_model_override(model_obj, model_hint: Optional[Any] = None) -> None:
        """Populate model overrides['model'] when Ultralytics drops it between train calls."""
        overrides = getattr(model_obj, "overrides", None)
        if not isinstance(overrides, dict):
            return
        if "model" in overrides and overrides.get("model"):
            return

        fallback = (
            getattr(model_obj, "ckpt_path", None)
            or getattr(model_obj, "model_name", None)
            or model_hint
        )
        if fallback:
            overrides["model"] = str(fallback)

    @staticmethod
    def _get_result_keypoints_xy_normalized(result, image_shape: Tuple[int, ...]) -> List[np.ndarray]:
        """Extract predicted keypoint coordinates from an Ultralytics result in normalized image coordinates."""
        if getattr(result, "keypoints", None) is None:
            return []

        keypoints = result.keypoints
        if getattr(keypoints, "xyn", None) is not None:
            predicted = keypoints.xyn.detach().cpu().numpy()
        elif getattr(keypoints, "xy", None) is not None:
            predicted = keypoints.xy.detach().cpu().numpy()
            height, width = image_shape[:2]
            scale = np.asarray([max(float(width), 1.0), max(float(height), 1.0)], dtype=float)
            predicted = predicted / scale
        else:
            return []

        if predicted.ndim == 2:
            predicted = predicted[np.newaxis, :, :]
        return [np.asarray(person[:, :2], dtype=float) for person in predicted if np.asarray(person).size > 0]

    @staticmethod
    def _keypoint_prediction_error(
        actual_keypoints: np.ndarray,
        predicted_keypoints: Optional[np.ndarray],
        visibility_threshold: Optional[float] = 0.0,
    ) -> Tuple[float, np.ndarray, int]:
        """Compute summed Euclidean keypoint error for one actual/predicted person pair."""
        actual = np.asarray(actual_keypoints, dtype=float)
        if actual.size == 0:
            return math.inf, np.array([], dtype=float), 0

        if actual.ndim != 2 or actual.shape[1] < 2:
            return math.inf, np.array([], dtype=float), 0

        # Unlabelled keypoints have no ground-truth position, so they never
        # count, whatever visibility_threshold is.
        visible_mask = labelled_keypoint_mask(actual)
        if visibility_threshold is not None and actual.shape[1] >= 3:
            visible_mask &= actual[:, 2] >= float(visibility_threshold)

        if predicted_keypoints is None:
            per_keypoint_errors = np.full(actual.shape[0], np.nan, dtype=float)
            per_keypoint_errors[visible_mask] = 1.0
            return float(np.nansum(per_keypoint_errors)), per_keypoint_errors, int(np.sum(visible_mask))

        predicted = np.asarray(predicted_keypoints, dtype=float)
        compared_count = min(actual.shape[0], predicted.shape[0])
        per_keypoint_errors = np.full(actual.shape[0], np.nan, dtype=float)

        if compared_count > 0:
            compared_mask = visible_mask[:compared_count]
            errors = np.linalg.norm(
                actual[:compared_count, :2] - predicted[:compared_count, :2],
                axis=1,
            )
            per_keypoint_errors[:compared_count] = np.where(compared_mask, errors, np.nan)

        if actual.shape[0] > compared_count:
            per_keypoint_errors[compared_count:][visible_mask[compared_count:]] = 1.0

        finite_errors = per_keypoint_errors[np.isfinite(per_keypoint_errors)]
        return float(np.sum(finite_errors)) if finite_errors.size else math.inf, per_keypoint_errors, int(finite_errors.size)

    @staticmethod
    def _match_predictions_to_actual_persons(
        actual_person_keypoints: Sequence[np.ndarray],
        predicted_person_keypoints: Sequence[np.ndarray],
        visibility_threshold: Optional[float] = 0.0,
    ) -> List[Tuple[Optional[int], float, np.ndarray, int]]:
        """Greedily match predicted persons to actual persons by lowest keypoint error."""
        unmatched_predictions = set(range(len(predicted_person_keypoints)))
        matches: List[Tuple[Optional[int], float, np.ndarray, int]] = []

        for actual_keypoints in actual_person_keypoints:
            best_prediction_idx = None
            best_error = math.inf
            best_per_keypoint_errors = np.array([], dtype=float)
            best_valid_count = 0

            for prediction_idx in list(unmatched_predictions):
                error, per_keypoint_errors, valid_count = YPSetValidation._keypoint_prediction_error(
                    actual_keypoints,
                    predicted_person_keypoints[prediction_idx],
                    visibility_threshold=visibility_threshold,
                )
                if error < best_error:
                    best_prediction_idx = prediction_idx
                    best_error = error
                    best_per_keypoint_errors = per_keypoint_errors
                    best_valid_count = valid_count

            if best_prediction_idx is not None:
                unmatched_predictions.remove(best_prediction_idx)
                matches.append((best_prediction_idx, best_error, best_per_keypoint_errors, best_valid_count))
            else:
                error, per_keypoint_errors, valid_count = YPSetValidation._keypoint_prediction_error(
                    actual_keypoints,
                    None,
                    visibility_threshold=visibility_threshold,
                )
                matches.append((None, error, per_keypoint_errors, valid_count))

        return matches

    @staticmethod
    def _write_split_file_from_indices(dataset, indices: Sequence[int], output_path: str) -> str:
        with open(output_path, "w", encoding="utf-8") as file_handle:
            for index in indices:
                image_path = dataset._valid_pairs[int(index)][0]
                # Use absolute paths to avoid CWD-dependent failures in Ultralytics data loading.
                file_handle.write(os.path.abspath(image_path) + "\n")
        return output_path

    @staticmethod
    def _ensure_ultralytics_label_layout(dataset, image_paths: Optional[Sequence[str]] = None) -> None:
        """Create labels/train compatibility links when labels are stored flat in labels/."""
        dataset_dir = os.path.abspath(getattr(dataset, "dataset_dir", os.path.dirname(getattr(dataset, "dataset_file", ""))))
        images_train_dir = os.path.join(dataset_dir, "images", "train")
        labels_root_dir = os.path.join(dataset_dir, "labels")
        labels_train_dir = os.path.join(labels_root_dir, "train")

        if not os.path.isdir(images_train_dir) or not os.path.isdir(labels_root_dir):
            return

        if os.path.isdir(labels_train_dir):
            try:
                if any(os.scandir(labels_train_dir)):
                    return
            except OSError:
                return
        else:
            os.makedirs(labels_train_dir, exist_ok=True)

        resolved_image_paths = image_paths
        if resolved_image_paths is None:
            resolved_image_paths = [pair[0] for pair in getattr(dataset, "_valid_pairs", [])]

        for image_path in resolved_image_paths:
            label_name = os.path.splitext(os.path.basename(str(image_path)))[0] + ".txt"
            source_label = os.path.join(labels_root_dir, label_name)
            target_label = os.path.join(labels_train_dir, label_name)
            if not os.path.isfile(source_label) or os.path.exists(target_label):
                continue
            try:
                os.link(source_label, target_label)
            except OSError:
                shutil.copy2(source_label, target_label)

    @staticmethod
    def _find_dataset_yaml(dataset) -> Optional[str]:
        candidate_dirs = [
            os.path.dirname(os.path.abspath(getattr(dataset, "dataset_file", ""))),
            os.path.abspath(getattr(dataset, "dataset_dir", "")),
        ]
        for candidate_dir in candidate_dirs:
            if not candidate_dir:
                continue
            for filename in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
                candidate_path = os.path.join(candidate_dir, filename)
                if os.path.exists(candidate_path):
                    return candidate_path
        return None

    @staticmethod
    def _build_yolo_training_yaml(dataset, train_file: str, output_yaml: str, base_data_yaml: Optional[str] = None) -> str:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to build temporary YOLO data YAML files. Install: pip install pyyaml")

        data_config: Dict[str, Any] = {}
        resolved_base_yaml = base_data_yaml or YPSetValidation._find_dataset_yaml(dataset)
        if resolved_base_yaml is not None and os.path.exists(resolved_base_yaml):
            with open(resolved_base_yaml, "r", encoding="utf-8") as file_handle:
                loaded_config = yaml.safe_load(file_handle) or {}
            if isinstance(loaded_config, dict):
                data_config.update(loaded_config)

        data_config.setdefault("path", os.path.abspath(getattr(dataset, "dataset_dir", os.path.dirname(train_file))))
        data_config["train"] = os.path.abspath(train_file)
        data_config.setdefault("val", os.path.abspath(train_file))

        with open(output_yaml, "w", encoding="utf-8") as file_handle:
            yaml.safe_dump(data_config, file_handle, sort_keys=False, allow_unicode=True)
        return output_yaml

    def _rank_model_keypoint_errors(
        self,
        model,
        visibility_threshold: Optional[float] = 0.0,
        predict_kwargs: Optional[Dict[str, Any]] = None,
        stage: str = "model_prediction",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Predict the dataset and rank person annotations by summed keypoint error."""
        resolved_predict_kwargs = dict(predict_kwargs or {})
        records: List[Dict[str, Any]] = []
        total = len(self.dataset)
        iterator = self.dataset
        if verbose:
            iterator = tqdm(iterator, total=total, desc=f"Predicting keypoints ({stage})")

        for sample_index, sample in enumerate(iterator):
            image_path = sample.get('image_path')
            image = sample.get('image')
            actual_person_keypoints = self._get_sample_person_keypoints(sample)

            try:
                prediction_args = {
                    "source": image_path,
                    "verbose": False,
                }
                prediction_args.update(resolved_predict_kwargs)
                result = model.predict(**prediction_args)[0]
                predicted_person_keypoints = self._get_result_keypoints_xy_normalized(result, image.shape)
            except Exception as exc:
                if verbose:
                    print(f"Prediction failed for {image_path}: {exc}")
                predicted_person_keypoints = []

            matched_predictions = self._match_predictions_to_actual_persons(
                actual_person_keypoints,
                predicted_person_keypoints,
                visibility_threshold=visibility_threshold,
            )

            for person_index, (prediction_index, error, per_keypoint_errors, valid_count) in enumerate(matched_predictions):
                records.append({
                    "image_path": image_path,
                    "person_id": int(person_index),
                    "person_index": int(person_index),
                    "sample_index": int(sample_index),
                    "prediction_index": None if prediction_index is None else int(prediction_index),
                    "error": float(error),
                    "sort_score": float(error),
                    "keypoint_error_sum": float(error),
                    "per_keypoint_errors": per_keypoint_errors,
                    "valid_keypoint_count": int(valid_count),
                    "stage": stage,
                })

        records.sort(key=lambda record: record["sort_score"], reverse=True)
        for rank, record in enumerate(records, start=1):
            record["rank"] = int(rank)

        return {
            "images": records,
            "records": records,
            "combined_ranking": records,
            "sort_by": "keypoint_error_sum_desc",
            "stage": stage,
            "statistics": {
                "total_images": int(total),
                "total_records": int(len(records)),
                "finite_error_records": int(sum(np.isfinite(record["error"]) for record in records)),
            },
        }

    def model_in_the_loop_validation(
        self,
        model,
        initial_epochs: int,
        remove_percent: float,
        fine_tune_epochs: int,
        data_yaml: Optional[str] = None,
        work_dir: Optional[str] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
        fine_tune_train_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
        visibility_threshold: Optional[float] = 0.0,
        lightweight_return: bool = False,
        lightweight_top_k: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run model-in-the-loop validation for YOLO pose training.

        Workflow:
        1. Train an Ultralytics YOLO pose model for ``initial_epochs``.
        2. Predict the full training set and rank annotations by summed keypoint error.
        3. Remove ``remove_percent`` percent of the highest-error images from the training split.
        4. Continue training the same model object for ``fine_tune_epochs`` on the reduced split.
        5. Predict the original full set again and return a final ranking compatible with
           ``MergedRankingEvaluator``-style consumers.

        Args:
            model: Existing Ultralytics YOLO model object or path to weights.
            initial_epochs: Number of epochs for the first training stage.
            remove_percent: Percent of training images to exclude before the second stage.
            fine_tune_epochs: Number of epochs for continued training on the reduced split.
            data_yaml: Optional source dataset YAML. When omitted, ``data.yaml`` is searched
                near the dataset split and copied with an overridden train split.
            work_dir: Directory for temporary split/YAML files. Defaults to a temp directory.
            train_kwargs: Extra keyword arguments passed to the first ``model.train`` call.
            fine_tune_train_kwargs: Extra keyword arguments for the second ``model.train`` call.
                Values override ``train_kwargs``.
            predict_kwargs: Extra keyword arguments passed to ``model.predict``.
            visibility_threshold: Minimum ground-truth visibility used when summing errors.
            lightweight_return: When ``True``, return a compact payload without heavyweight
                objects such as full rankings and model instance.
            lightweight_top_k: Optional maximum number of ranked records kept in compact
                output. Ignored when ``lightweight_return`` is ``False``.
            verbose: Print progress information.

        Returns:
            Dictionary with ``combined_ranking`` and ``images`` containing the final sorted
            ranking. It also includes ``initial_ranking``, ``removed_records``,
            ``kept_train_file``, ``ranking_meta`` and ``statistics``.
        """
        if initial_epochs < 0 or fine_tune_epochs < 0:
            raise ValueError("Epoch counts must be non-negative")
        if remove_percent < 0 or remove_percent >= 100:
            raise ValueError("remove_percent must be in the range [0, 100)")
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")

        self._ensure_ultralytics_label_layout(self.dataset)

        resolved_model = self._load_yolo_pose_model(model)
        self._ensure_ultralytics_model_override(resolved_model, model_hint=model)
        base_train_kwargs = dict(train_kwargs or {})
        # MITL is usually run repeatedly in notebooks; disable validation/artifacts by default
        # to reduce RAM and disk pressure unless caller explicitly overrides these flags.
        base_train_kwargs.setdefault("val", False)
        base_train_kwargs.setdefault("plots", False)
        base_train_kwargs.setdefault("save", False)
        second_train_kwargs = dict(base_train_kwargs)
        second_train_kwargs.update(fine_tune_train_kwargs or {})

        resolved_work_dir = tempfile.mkdtemp(prefix="model_in_loop_") if work_dir is None else os.path.abspath(work_dir)
        os.makedirs(resolved_work_dir, exist_ok=True)

        full_train_file = os.path.join(resolved_work_dir, "train_full.txt")
        full_data_yaml = os.path.join(resolved_work_dir, "data_full.yaml")
        all_indices = list(range(len(self.dataset)))
        self._write_split_file_from_indices(self.dataset, all_indices, full_train_file)
        self._build_yolo_training_yaml(self.dataset, full_train_file, full_data_yaml, base_data_yaml=data_yaml)

        if verbose:
            print(f"Initial training: {initial_epochs} epochs on {len(all_indices)} images")
        if initial_epochs > 0:
            self._ensure_ultralytics_model_override(resolved_model, model_hint=model)
            resolved_model.train(data=full_data_yaml, epochs=int(initial_epochs), **base_train_kwargs)

        initial_ranking = self._rank_model_keypoint_errors(
            resolved_model,
            visibility_threshold=visibility_threshold,
            predict_kwargs=predict_kwargs,
            stage="after_initial_training",
            verbose=verbose,
        )

        remove_image_count = int(math.ceil(len(self.dataset) * (float(remove_percent) / 100.0)))
        removed_image_paths = []
        removed_image_path_set = set()
        removed_records = []
        for record in initial_ranking["images"]:
            if len(removed_image_paths) >= remove_image_count:
                break
            normalized_path = os.path.normcase(os.path.abspath(record["image_path"]))
            if normalized_path in removed_image_path_set:
                continue
            removed_image_path_set.add(normalized_path)
            removed_image_paths.append(record["image_path"])
            removed_records.append(record)

        kept_indices = [
            index for index, (image_path, _) in enumerate(self.dataset._valid_pairs)
            if os.path.normcase(os.path.abspath(image_path)) not in removed_image_path_set
        ]
        if not kept_indices:
            raise ValueError("remove_percent removed all training images; choose a smaller value")

        kept_train_file = os.path.join(resolved_work_dir, "train_reduced.txt")
        kept_data_yaml = os.path.join(resolved_work_dir, "data_reduced.yaml")
        self._write_split_file_from_indices(self.dataset, kept_indices, kept_train_file)
        self._build_yolo_training_yaml(self.dataset, kept_train_file, kept_data_yaml, base_data_yaml=data_yaml)

        if verbose:
            print(
                f"Continuing training: {fine_tune_epochs} epochs on {len(kept_indices)} images "
                f"({len(removed_image_paths)} removed)"
            )
        if fine_tune_epochs > 0:
            self._ensure_ultralytics_model_override(resolved_model, model_hint=model)
            resolved_model.train(data=kept_data_yaml, epochs=int(fine_tune_epochs), **second_train_kwargs)

        final_ranking = self._rank_model_keypoint_errors(
            resolved_model,
            visibility_threshold=visibility_threshold,
            predict_kwargs=predict_kwargs,
            stage="after_reduced_training",
            verbose=verbose,
        )

        ranking_meta = {
            "initial_epochs": int(initial_epochs),
            "fine_tune_epochs": int(fine_tune_epochs),
            "remove_percent": float(remove_percent),
            "initial_train_image_count": int(len(all_indices)),
            "reduced_train_image_count": int(len(kept_indices)),
            "removed_image_count": int(len(removed_image_paths)),
            "removed_record_count": int(len(removed_records)),
        }

        if lightweight_return:
            compact_records = []
            for record in final_ranking["records"]:
                compact_records.append({
                    "rank": int(record.get("rank", 0)),
                    "image_path": record.get("image_path"),
                    "person_id": int(record.get("person_id", 0)),
                    "error": float(record.get("error", math.inf)),
                })
            if isinstance(lightweight_top_k, int) and lightweight_top_k > 0:
                compact_records = compact_records[:lightweight_top_k]

            return {
                "weights": (1.0,),
                "output_path": None,
                "ranking_meta": ranking_meta,
                "combined_ranking": compact_records,
                "statistics": {
                    **ranking_meta,
                    "final_total_records": int(len(final_ranking["records"])),
                    "returned_records": int(len(compact_records)),
                    "work_dir": resolved_work_dir,
                    "lightweight_return": True,
                },
                "kept_train_file": kept_train_file,
                "kept_data_yaml": kept_data_yaml,
                "full_train_file": full_train_file,
                "full_data_yaml": full_data_yaml,
                "removed_image_paths": removed_image_paths,
            }

        return {
            "weights": (1.0,),
            "output_path": None,
            "ranking_meta": ranking_meta,
            "combined_ranking": final_ranking["combined_ranking"],
            "images": final_ranking["images"],
            "records": final_ranking["records"],
            "initial_ranking": initial_ranking,
            "final_ranking": final_ranking,
            "removed_records": removed_records,
            "removed_image_paths": removed_image_paths,
            "kept_train_file": kept_train_file,
            "kept_data_yaml": kept_data_yaml,
            "full_train_file": full_train_file,
            "full_data_yaml": full_data_yaml,
            "statistics": {
                **ranking_meta,
                "final_total_records": int(len(final_ranking["records"])),
                "work_dir": resolved_work_dir,
            },
            "model": resolved_model,
        }

    def _sample_loader_by_path(self) -> Callable[[str], Optional[Any]]:
        """Return a function that loads one dataset sample by image path, on demand.

        Loading one sample at a time keeps a single decoded image in memory
        instead of decoding every image of the dataset up front.
        """
        valid_pairs = getattr(self.dataset, "_valid_pairs", None)
        if valid_pairs is not None and hasattr(type(self.dataset), "__getitem__"):
            index_by_path = {
                os.path.abspath(image_path): index
                for index, (image_path, _) in enumerate(valid_pairs)
            }

            def load_sample(image_path: str):
                index = index_by_path.get(os.path.abspath(image_path))
                if index is None:
                    return None
                try:
                    sample = self.dataset[index]
                except Exception:
                    # Same as dataset iteration: skip a sample whose files became unreadable.
                    return None
                return sample if sample.get('image') is not None else None

            return load_sample

        # Datasets without random access: fall back to loading everything once.
        samples = {os.path.abspath(sample['image_path']): sample for sample in self.dataset}
        return lambda image_path: samples.get(os.path.abspath(image_path))

    @staticmethod
    def _get_sample_person_keypoints(sample) -> List[np.ndarray]:
        """Return a normalized list of per-person keypoint arrays for a sample."""
        all_keypoints = sample.get('all_keypoints') if hasattr(sample, 'get') else None
        if all_keypoints is not None:
            return [np.asarray(kpts) for kpts in all_keypoints if kpts is not None and np.asarray(kpts).size > 0]

        keypoints = sample['keypoints']
        if getattr(keypoints, 'ndim', 0) == 3:
            return [np.asarray(person_keypoints) for person_keypoints in keypoints if np.asarray(person_keypoints).size > 0]
        if keypoints is None or np.asarray(keypoints).size == 0:
            return []
        return [np.asarray(keypoints)]
    
    def get_embeddings(self, device: Optional[str] = None, image_folder: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Generate CLIP embeddings for all images in the dataset.
        
        Args:
            device: Device to use ("cuda" or "cpu"). Auto-detects if None.
            image_folder: Optional folder path to load images from. If provided, images are loaded 
                         from this folder using relative paths from the dataset. If None, uses 
                         pre-loaded images from dataset iterator.
            
        Returns:
            Tuple of (embeddings array, list of filenames)
            
        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> validator = YPSetValidation(dataset)
            >>> # Load from dataset's resolved paths:
            >>> embeddings, filenames = validator.get_embeddings()
            >>> # Or load from explicit folder:
            >>> embeddings, filenames = validator.get_embeddings(image_folder="path/to/images")
        """
        embeddings, image_paths = self._get_embeddings_with_image_paths(
            device=device,
            image_folder=image_folder,
            verbose=True,
        )
        filenames = [os.path.basename(path) for path in image_paths]
        return embeddings, filenames

    def _get_embeddings_with_image_paths(
        self,
        device: Optional[str] = None,
        image_folder: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate CLIP embeddings together with full dataset image paths."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, preprocess = _load_clip_model(device)

        embeddings = []
        image_paths = []
        iterator = self.dataset
        if verbose:
            iterator = tqdm(iterator, total=len(self.dataset), desc="Computing embeddings")

        with torch.no_grad():
            for sample in iterator:
                try:
                    if image_folder is not None:
                        rel_path = os.path.relpath(sample['image_path'], self.dataset.dataset_dir)
                        img_path = os.path.join(image_folder, rel_path)
                        image = cv2.imread(img_path)
                        if image is None:
                            continue
                    else:
                        image = sample['image']

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                    processed = preprocess(pil_image).unsqueeze(0).to(device)
                    embedding = model.encode_image(processed)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                    embeddings.append(embedding.cpu().numpy())
                    image_paths.append(sample['image_path'])
                except Exception:
                    continue

        if not embeddings:
            return np.zeros((0, 0), dtype=np.float32), []

        return np.vstack(embeddings).astype("float32"), image_paths
    
    
    def get_all_lbp_decimals(self, patch_size: int = 5, verbose: bool = True,
                             visibility_threshold: float = 2.0) -> tuple:
        """
        Compute binary decimal representations for all visible keypoints in entire dataset.
        
        Skips hidden/occluded keypoints based on visibility threshold.
        
        Args:
            patch_size: Size of patch for LBP computation
            verbose: Show progress bar
            visibility_threshold: Minimum visibility score (0-1) to process a keypoint.
                                Default 0.5. Keypoints below this are skipped.
            
        Returns:
            Tuple of (decimals, keypoints_per_image, image_paths):
            - decimals: Array (total_visible_keypoints,) of uint64
            - keypoints_per_image: List of visible keypoint counts per image
            - image_paths: List of image paths
        """
        decimal_list = []
        paths = []
        counts = []
        
        iterator = self.dataset
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self.dataset), 
                              desc=f"Computing binary decimals (patch_size={patch_size}, visibility_threshold={visibility_threshold})")
            except ImportError:
                pass
        
        for sample in iterator:
            image_decimals = []
            visible_count = 0

            for keypoints in self._get_sample_person_keypoints(sample):
                validator = YPImageValidation(sample['image'], keypoints)
                decimals = validator.compute_lbp_decimals(patch_size, visibility_threshold)
                if len(decimals) > 0:
                    image_decimals.append(decimals)
                    visible_count += len(decimals)
            
            if image_decimals:
                decimal_list.append(np.concatenate(image_decimals))
            paths.append(sample['image_path'])
            counts.append(visible_count)
        
        all_decimals = np.concatenate(decimal_list) if decimal_list else np.array([], dtype=np.uint64)
        return all_decimals, counts, paths
    
    
    def get_stats(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with:
            - total_images: Number of images
            - total_keypoints: Total keypoints across all images
            - avg_keypoints_per_image: Average keypoints per image
            - dataset_file: Path to dataset file
            - labels_dir: Path to labels directory
        """
        return self.dataset.stats
    
    def _predict_sample_segmentation(
        self,
        model,
        model_family: str,
        sample,
        target_class: int,
        score_threshold: float,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run segmentation for a single dataset sample and return masks and metadata."""
        image = sample['image']
        image_path = sample['image_path']
        height, width = image.shape[:2]

        instance_masks: List[np.ndarray] = []
        instance_scores: List[float] = []
        instance_boxes: List[np.ndarray] = []

        if model_family == "ultralytics_yolo_seg":
            predict_kwargs = {
                "source": image_path,
                "verbose": False,
                "conf": score_threshold,
            }
            if device is not None:
                predict_kwargs["device"] = device

            result = model.predict(**predict_kwargs)[0]

            if result.masks is not None and result.boxes is not None:
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                confidences = result.boxes.conf.detach().cpu().numpy()
                boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
                polygons = result.masks.xy

                for idx, polygon in enumerate(polygons):
                    if idx >= len(classes) or classes[idx] != target_class:
                        continue
                    if idx < len(confidences) and confidences[idx] < score_threshold:
                        continue
                    if len(polygon) < 3:
                        continue

                    binary = np.zeros((height, width), dtype=np.uint8)
                    points = np.round(polygon).astype(np.int32)
                    cv2.fillPoly(binary, [points], 255)
                    instance_masks.append(binary)
                    instance_scores.append(float(confidences[idx]))
                    instance_boxes.append(boxes_xyxy[idx])

        elif model_family == "torchvision_detection":
            import torchvision.transforms.functional as TF

            torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_t = TF.to_tensor(img_rgb).to(torch_device)

            with torch.no_grad():
                output = model([img_t])[0]

            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            boxes = output["boxes"].detach().cpu().numpy()
            masks = output["masks"].detach().cpu().numpy()

            for idx in range(len(labels)):
                if labels[idx] != target_class or scores[idx] < score_threshold:
                    continue
                binary = (masks[idx, 0] > 0.5).astype(np.uint8) * 255
                instance_masks.append(binary)
                instance_scores.append(float(scores[idx]))
                instance_boxes.append(boxes[idx])

        else:
            import torchvision.transforms.functional as TF

            torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_t = TF.to_tensor(img_rgb).to(torch_device)
            mean = torch.tensor([0.485, 0.456, 0.406], device=torch_device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=torch_device).view(3, 1, 1)

            with torch.no_grad():
                img_batch = ((img_t - mean) / std).unsqueeze(0)
                output = model(img_batch)["out"]

            pred_class = output.argmax(dim=1).squeeze(0).detach().cpu().numpy()
            binary = (pred_class == target_class).astype(np.uint8) * 255
            if np.any(binary):
                instance_masks.append(binary)
                instance_scores.append(1.0)
                instance_boxes.append(np.array([0, 0, width - 1, height - 1], dtype=float))

        if instance_masks:
            merged_mask = np.maximum.reduce(instance_masks)
        else:
            merged_mask = np.zeros((height, width), dtype=np.uint8)

        return {
            "mask": merged_mask,
            "instance_masks": instance_masks,
            "instance_scores": instance_scores,
            "instance_boxes": instance_boxes,
        }
    
    # ==================== Drawing & Visualization ====================

    
    def _get_distance_matrices(
        self,
        verbose: bool = True,
        visibility_threshold: float = 2.0,
        distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Extract distance matrices from all keypoint sequences in dataset.
        
        By default computes sequential distances between consecutive visible
        keypoints in each image. When ``distance_connections`` is provided,
        uses that explicit ordered list of keypoint pairs instead.
        
        Args:
            verbose: Show progress bar
            visibility_threshold: Minimum visibility score (0-1) to consider a keypoint visible.
                                Default 0.5. Keypoints below this are skipped.
            distance_connections: Optional ordered list of ``(start_idx, end_idx)``
                                keypoint pairs to measure instead of consecutive pairs.

        Returns:
            Tuple of (distance_matrix, image_paths, distance_indices):
            - distance_matrix: Array of shape ``(num_images, num_distance_types)``
              with measured distances
            - image_paths: List of image paths corresponding to each row
            - distance_indices: Array of distance column indices [0, 1, 2, ...]
        """
        distance_data = []  # List of (distances_struct, image_path) tuples
        paths = []
        normalized_connections = normalize_distance_connections(distance_connections)
        
        iterator = self.dataset
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(self.dataset), 
                              desc=f"Extracting sequential distances (visibility_threshold={visibility_threshold})")
            except ImportError:
                pass
        
        max_distance_index = -1
        for sample in iterator:
            image = sample['image']
            for kpts in self._get_sample_person_keypoints(sample):
                if len(kpts) < 2:
                    continue

                validator = YPImageValidation(image, kpts)
                distances_struct = validator.sequential_distances(
                    visibility_threshold=visibility_threshold,
                    distance_connections=normalized_connections,
                )

                if len(distances_struct) > 0:
                    max_idx_in_image = distances_struct['distance_index'].max()
                    max_distance_index = max(max_distance_index, max_idx_in_image)

                    distance_data.append((distances_struct, sample['image_path']))
                    paths.append(sample['image_path'])
        
        # Create matrix with columns for each distance index [0, 1, 2, ...]
        if distance_data:
            if normalized_connections is not None:
                num_cols = len(normalized_connections)
            else:
                num_cols = max_distance_index + 1
            distance_matrix = np.full((len(distance_data), num_cols), np.nan, dtype=np.float32)
            
            # Fill in distances at their absolute indices
            for row_idx, (distances_struct, _) in enumerate(distance_data):
                for dist_index, distance_value in zip(distances_struct['distance_index'], distances_struct['distance']):
                    distance_matrix[row_idx, dist_index] = distance_value
            
            distance_indices = np.arange(num_cols, dtype=np.int64)
        else:
            distance_matrix = np.zeros((0, 0), dtype=np.float32)
            distance_indices = np.array([], dtype=np.int64)
        
        return distance_matrix, paths, distance_indices

    def train_distance_models(
        self,
        model_factory: Union[Callable[[], BaseEstimator], str] = "hist_gradient_boosting",
        exclude_endpoints: bool = False,
        distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
        visibility_threshold: float = 2.0,
        verbose: bool = True,
    ) -> Tuple[List[BaseEstimator], np.ndarray]:
        """Train regression models to predict one distance from the others.
        
        Extracts distances from dataset and trains a separate regressor for each
        distance index. By default uses consecutive keypoint pairs; when
        ``distance_connections`` is provided, uses that explicit ordered list.
        Each model predicts one distance from all others.
        
        Args:
            model_factory: Either a callable returning a fresh estimator or a string
                key: linear, ridge, lasso, elasticnet, bayesian_ridge, huber, ransac,
                svr, decision_tree, random_forest, gradient_boosting, knn, mlp
            exclude_endpoints: If True, ignore first and last distances (for interior
                distances only). Defaults to False.
            distance_connections: Optional ordered list of ``(start_idx, end_idx)``
                keypoint pairs to measure instead of consecutive pairs.
            visibility_threshold: Minimum visibility score to consider a keypoint
                visible. Use 2.0 for YOLO-format fully-visible keypoints (default),
                or 1.0 to also include occluded keypoints.
            verbose: Show progress bar
            
        Returns:
            Tuple of (models, valid_distance_indices):
            - models: List of fitted sklearn estimators
            - valid_distance_indices: Array of original distance column indices that were kept
              (each model[i] predicts distance at valid_distance_indices[i])
        """
        if BaseEstimator is None:
            raise ImportError("scikit-learn is required for training distance models")

        # Get distance matrices (with NaN padding for uneven sequence lengths)
        distance_matrix, _, distance_indices = self._get_distance_matrices(
            verbose=verbose,
            visibility_threshold=visibility_threshold,
            distance_connections=distance_connections,
        )
        
        if distance_matrix.size == 0:
            raise ValueError("Dataset has no valid keypoint sequences")

        # resolve string names to constructors
        if isinstance(model_factory, str):
            key = model_factory.lower()
            if key not in DEFAULT_REGRESSORS:
                raise ValueError(f"unknown regressor '{model_factory}'; valid options are: {', '.join(sorted(DEFAULT_REGRESSORS.keys()))}")
            model_factory = DEFAULT_REGRESSORS[key]

        distance_matrix = np.asarray(distance_matrix)
        if distance_matrix.ndim != 2:
            raise ValueError(f"distance_matrix must be 2‑D, got {distance_matrix.ndim}‑D")

        num_samples, num_distances = distance_matrix.shape
        if num_samples < 2:
            raise ValueError("At least two samples are required to train models")

        # Columns with some NaN are kept: each model is trained only on the rows
        # where its target and inputs are valid. Only all-NaN columns are dropped.
        valid_distance_indices = distance_indices.copy()
        
        # Calculate how many valid (non-NaN) distances each row has
        num_valid_per_row = (~np.isnan(distance_matrix)).sum(axis=1)
        
        # Filter ROWS: keep only persons with at least 2 valid (non-NaN) distances ACROSS ALL COLUMNS
        rows_with_multiple_distances = num_valid_per_row >= 2
        distance_matrix_filtered = distance_matrix[rows_with_multiple_distances, :]
        
        if distance_matrix_filtered.shape[0] == 0:
            if verbose:
                print("   No persons have multiple valid distances. Returning empty models.")
            return [], np.array([], dtype=np.int64)
        
        # Remove columns that are entirely NaN (no valid data for training)
        non_nan_mask = ~np.isnan(distance_matrix_filtered).all(axis=0)
        distance_matrix_filtered = distance_matrix_filtered[:, non_nan_mask]
        valid_distance_indices = valid_distance_indices[non_nan_mask]
        
        # Need at least 2 distance COLUMNS (to predict one from others)
        if distance_matrix_filtered.shape[1] < 2:
            if verbose:
                print(f"  Only {distance_matrix_filtered.shape[1]} distance column(s) available after removing empty columns.")
                print(f"    Need at least 2 different distance types to train predictive models.")
                print(f"    Returning empty models.")
            return [], np.array([], dtype=np.int64)

        models: List[BaseEstimator] = []
        indices = list(range(distance_matrix_filtered.shape[1]))
        if exclude_endpoints:
            if distance_matrix_filtered.shape[1] < 3:
                raise ValueError(
                    f"Cannot exclude endpoints with only {distance_matrix_filtered.shape[1]} distances. "
                    f"Need at least 3 distances (first, middle, last)."
                )
            indices = list(range(1, distance_matrix_filtered.shape[1] - 1))

        # Train a model for each distance, using only rows with valid data for that model
        trained_distance_indices: List[int] = []
        all_positions = np.arange(distance_matrix_filtered.shape[1])
        for idx in indices:
            feature_positions = np.delete(all_positions, idx)
            X = distance_matrix_filtered[:, feature_positions]
            y = distance_matrix_filtered[:, idx]

            # Find rows where BOTH y and ALL features X are valid (not NaN)
            valid_y = ~np.isnan(y)
            valid_X = ~np.isnan(X).any(axis=1)
            valid_rows = valid_y & valid_X

            # Train on valid rows only
            if valid_rows.sum() < 2:
                if verbose:
                    print(f"  Skipping distance {valid_distance_indices[idx]}: < 2 valid samples")
                continue

            X_train = X[valid_rows, :]
            y_train = y[valid_rows]

            model = model_factory()
            model.fit(X_train, y_train)
            # Original distance columns used as inputs. Columns dropped above
            # (all NaN) are not among them, so predict_distance_anomalies must
            # build its inputs from this list, not from "all columns but y".
            model.feature_distance_indices_ = valid_distance_indices[feature_positions].copy()
            models.append(model)
            # Track the target of every model actually trained, so a skipped
            # distance does not shift the model -> distance mapping.
            trained_distance_indices.append(int(valid_distance_indices[idx]))

        if len(models) == 0:
            if verbose:
                print("  No models could be trained (insufficient valid data per distance type).")
            return [], np.array([], dtype=np.int64)

        return models, np.asarray(trained_distance_indices, dtype=np.int64)
    
    def predict_distance_anomalies(
        self,
        models: Sequence[BaseEstimator],
        valid_distance_indices: np.ndarray,
        exclude_endpoints: bool = False,
        distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
        visibility_threshold: float = 2.0,
        sort_by: str = "mean",
        threshold_percentile: float = 95.0,
        verbose: bool = True,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """Identify images with anomalous distances using trained models.
        
        Uses trained models to predict distances and compares with actual values.
        By default uses consecutive keypoint pairs; when ``distance_connections``
        is provided, uses that explicit ordered list. Images where predictions
        deviate significantly are flagged as anomalies.
        
        Args:
            models: List of fitted estimators from train_distance_models()
            valid_distance_indices: Array of distance indices returned from train_distance_models()
                (maps each model to its original distance column index)
            exclude_endpoints: Must match the flag used in train_distance_models()
            distance_connections: Optional ordered list of ``(start_idx, end_idx)``
                keypoint pairs. Must match the list used in training.
            visibility_threshold: Minimum visibility score to consider a keypoint
                visible. Must match the value used in train_distance_models().
            sort_by: Metric used for ranking and thresholding anomalies.
                Supported values: ``"mean"`` and ``"max"``.
            threshold_percentile: Percentile of error to flag as anomaly (0-100)
            verbose: Show progress bar
            
        Returns:
            Dict with:
            - 'images': List of dicts with per-image details:
                - image_path: Path to image
                - error: Backward-compatible alias for ``mean_error``
                - mean_error: Mean absolute error for this image
                - max_error: Maximum absolute error for this image
                - sort_score: Value actually used for ranking
                - is_anomaly: Boolean, True if ``sort_score`` exceeds threshold
                - distance_indices: Array of distance indices used for prediction
                - rank: Ranking by error (1 = worst)
                - per_distance_errors: Array of errors for each distance in this image
                - per_distance_ratios: Array of error/average_error ratios for each distance
            - 'distance_connections': Resolved list of connection pairs used to compute distances
            - 'sort_by': Metric used for ranking and thresholding
            - 'errors_per_distance': 2D array of per-distance errors (n_images × n_distances)
            - 'ratios_per_distance': 2D array of per-distance ratios (n_images × n_distances)
        """
        if BaseEstimator is None:
            raise ImportError("scikit-learn is required")

        normalized_connections = normalize_distance_connections(distance_connections)
        sort_key = sort_by.lower().strip()
        if sort_key not in {"mean", "max"}:
            raise ValueError("sort_by must be either 'mean' or 'max'")

        # Handle case where no models were trained (insufficient data)
        if len(models) == 0:
            if verbose:
                print("   No models available. Training returned empty models (insufficient distance data).")
            return {
                'images': [],
                'distance_connections': normalized_connections,
                'sort_by': sort_key,
                'errors_per_distance': np.array([]),
                'ratios_per_distance': np.array([])
            }

        distance_matrix, paths, _ = self._get_distance_matrices(
            verbose=verbose,
            visibility_threshold=visibility_threshold,
            distance_connections=normalized_connections,
        )
        
        if distance_matrix.size == 0:
            raise ValueError("Dataset has no valid keypoint sequences")

        # Columns with some NaN are kept; NaN is handled per model below.
        
        # Filter rows: keep only persons with at least 2 valid (non-NaN) distances
        num_valid_per_row = (~np.isnan(distance_matrix)).sum(axis=1)
        rows_with_enough_distances = num_valid_per_row >= 2
        distance_matrix_filtered = distance_matrix[rows_with_enough_distances, :]
        
        # Update paths to match filtered rows
        paths = [p for i, p in enumerate(paths) if rows_with_enough_distances[i]]
        
        if distance_matrix_filtered.shape[0] == 0:
            raise ValueError(
                f"No persons have at least 2 valid distances. "
                f"Cannot predict with no valid data."
            )

        if len(models) != len(valid_distance_indices):
            raise ValueError(
                f"number of models ({len(models)}) does not match expected number of distance indices ({len(valid_distance_indices)})"
            )

        num_columns = distance_matrix_filtered.shape[1]
        valid_distance_indices = np.asarray(valid_distance_indices, dtype=np.int64)
        if valid_distance_indices.size and int(valid_distance_indices.max()) >= num_columns:
            raise ValueError(
                f"Models predict distance {int(valid_distance_indices.max())}, but only {num_columns} "
                "distance columns were found. distance_connections and visibility_threshold must match training."
            )

        # Compute predictions using only valid rows for each model
        preds = []
        for model_idx, original_idx in enumerate(valid_distance_indices):
            original_idx = int(original_idx)
            feature_indices = getattr(models[model_idx], "feature_distance_indices_", None)
            if feature_indices is None:
                # No recorded input columns (model not built by train_distance_models):
                # use every column except the target.
                X = np.delete(distance_matrix_filtered, original_idx, axis=1)
            else:
                feature_indices = np.asarray(feature_indices, dtype=np.int64)
                if feature_indices.size and int(feature_indices.max()) >= num_columns:
                    raise ValueError(
                        f"Model for distance {original_idx} was trained on distance column "
                        f"{int(feature_indices.max())}, which is missing here. "
                        "distance_connections and visibility_threshold must match training."
                    )
                X = distance_matrix_filtered[:, feature_indices]
            y_actual = distance_matrix_filtered[:, original_idx]
            
            # Find rows where BOTH y and ALL features X are valid (not NaN)
            valid_y = ~np.isnan(y_actual)
            valid_X = ~np.isnan(X).any(axis=1)
            valid_rows = valid_y & valid_X
            
            # Make predictions only on valid rows
            y_pred = np.full(distance_matrix_filtered.shape[0], np.nan)
            if valid_rows.sum() > 0:
                y_pred[valid_rows] = models[model_idx].predict(X[valid_rows, :])
            
            preds.append(y_pred)
        
        predictions = np.column_stack(preds)
        
        # For each person, compute per-distance errors (keep NaN for invalid)
        actual = distance_matrix_filtered[:, valid_distance_indices]
        
        # Compute absolute errors for each distance per person
        errors_per_distance = np.abs(predictions - actual)  # Shape: (n_persons, n_distances)
        
        # Aggregate per-person error metrics from the per-distance errors.
        mean_errors = []
        max_errors = []
        for i in range(actual.shape[0]):
            valid_entries = ~(np.isnan(actual[i, :]) | np.isnan(predictions[i, :]))
            if valid_entries.sum() > 0:
                per_person_errors = errors_per_distance[i, valid_entries]
                mean_errors.append(np.mean(per_person_errors))
                max_errors.append(np.max(per_person_errors))
            else:
                mean_errors.append(np.inf)  # No valid predictions for this sample
                max_errors.append(np.inf)
        
        mean_errors = np.array(mean_errors)
        max_errors = np.array(max_errors)
        ranking_scores = mean_errors if sort_key == "mean" else max_errors
        
        # Determine anomaly threshold (only from finite scores)
        finite_scores = ranking_scores[np.isfinite(ranking_scores)]
        if len(finite_scores) > 0:
            threshold = np.percentile(finite_scores, threshold_percentile)
        else:
            threshold = np.inf
        
        # Filter out samples with inf errors (no valid predictions) - they skew the analysis
        finite_mask = np.isfinite(ranking_scores)
        finite_scores = ranking_scores[finite_mask]
        finite_idx = np.where(finite_mask)[0]
        
        if len(finite_idx) == 0:
            if verbose:
                print("   All samples have no valid predictions (insufficient distance data)")
            return {
                'images': [],
                'distance_connections': normalized_connections,
                'sort_by': sort_key,
                'errors_per_distance': np.array([]),
                'ratios_per_distance': np.array([])
            }
        
        # Sort by selected score
        sorted_positions = np.argsort(-finite_scores)
        sorted_idx = finite_idx[sorted_positions]
        sorted_scores = finite_scores[sorted_positions]
        
        # Compute ratios (error / average error) per distance
        avg_error = np.mean(sorted_scores)
        
        report = []
        for rank, (idx, score) in enumerate(zip(sorted_idx, sorted_scores), start=1):
            # Get per-distance errors and ratios for this person
            per_distance_errors = errors_per_distance[idx, :]
            per_distance_ratios = per_distance_errors / avg_error
            
            report.append({
                'rank': rank,
                'image_path': paths[idx],
                'error': float(mean_errors[idx]),
                'mean_error': float(mean_errors[idx]),
                'max_error': float(max_errors[idx]),
                'sort_score': float(score),
                'is_anomaly': score > threshold,
                'distance_indices': valid_distance_indices.copy(),
                'distance_connections': None if normalized_connections is None else list(normalized_connections),
                'per_distance_errors': per_distance_errors,
                'per_distance_ratios': per_distance_ratios,
            })
        
        # Build 2D arrays for all persons (sorted by rank)
        errors_per_distance_sorted = errors_per_distance[sorted_idx, :]
        ratios_per_distance_sorted = errors_per_distance_sorted / avg_error
        
        return {
            'images': report,
            'distance_connections': normalized_connections,
            'sort_by': sort_key,
            'errors_per_distance': errors_per_distance_sorted,
            'ratios_per_distance': ratios_per_distance_sorted
        }

    
    # ==================== LBP Anomaly Detection ====================

    @staticmethod
    def _lbp_decimals_to_feature_matrix(decimals: Sequence[Any]) -> np.ndarray:
        """Convert LBP decimal values to a binary bit feature matrix for anomaly models.

        Each decimal is expanded to its binary representation so that every bit
        position carries equal weight. Output shape: (N, n_bits) where n_bits is
        determined by the largest value in the sequence.
        """
        int_decimals = [int(d) for d in decimals]
        if not int_decimals:
            return np.empty((0, 1), dtype=np.float64)
        max_val = max(int_decimals)
        n_bits = max_val.bit_length() if max_val > 0 else 1
        return np.array(
            [[(d >> i) & 1 for i in range(n_bits - 1, -1, -1)] for d in int_decimals],
            dtype=np.float64,
        )

    def _predict_lbp_anomalies_by_groups(
        self,
        score_keypoint: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any], Any]],
        patch_size: int,
        visibility_threshold: float,
        similarity_threshold: float,
        similarity_k: int,
        min_group_size: int,
        device: Optional[str],
        image_folder: Optional[str],
        verbose: bool,
        precomputed_embeddings: Optional[np.ndarray],
        precomputed_image_paths: Optional[List[str]],
        keep_models: bool = False,
    ) -> Dict[str, Any]:
        """Shared LBP pipeline of the ``predict_lbp_anomalies_by_embedding_groups*`` methods.

        Groups similar images by embedding, collects visible-keypoint LBP features
        per group and keypoint index, scores each (group, keypoint) sample set with
        ``score_keypoint`` and ranks persons by their highest keypoint score.

        ``score_keypoint(features)`` receives the ``(N, n_bits)`` feature matrix and
        returns ``(scores, is_anomalous, report_fields, model)``: per-sample scores
        (higher = more anomalous), per-sample anomaly flags, extra fields for the
        keypoint report and the fitted model (kept only when ``keep_models``).
        """
        if precomputed_embeddings is not None and precomputed_image_paths is not None:
            embeddings = precomputed_embeddings
            embedding_image_paths = precomputed_image_paths
        else:
            embeddings, embedding_image_paths = self._get_embeddings_with_image_paths(
                device=device,
                image_folder=image_folder,
                verbose=verbose,
            )

        if embeddings.shape[0] == 0:
            if verbose:
                print("   No embeddings computed for dataset images.")
            return {'records': [], 'groups': [], 'all_similarity_groups': [], 'trained_group_ids': []}

        if embeddings.shape[0] == 1:
            all_similarity_groups: List[List[int]] = []
        else:
            effective_k = max(1, min(int(similarity_k), embeddings.shape[0]))
            all_similarity_groups = [
                sorted(group)
                for group in self.build_similarity_groups(embeddings, threshold=similarity_threshold, k=effective_k)
            ]

        trained_groups = [
            (group_id, group)
            for group_id, group in enumerate(all_similarity_groups)
            if len(group) > min_group_size
        ]

        if verbose:
            print(f"All similarity groups: {len(all_similarity_groups)}")
            print(f"Training groups with size > {min_group_size}: {len(trained_groups)}")

        if not trained_groups:
            return {'records': [], 'groups': [], 'all_similarity_groups': all_similarity_groups, 'trained_group_ids': []}

        load_sample_by_path = self._sample_loader_by_path()
        flat_records: List[Dict[str, Any]] = []
        group_reports: List[Dict[str, Any]] = []

        for group_id, group in trained_groups:
            group_image_paths = [embedding_image_paths[index] for index in group]
            keypoint_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            image_records: List[Dict[str, Any]] = []

            for image_path in group_image_paths:
                sample = load_sample_by_path(image_path)
                if sample is None:
                    continue

                person_reports = []
                for person_index, keypoints in enumerate(self._get_sample_person_keypoints(sample)):
                    keypoints_array = np.asarray(keypoints, dtype=float)
                    num_keypoints = keypoints_array.shape[0] if keypoints_array.ndim >= 2 else 0
                    decimal_values = YPImageValidation(sample['image'], keypoints_array).compute_lbp_decimals(
                        patch_size=patch_size,
                        visibility_threshold=visibility_threshold,
                    )
                    if keypoints_array.size > 0 and keypoints_array.shape[1] >= 3:
                        visible_indices = np.where(keypoints_array[:, 2] >= visibility_threshold)[0]
                    else:
                        visible_indices = np.arange(num_keypoints, dtype=int)

                    person_report = {
                        'group_id': group_id,
                        'image_path': image_path,
                        'person_index': person_index,
                        'visible_keypoint_indices': visible_indices.copy(),
                        'keypoint_anomaly_scores': np.full(num_keypoints, np.nan, dtype=float),
                    }
                    for keypoint_index, decimal_value in zip(visible_indices, decimal_values):
                        keypoint_samples[int(keypoint_index)].append({
                            'image_path': image_path,
                            'decimal_value': decimal_value,
                            'person_report': person_report,
                        })
                    person_reports.append(person_report)
                    flat_records.append(person_report)

                image_records.append({'group_id': group_id, 'image_path': image_path, 'persons': person_reports})

            keypoint_models: Dict[int, Any] = {}
            keypoint_group_reports: List[Dict[str, Any]] = []
            for keypoint_index, kpt_samples in sorted(keypoint_samples.items()):
                if len(kpt_samples) < 2:
                    continue

                features = self._lbp_decimals_to_feature_matrix([s['decimal_value'] for s in kpt_samples])
                scores, is_anomalous, report_fields, model = score_keypoint(features)
                for sample_info, score, flagged in zip(kpt_samples, scores, is_anomalous):
                    sample_info['person_report']['keypoint_anomaly_scores'][keypoint_index] = float(score)
                    if flagged:
                        sample_info['person_report'].setdefault('_anomaly_kpts', set()).add(keypoint_index)

                keypoint_models[keypoint_index] = model
                keypoint_group_reports.append({
                    'keypoint_index': keypoint_index,
                    'sample_count': len(kpt_samples),
                    'image_count': len({s['image_path'] for s in kpt_samples}),
                    **report_fields,
                })

            group_report = {
                'group_id': group_id,
                'group_size': len(group),
                'image_paths': group_image_paths,
                'images': image_records,
            }
            if keep_models:
                group_report['keypoint_models'] = keypoint_models
            group_report['keypoint_reports'] = keypoint_group_reports
            group_reports.append(group_report)

        for record in flat_records:
            keypoint_scores = np.asarray(record['keypoint_anomaly_scores'], dtype=float)
            finite_scores = keypoint_scores[np.isfinite(keypoint_scores)]
            record['max_anomaly_score'] = float(np.max(finite_scores)) if finite_scores.size > 0 else np.nan
            record['mean_anomaly_score'] = float(np.mean(finite_scores)) if finite_scores.size > 0 else np.nan
            record['is_anomaly'] = bool(record.pop('_anomaly_kpts', set()))

        # Persons without any scored keypoint (NaN) go last.
        flat_records.sort(
            key=lambda record: (
                np.isfinite(record['max_anomaly_score']),
                record['max_anomaly_score'] if np.isfinite(record['max_anomaly_score']) else float('-inf'),
            ),
            reverse=True,
        )
        for rank, record in enumerate(flat_records, start=1):
            record['rank'] = rank

        return {
            'records': flat_records,
            'groups': group_reports,
            'all_similarity_groups': all_similarity_groups,
            'trained_group_ids': [group_id for group_id, _ in trained_groups],
        }

    @staticmethod
    def _min_max_scale(values: np.ndarray, invert: bool = False) -> np.ndarray:
        """Scale values to [0, 1] (``1 - scaled`` with ``invert``); all zeros when constant."""
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value <= min_value:
            return np.zeros(len(values), dtype=float)
        scaled = (values - min_value) / (max_value - min_value)
        return 1.0 - scaled if invert else scaled

    def predict_lbp_anomalies_by_embedding_groups(
        self,
        patch_size: int = 5,
        visibility_threshold: float = 2.0,
        contamination: float = 0.1,
        random_state: int = 42,
        similarity_threshold: float = 0.95,
        similarity_k: int = DEFAULT_K_NEIGHBORS,
        n_estimators: int = 100,
        min_group_size: int = 10,
        device: Optional[str] = None,
        image_folder: Optional[str] = None,
        verbose: bool = True,
        precomputed_embeddings: Optional[np.ndarray] = None,
        precomputed_image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train per-group, per-keypoint Isolation Forest models from visible LBP decimals.

        Workflow:
        1. Build image similarity groups from CLIP embeddings.
        2. Keep only groups larger than ``min_group_size`` images.
        3. Compute visible-keypoint LBP decimals for images in each kept group.
        4. For each keypoint index inside each group, train a separate Isolation Forest.
           Its ``score_samples`` are min-max scaled and inverted (1 = most anomalous);
           a keypoint is anomalous when the forest predicts ``-1``.

        Returns:
            Dictionary with keys:
            - ``records``: flat list of per-person results with group id, image path,
                            person index, per-keypoint anomaly scores, and max anomaly score;
                            sorted descending by max anomaly score
            - ``groups``: detailed group reports (including the fitted ``keypoint_models``)
            - ``all_similarity_groups``: all image similarity groups from embeddings
            - ``trained_group_ids``: ids of groups used for model training
        """
        def score_keypoint(features):
            model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=n_estimators,
            )
            model.fit(features)
            normalized_scores = self._min_max_scale(model.score_samples(features), invert=True)
            predictions = model.predict(features)  # -1 = anomaly, 1 = inlier
            return normalized_scores, predictions == -1, {'scores': normalized_scores.copy()}, model

        return self._predict_lbp_anomalies_by_groups(
            score_keypoint, patch_size, visibility_threshold, similarity_threshold, similarity_k,
            min_group_size, device, image_folder, verbose, precomputed_embeddings, precomputed_image_paths,
            keep_models=True,
        )

    def predict_lbp_anomalies_by_embedding_groups_zscore(
        self,
        patch_size: int = 5,
        visibility_threshold: float = 2.0,
        z_threshold: float = 2.0,
        similarity_threshold: float = 0.95,
        similarity_k: int = DEFAULT_K_NEIGHBORS,
        min_group_size: int = 10,
        device: Optional[str] = None,
        image_folder: Optional[str] = None,
        verbose: bool = True,
        precomputed_embeddings: Optional[np.ndarray] = None,
        precomputed_image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detect LBP anomalies per group and keypoint using a Z-score.

        Same pipeline as :meth:`predict_lbp_anomalies_by_embedding_groups`, but each
        sample's score is ``|d - mean(d)| / std(d)``, where ``d`` is the distance of
        its LBP bit vector from the group's mean bit vector. A keypoint is anomalous
        when its Z-score exceeds ``z_threshold`` (default 2.0, roughly the top 5 %
        of a normal distribution). Keypoint reports contain ``mean``, ``std`` and
        ``z_scores``.
        """
        def score_keypoint(features):
            distances = np.linalg.norm(features - features.mean(axis=0), axis=1)
            mean_val = float(np.mean(distances))
            std_val = float(np.std(distances))
            if std_val == 0.0:
                z_scores = np.zeros(len(distances), dtype=float)
            else:
                z_scores = np.abs((distances - mean_val) / std_val)
            return z_scores, z_scores > z_threshold, {'mean': mean_val, 'std': std_val, 'z_scores': z_scores.copy()}, None

        return self._predict_lbp_anomalies_by_groups(
            score_keypoint, patch_size, visibility_threshold, similarity_threshold, similarity_k,
            min_group_size, device, image_folder, verbose, precomputed_embeddings, precomputed_image_paths,
        )

    def predict_lbp_anomalies_by_embedding_groups_ocsvm(
        self,
        patch_size: int = 5,
        visibility_threshold: float = 2.0,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
        similarity_threshold: float = 0.95,
        similarity_k: int = DEFAULT_K_NEIGHBORS,
        min_group_size: int = 10,
        device: Optional[str] = None,
        image_folder: Optional[str] = None,
        verbose: bool = True,
        precomputed_embeddings: Optional[np.ndarray] = None,
        precomputed_image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detect LBP anomalies per group and keypoint using a One-Class SVM.

        Same pipeline as :meth:`predict_lbp_anomalies_by_embedding_groups`, but each
        (group, keypoint) gets a :class:`sklearn.svm.OneClassSVM` (``nu``, ``kernel``,
        ``gamma`` are passed through). The negated decision function is min-max
        scaled to [0, 1] (1 = most anomalous); a keypoint is anomalous when the
        model predicts ``-1``. Keypoint reports contain ``normalized_scores``.
        """
        def score_keypoint(features):
            model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
            model.fit(features)
            predictions = model.predict(features)  # -1 = anomaly, 1 = inlier
            normalized_scores = self._min_max_scale(-model.decision_function(features).ravel())
            return normalized_scores, predictions == -1, {'normalized_scores': normalized_scores.copy()}, None

        return self._predict_lbp_anomalies_by_groups(
            score_keypoint, patch_size, visibility_threshold, similarity_threshold, similarity_k,
            min_group_size, device, image_folder, verbose, precomputed_embeddings, precomputed_image_paths,
        )

    @staticmethod
    def _build_similarity_groups(
        embeddings: np.ndarray,
        threshold: float,
        k: int = DEFAULT_K_NEIGHBORS
    ) -> List[List[int]]:
        """Build groups of similar images using FAISS.
        
        Args:
            embeddings: Array of image embeddings
            threshold: Similarity threshold for grouping
            k: Number of neighbors to search
            
        Returns:
            List of groups, where each group is a list of image indices
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        similarities, indices = index.search(embeddings, k)

        graph = defaultdict(set)

        for i in range(len(embeddings)):
            for j in range(1, k):
                if similarities[i][j] > threshold:
                    neighbor = indices[i][j]
                    graph[i].add(neighbor)
                    graph[neighbor].add(i)

        visited = set()
        groups = []

        for node in range(len(embeddings)):
            if node not in visited:
                stack = [node]
                component = []

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(graph[current])

                if len(component) > 1:
                    groups.append(component)

        return groups
    
    @staticmethod
    def find_near_duplicates(
        embeddings: np.ndarray,
        filenames: List[str],
        k: int = DEFAULT_K_NEIGHBORS,
        threshold: float = 0.99
    ) -> List[Tuple[str, str, float]]:
        """Find near-duplicate images based on embeddings.
        
        Args:
            embeddings: Array of image embeddings
            filenames: List of image filenames
            k: Number of neighbors to search
            threshold: Similarity threshold
            
        Returns:
            List of (file1, file2, similarity) tuples for duplicates
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        similarities, indices = index.search(embeddings, k)

        duplicates = []

        for i in range(len(filenames)):
            for j in range(1, k):
                if similarities[i][j] > threshold:
                    duplicates.append(
                        (filenames[i], filenames[indices[i][j]], float(similarities[i][j]))
                    )

        return duplicates
    
    @staticmethod
    def build_similarity_groups(
        embeddings: np.ndarray,
        threshold: float = 0.95,
        k: int = DEFAULT_K_NEIGHBORS
    ) -> List[List[int]]:
        """Build groups of similar images using FAISS.
        
        Args:
            embeddings: Array of image embeddings
            threshold: Similarity threshold for grouping
            k: Number of neighbors to search
            
        Returns:
            List of groups, where each group is a list of image indices
        """
        return YPSetValidation._build_similarity_groups(embeddings, threshold, k)


    @staticmethod
    def get_supported_segmentation_models() -> List[str]:
        """Return canonical names of segmentation models supported by this class."""
        return [
            "maskrcnn_resnet50_fpn",
            "deeplabv3_resnet50",
            "lraspp_mobilenet_v3_large",
            "fcn_resnet50",
            "yolo26n-seg",
            "yolo26s-seg",
            "yolo26m-seg",
            "yolo26l-seg",
            "yolo26x-seg",
        ]

    @staticmethod
    def _canonicalize_segmentation_model_name(model_name: str) -> str:
        normalized = str(model_name).strip().lower().replace("_", "-")
        aliases = {
            "maskrcnn": "maskrcnn_resnet50_fpn",
            "maskrcnn-resnet50-fpn": "maskrcnn_resnet50_fpn",
            "maskrcnn_resnet50_fpn": "maskrcnn_resnet50_fpn",
            "deeplabv3": "deeplabv3_resnet50",
            "deeplabv3-resnet50": "deeplabv3_resnet50",
            "deeplabv3_resnet50": "deeplabv3_resnet50",
            "lraspp": "lraspp_mobilenet_v3_large",
            "lraspp-mobilenet-v3-large": "lraspp_mobilenet_v3_large",
            "lraspp_mobilenet_v3_large": "lraspp_mobilenet_v3_large",
            "fcn": "fcn_resnet50",
            "fcn-resnet50": "fcn_resnet50",
            "fcn_resnet50": "fcn_resnet50",
            "yolo26n": "yolo26n-seg",
            "yolo26n-seg": "yolo26n-seg",
            "yolo26n-seg.pt": "yolo26n-seg",
            "yolo26s": "yolo26s-seg",
            "yolo26s-seg": "yolo26s-seg",
            "yolo26s-seg.pt": "yolo26s-seg",
            "yolo26m": "yolo26m-seg",
            "yolo26m-seg": "yolo26m-seg",
            "yolo26m-seg.pt": "yolo26m-seg",
            "yolo26l": "yolo26l-seg",
            "yolo26l-seg": "yolo26l-seg",
            "yolo26l-seg.pt": "yolo26l-seg",
            "yolo26x": "yolo26x-seg",
            "yolo26x-seg": "yolo26x-seg",
            "yolo26x-seg.pt": "yolo26x-seg",
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            supported = ", ".join(YPSetValidation.get_supported_segmentation_models())
            raise ValueError(
                f"Unsupported segmentation model '{model_name}'. Supported models: {supported}"
            ) from exc

    @staticmethod
    def _get_segmentation_model_metadata(model_name: str) -> Dict[str, Any]:
        canonical_name = YPSetValidation._canonicalize_segmentation_model_name(model_name)

        if canonical_name == "maskrcnn_resnet50_fpn":
            return {
                "canonical_name": canonical_name,
                "family": "torchvision_detection",
                "class_names": COCO_DETECTION_CATEGORIES,
                "class_lookup": COCO_DETECTION_CLASS_LOOKUP,
            }

        if canonical_name in {
            "deeplabv3_resnet50",
            "lraspp_mobilenet_v3_large",
            "fcn_resnet50",
        }:
            return {
                "canonical_name": canonical_name,
                "family": "torchvision_segmentation",
                "class_names": VOC_SEGMENTATION_CLASSES,
                "class_lookup": VOC_SEGMENTATION_CLASS_LOOKUP,
            }

        return {
            "canonical_name": canonical_name,
            "family": "ultralytics_yolo_seg",
            "class_names": COCO_YOLO_CLASSES,
            "class_lookup": COCO_YOLO_CLASS_LOOKUP,
        }

    @staticmethod
    def get_supported_segmentation_classes(model_name: str) -> List[str]:
        """Return valid class names for the selected segmentation model."""
        metadata = YPSetValidation._get_segmentation_model_metadata(model_name)
        return [
            class_name
            for class_name in metadata["class_names"]
            if class_name not in {"N/A", "__background__"}
        ]

    @staticmethod
    def _resolve_segmentation_target_class(
        model_name: str,
        target_class: Union[str, int],
    ) -> Tuple[int, str]:
        metadata = YPSetValidation._get_segmentation_model_metadata(model_name)
        class_names = metadata["class_names"]
        class_lookup = metadata["class_lookup"]

        if isinstance(target_class, str):
            normalized_name = _normalize_class_name(target_class)
            if normalized_name not in class_lookup:
                raise ValueError(
                    f"Class '{target_class}' is not available for model '{metadata['canonical_name']}'."
                )
            class_index = class_lookup[normalized_name]
        else:
            class_index = int(target_class)
            if class_index < 0 or class_index >= len(class_names):
                raise ValueError(
                    f"Class id {class_index} is out of range for model '{metadata['canonical_name']}'."
                )
            if class_names[class_index] in {"N/A", "__background__"}:
                raise ValueError(
                    f"Class id {class_index} is not a valid selectable class for model '{metadata['canonical_name']}'."
                )

        return class_index, class_names[class_index]

    @staticmethod
    def _load_segmentation_model(model_name: str):
        canonical_name = YPSetValidation._canonicalize_segmentation_model_name(model_name)

        if canonical_name == "maskrcnn_resnet50_fpn":
            from torchvision.models.detection import (
                MaskRCNN_ResNet50_FPN_Weights,
                maskrcnn_resnet50_fpn,
            )

            return maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

        if canonical_name == "deeplabv3_resnet50":
            from torchvision.models.segmentation import (
                DeepLabV3_ResNet50_Weights,
                deeplabv3_resnet50,
            )

            return deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            )

        if canonical_name == "lraspp_mobilenet_v3_large":
            from torchvision.models.segmentation import (
                LRASPP_MobileNet_V3_Large_Weights,
                lraspp_mobilenet_v3_large,
            )

            return lraspp_mobilenet_v3_large(
                weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            )

        if canonical_name == "fcn_resnet50":
            from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

            return fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)

        from ultralytics import YOLO

        return YOLO(f"{canonical_name}.pt")

    def _segment_dataset_with_yolo(
        self,
        model,
        target_class: int = 0,
        score_threshold: float = 0.5,
        device: Optional[str] = None,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """Apply an Ultralytics YOLO segmentation model to every image in the dataset."""
        full_masks = []
        image_paths = []

        iterator = self.dataset
        if verbose:
            iterator = tqdm(iterator, total=len(self.dataset), desc="Segmenting dataset (yolo)")

        total = len(self.dataset)
        for idx, sample in enumerate(iterator):
            try:
                image = sample['image']
                image_path = sample['image_path']
                h, w = image.shape[:2]
                binary = np.zeros((h, w), dtype=np.uint8)

                predict_kwargs = {
                    "source": image_path,
                    "verbose": False,
                    "conf": score_threshold,
                }
                if device is not None:
                    predict_kwargs["device"] = device

                result = model.predict(**predict_kwargs)[0]

                if result.masks is not None and result.boxes is not None:
                    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                    confidences = result.boxes.conf.detach().cpu().numpy()
                    polygons = result.masks.xy

                    for det_idx, polygon in enumerate(polygons):
                        if det_idx >= len(classes) or classes[det_idx] != target_class:
                            continue
                        if det_idx < len(confidences) and confidences[det_idx] < score_threshold:
                            continue
                        if len(polygon) < 3:
                            continue
                        points = np.round(polygon).astype(np.int32)
                        cv2.fillPoly(binary, [points], 255)

                full_masks.append(binary)
                image_paths.append(image_path)

            except Exception as e:
                if verbose:
                    print(f"Error processing {sample.get('image_path', 'unknown')}: {e}")
                continue
            
            if progress_callback is not None:
                progress_callback(idx + 1, total)

        return {
            "masks": full_masks,
            "image_paths": image_paths,
        }

    def segment_dataset_with_model(
        self,
        model_name: str,
        target_class: Union[str, int] = "person",
        score_threshold: float = 0.5,
        device: Optional[str] = None,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Load a supported segmentation model, run it on the dataset, and return masks.

        This method wraps the manual notebook workflow into one API call.
        The selected model is loaded from its canonical name or alias, the requested
        class is validated against that model's label space, and a binary mask is
        generated for every image in the dataset.

        Supported models:
        - maskrcnn_resnet50_fpn
        - deeplabv3_resnet50
        - lraspp_mobilenet_v3_large
        - fcn_resnet50
        - yolo26n-seg, yolo26s-seg, yolo26m-seg, yolo26l-seg, yolo26x-seg

        Args:
            model_name: Model name or alias.
            target_class: Class name such as "person" or a numeric class id.
            score_threshold: Confidence threshold for detection and YOLO models.
            device: "cuda" or "cpu". Auto-detected when None.
            verbose: Show tqdm progress bar.

        Returns:
            Dictionary with keys:
            - 'masks': list of binary masks
            - 'image_paths': list of image paths
            - 'model_name': canonical model name that was used
            - 'model_family': torchvision or YOLO family identifier
            - 'target_class_id': resolved numeric class id
            - 'target_class_name': resolved class name

        Raises:
            ValueError: If the model is unsupported or the requested class is not
                available for that model.
        """
        metadata = self._get_segmentation_model_metadata(model_name)
        class_index, class_name = self._resolve_segmentation_target_class(
            metadata["canonical_name"],
            target_class,
        )
        model = self._load_segmentation_model(metadata["canonical_name"])

        if metadata["family"] == "ultralytics_yolo_seg":
            result = self._segment_dataset_with_yolo(
                model=model,
                target_class=class_index,
                score_threshold=score_threshold,
                device=device,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        else:
            torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(torch_device)
            model.eval()

            full_masks = []
            image_paths = []

            iterator = self.dataset
            if verbose:
                iterator = tqdm(iterator, total=len(self.dataset), desc=f"Segmenting dataset ({metadata['family']})")

            total = len(self.dataset)
            for idx, sample in enumerate(iterator):
                try:
                    segmentation = self._predict_sample_segmentation(
                        model=model,
                        model_family=metadata["family"],
                        sample=sample,
                        target_class=class_index,
                        score_threshold=score_threshold,
                        device=torch_device,
                    )
                    full_masks.append(segmentation["mask"])
                    image_paths.append(sample["image_path"])
                except Exception as exc:
                    if verbose:
                        print(f"Error processing {sample.get('image_path', 'unknown')}: {exc}")
                    continue

                if progress_callback is not None:
                    progress_callback(idx + 1, total)

            result = {
                "masks": full_masks,
                "image_paths": image_paths,
            }

        result.update(
            {
                "model_name": metadata["canonical_name"],
                "model_family": metadata["family"],
                "target_class_id": class_index,
                "target_class_name": class_name,
            }
        )
        return result

    def evaluate_dataset_keypoints_against_masks(
        self,
        masks: Sequence[np.ndarray],
        image_paths: Optional[Sequence[str]] = None,
        visibility_threshold: Optional[float] = 2.0,
        mask_tolerance_px: int = 12,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Evaluate dataset keypoints against binary masks.

        This method computes, for each keypoint, whether it lies inside the mask
        and its normalized horizontal distances to the left and right mask edges.

        Args:
            masks: Sequence of binary masks.
            image_paths: Optional paths aligned with ``masks``. When provided,
                masks are matched by image path instead of dataset order.
            visibility_threshold: Optional visibility threshold for keypoints.
            mask_tolerance_px: Pixel tolerance used for inside-mask checks.
                ``0`` means exact pixel-only matching; ``1`` checks 3x3 neighborhood.
            verbose: Show tqdm progress bar.

        Returns:
            Dictionary with keys:
            - ``images``: list of per-image results
            - ``image_paths``: processed image paths
            - ``visibility_threshold``: threshold used during evaluation

        Each element of ``images`` contains:
            - ``image_path``: path to image
            - ``keypoint_metrics``: structured array for single-person images
            - ``person_metrics``: list of structured arrays for multi-person images
        """
        if image_paths is not None:
            if len(masks) != len(image_paths):
                raise ValueError("masks and image_paths must have the same length.")
            mask_by_path = dict(zip(image_paths, masks))
            iterator = self.dataset
        else:
            if len(masks) != len(self.dataset):
                raise ValueError(
                    f"Expected {len(self.dataset)} masks in dataset order, got {len(masks)}."
                )
            iterator = zip(self.dataset, masks)
            mask_by_path = None

        results = []
        processed_paths = []

        if verbose:
            iterator = tqdm(iterator, total=len(self.dataset), desc="Evaluating keypoints vs masks")

        total = len(self.dataset)
        for idx, item in enumerate(iterator):
            if image_paths is not None:
                sample = item
                image_path = sample['image_path']
                if image_path not in mask_by_path:
                    continue
                mask = mask_by_path[image_path]
            else:
                sample, mask = item
                image_path = sample['image_path']

            person_keypoints = self._get_sample_person_keypoints(sample)
            mask_has_foreground = bool(np.any(np.asarray(mask) > 0))

            if len(person_keypoints) > 1:
                person_metrics = []
                for single_person_keypoints in person_keypoints:
                    validator = YPImageValidation(sample['image'], single_person_keypoints)
                    person_metrics.append(
                        validator.evaluate_keypoints_against_mask(
                            mask=mask,
                            visibility_threshold=visibility_threshold,
                            mask_tolerance_px=mask_tolerance_px,
                        )
                    )
                results.append({
                    "image_path": image_path,
                    "person_metrics": person_metrics,
                    "mask_has_foreground": mask_has_foreground,
                })
            else:
                if not person_keypoints:
                    continue
                validator = YPImageValidation(sample['image'], person_keypoints[0])
                results.append({
                    "image_path": image_path,
                    "keypoint_metrics": validator.evaluate_keypoints_against_mask(
                        mask=mask,
                        visibility_threshold=visibility_threshold,
                        mask_tolerance_px=mask_tolerance_px,
                    ),
                    "mask_has_foreground": mask_has_foreground,
                })

            processed_paths.append(image_path)
            
            if progress_callback is not None:
                progress_callback(idx + 1, total)

        return {
            "images": results,
            "image_paths": processed_paths,
            "visibility_threshold": visibility_threshold,
            "mask_tolerance_px": int(mask_tolerance_px),
        }

    def _prepare_keypoint_mask_distance_datasets(
        self,
        evaluation_results: Dict[str, Any],
        variant: Union[int, Sequence[int], Dict[int, int]] = 0,
        drop_nan: bool = True,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Build per-keypoint distance datasets from mask evaluation results."""

        image_results = evaluation_results.get("images")
        if image_results is None:
            raise ValueError("evaluation_results must contain an 'images' key.")

        metric_arrays: List[np.ndarray] = []
        max_keypoint_index = -1

        for image_result in image_results:
            if "person_metrics" in image_result:
                person_metrics = image_result["person_metrics"]
                metric_arrays.extend(person_metrics)
            elif "keypoint_metrics" in image_result:
                metric_arrays.append(image_result["keypoint_metrics"])

        for metrics in metric_arrays:
            if len(metrics) == 0:
                continue
            max_keypoint_index = max(max_keypoint_index, int(np.max(metrics["keypoint_index"])))

        if max_keypoint_index < 0:
            return [], []

        resolved_variants = self._resolve_keypoint_mask_distance_variants(
            variant=variant,
            keypoint_count=max_keypoint_index + 1,
        )

        per_keypoint_values: List[List[Any]] = [list() for _ in range(max_keypoint_index + 1)]

        for metrics in metric_arrays:
            for row in metrics:
                keypoint_index = int(row["keypoint_index"])
                left_distance = float(row["left_distance"])
                right_distance = float(row["right_distance"])
                is_valid = np.isfinite(left_distance) and np.isfinite(right_distance)

                if drop_nan and not is_valid:
                    continue

                current_variant = resolved_variants[keypoint_index]

                if current_variant == 0:
                    if is_valid:
                        per_keypoint_values[keypoint_index].append(
                            min(abs(left_distance), abs(right_distance))
                        )
                    else:
                        per_keypoint_values[keypoint_index].append(np.nan)
                else:
                    if is_valid:
                        per_keypoint_values[keypoint_index].append([left_distance, right_distance])
                    else:
                        per_keypoint_values[keypoint_index].append([np.nan, np.nan])

        datasets: List[np.ndarray] = []
        for keypoint_index, values in enumerate(per_keypoint_values):
            current_variant = resolved_variants[keypoint_index]
            if current_variant == 0:
                datasets.append(np.asarray(values, dtype=float))
            else:
                datasets.append(
                    np.asarray(values, dtype=float).reshape(-1, 2)
                    if values else np.empty((0, 2), dtype=float)
                )

        return datasets, resolved_variants

    @staticmethod
    def _resolve_keypoint_mask_distance_variants(
        variant: Union[int, Sequence[int], Dict[int, int]],
        keypoint_count: int,
    ) -> List[int]:
        """Resolve mask-distance dataset variants for each keypoint index."""
        if isinstance(variant, int):
            if variant not in (0, 1):
                raise ValueError(f"Unsupported variant {variant}. Expected 0 or 1.")
            return [variant] * keypoint_count

        if isinstance(variant, dict):
            resolved_variants = [0] * keypoint_count
            for keypoint_index, keypoint_variant in variant.items():
                index = int(keypoint_index)
                if index < 0 or index >= keypoint_count:
                    raise ValueError(
                        f"Keypoint index {index} is out of range for {keypoint_count} keypoints."
                    )
                if keypoint_variant not in (0, 1):
                    raise ValueError(
                        f"Unsupported variant {keypoint_variant} for keypoint {index}. Expected 0 or 1."
                    )
                resolved_variants[index] = int(keypoint_variant)
            return resolved_variants

        if isinstance(variant, np.ndarray):
            variant_values = variant.tolist()
        elif isinstance(variant, Sequence) and not isinstance(variant, (str, bytes)):
            variant_values = list(variant)
        else:
            raise TypeError(
                "variant must be an int, a dict keyed by keypoint index, or a sequence of per-keypoint variants."
            )

        if len(variant_values) != keypoint_count:
            raise ValueError(
                f"Per-keypoint variant sequence must have length {keypoint_count}, got {len(variant_values)}."
            )

        resolved_variants = []
        for keypoint_index, keypoint_variant in enumerate(variant_values):
            keypoint_variant = int(keypoint_variant)
            if keypoint_variant not in (0, 1):
                raise ValueError(
                    f"Unsupported variant {keypoint_variant} for keypoint {keypoint_index}. Expected 0 or 1."
                )
            resolved_variants.append(keypoint_variant)

        return resolved_variants

    def build_keypoint_mask_distance_datasets(
        self,
        evaluation_results: Dict[str, Any],
        variant: Union[int, Sequence[int], Dict[int, int]] = 0,
        drop_nan: bool = True,
        contamination: Union[str, float] = 'auto',
        random_state: Optional[int] = 42,
        n_estimators: int = 100,
    ) -> Dict[str, Any]:
        """Build per-keypoint datasets and train Isolation Forest models.

        The input must come from ``evaluate_dataset_keypoints_against_masks``.

        Variant semantics:
        - ``0``: for each keypoint, train on the smaller absolute value from
          ``left_distance`` and ``right_distance``
        - ``1``: for each keypoint, train on both distances as
          ``[left_distance, right_distance]``

        Args:
            evaluation_results: Output dictionary from
                ``evaluate_dataset_keypoints_against_masks``.
            variant: Dataset construction mode. Can be a single integer applied
                to all keypoints, a dict mapping ``keypoint_index -> variant``,
                or a full per-keypoint sequence. Supported values are ``0`` and ``1``.
                For dict input, unspecified keypoints default to variant ``0``.
            drop_nan: When ``True``, skip entries with non-finite distances.
            contamination: Isolation Forest contamination parameter.
            random_state: Random seed passed to Isolation Forest.
            n_estimators: Number of trees in each Isolation Forest.

        Returns:
            Dictionary with keys:
            - ``variant``: original variant specification
            - ``drop_nan``: whether non-finite distances were removed before training
            - ``resolved_variants``: resolved variant per keypoint index
            - ``datasets``: list of numpy arrays, one per keypoint
            - ``models``: list of fitted ``IsolationForest`` models or ``None``
            - ``predictions``: list of model predictions arrays
            - ``anomaly_scores``: list of raw ``score_samples`` arrays
            - ``normalized_anomaly_scores``: list of normalized anomaly scores in ``[0, 1]``
            - ``records``: ranked per-person records with ``image_path`` and ``person_id``
        """
        datasets, resolved_variants = self._prepare_keypoint_mask_distance_datasets(
            evaluation_results=evaluation_results,
            variant=variant,
            drop_nan=drop_nan,
        )

        models: List[Optional[IsolationForest]] = []
        predictions: List[np.ndarray] = []
        anomaly_scores: List[np.ndarray] = []
        normalized_anomaly_scores: List[np.ndarray] = []

        for keypoint_index, dataset in enumerate(datasets):
            current_variant = resolved_variants[keypoint_index]
            if current_variant == 0:
                X = np.asarray(dataset, dtype=float).reshape(-1, 1)
            else:
                X = np.asarray(dataset, dtype=float).reshape(-1, 2)

            if X.shape[0] == 0:
                models.append(None)
                predictions.append(np.empty((0,), dtype=int))
                anomaly_scores.append(np.empty((0,), dtype=float))
                normalized_anomaly_scores.append(np.empty((0,), dtype=float))
                continue

            model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=n_estimators,
            )
            model.fit(X)

            current_predictions = model.predict(X)
            current_scores = model.score_samples(X)

            min_score = float(np.min(current_scores))
            max_score = float(np.max(current_scores))
            if max_score > min_score:
                current_normalized_scores = 1.0 - (
                    (current_scores - min_score) / (max_score - min_score)
                )
            else:
                current_normalized_scores = np.zeros_like(current_scores)

            models.append(model)
            predictions.append(current_predictions)
            anomaly_scores.append(current_scores)
            normalized_anomaly_scores.append(current_normalized_scores)

        keypoint_count = len(normalized_anomaly_scores)
        score_cursors = [0] * keypoint_count
        records: List[Dict[str, Any]] = []

        image_results = evaluation_results.get("images") or []
        for image_result in image_results:
            image_path = image_result.get("image_path")
            if "person_metrics" in image_result:
                person_metric_arrays = image_result["person_metrics"]
            elif "keypoint_metrics" in image_result:
                person_metric_arrays = [image_result["keypoint_metrics"]]
            else:
                continue

            for person_id, metrics in enumerate(person_metric_arrays):
                per_keypoint_scores = np.full((keypoint_count,), np.nan, dtype=float)

                for row in metrics:
                    keypoint_index = int(row["keypoint_index"])
                    if keypoint_index < 0 or keypoint_index >= keypoint_count:
                        continue

                    left_distance = float(row["left_distance"])
                    right_distance = float(row["right_distance"])
                    if drop_nan and not (
                        np.isfinite(left_distance) and np.isfinite(right_distance)
                    ):
                        continue

                    cursor = score_cursors[keypoint_index]
                    current_scores = normalized_anomaly_scores[keypoint_index]
                    if cursor >= len(current_scores):
                        raise ValueError(
                            "distance dataset scores do not align with evaluation_results"
                        )

                    per_keypoint_scores[keypoint_index] = float(current_scores[cursor])
                    score_cursors[keypoint_index] += 1

                finite_scores = per_keypoint_scores[np.isfinite(per_keypoint_scores)]
                records.append({
                    "image_path": image_path,
                    "person_id": person_id,
                    "person_index": person_id,
                    "max_anomaly_score": float(np.max(finite_scores)) if finite_scores.size else np.nan,
                    "mean_anomaly_score": float(np.mean(finite_scores)) if finite_scores.size else np.nan,
                    "per_keypoint_scores": per_keypoint_scores.copy(),
                })

        records.sort(
            key=lambda item: item["max_anomaly_score"] if np.isfinite(item["max_anomaly_score"]) else -np.inf,
            reverse=True,
        )

        max_scores = np.asarray(
            [item["max_anomaly_score"] for item in records if np.isfinite(item["max_anomaly_score"])],
            dtype=float,
        )
        threshold = (
            float(np.mean(max_scores) + np.std(max_scores))
            if max_scores.size > 0
            else np.nan
        )

        for rank, record in enumerate(records, start=1):
            record["rank"] = rank
            record["is_anomaly"] = bool(
                np.isfinite(threshold)
                and np.isfinite(record["max_anomaly_score"])
                and record["max_anomaly_score"] > threshold
            )

        return {
            "variant": variant,
            "drop_nan": drop_nan,
            "resolved_variants": resolved_variants,
            "datasets": datasets,
            "models": models,
            "predictions": predictions,
            "anomaly_scores": anomaly_scores,
            "normalized_anomaly_scores": normalized_anomaly_scores,
            "records": records,
        }

    def collect_visible_keypoints_outside_masks(
        self,
        evaluation_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect visible keypoints, separating those outside masks from those without masks.

        The input must come from ``evaluate_dataset_keypoints_against_masks``
        called with a visibility threshold. Invisible keypoints are excluded by
        checking the explicit ``is_visible`` flag stored in the evaluation metrics.

        Keypoints are categorized as:
        - **outside_masks**: inside_mask == 0 and mask was available (has foreground pixels)
        - **without_masks**: inside_mask == 0 and mask was empty (no foreground pixels)

        Args:
            evaluation_results: Output dictionary from
                ``evaluate_dataset_keypoints_against_masks``.

        Returns:
            Dictionary with keys:
            - ``outside_masks``: list of records with visible keypoints outside masks
            - ``without_masks``: list of records with visible keypoints from empty masks
            - ``outside_persons``: unique person-level records containing outside keypoint counts
            - ``without_mask_persons``: unique person-level records from images without masks
            - ``outside_person_count``: number of unique persons with outside keypoints
            - ``without_mask_person_count``: number of unique persons without masks
            
            Each record contains:
            - ``image_path``
            - ``keypoint_index``
            - ``person_index``
            - ``left_distance``
            - ``right_distance``
        """
        image_results = evaluation_results.get("images")
        if image_results is None:
            raise ValueError("evaluation_results must contain an 'images' key.")

        outside_masks: List[Dict[str, Any]] = []
        without_masks: List[Dict[str, Any]] = []

        for image_result in image_results:
            image_path = image_result.get("image_path")

            if "person_metrics" in image_result:
                metric_groups = image_result["person_metrics"]
            elif "keypoint_metrics" in image_result:
                metric_groups = [image_result["keypoint_metrics"]]
            else:
                continue

            # Mask availability = the mask has foreground pixels. Finite edge
            # distances are not a substitute: they exist only for keypoints
            # inside the mask, so a person whose keypoints all fall outside an
            # existing mask would be misfiled as "without mask".
            if "mask_has_foreground" in image_result:
                mask_available = bool(image_result["mask_has_foreground"])
            else:
                # Results without the flag (e.g. saved by earlier runs): a finite
                # edge distance means the person has a mask.
                mask_available = False
                for person_index, metrics in enumerate(metric_groups):
                    for row in metrics:
                        left_distance = float(row["left_distance"])
                        right_distance = float(row["right_distance"])
                        if not (np.isnan(left_distance) and np.isnan(right_distance)):
                            mask_available = True
                            break
                    if mask_available:
                        break

            # Collect keypoints into appropriate category
            for person_index, metrics in enumerate(metric_groups):
                for row in metrics:
                    is_visible = int(row["is_visible"]) if "is_visible" in metrics.dtype.names else 1
                    inside_mask = int(row["inside_mask"])
                    left_distance = float(row["left_distance"])
                    right_distance = float(row["right_distance"])

                    if not is_visible or inside_mask != 0:
                        continue

                    record = {
                        "image_path": image_path,
                        "keypoint_index": int(row["keypoint_index"]),
                        "person_index": person_index,
                        "left_distance": left_distance,
                        "right_distance": right_distance,
                    }

                    if mask_available:
                        outside_masks.append(record)
                    else:
                        without_masks.append(record)

        outside_person_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        without_mask_person_map: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for record in outside_masks:
            key = (os.path.normcase(str(record["image_path"])), int(record["person_index"]))
            bucket = outside_person_map.setdefault(
                key,
                {
                    "image_path": record["image_path"],
                    "person_index": int(record["person_index"]),
                    "outside_keypoint_count": 0,
                    "outside_keypoint_indices": [],
                },
            )
            kp_idx = int(record["keypoint_index"])
            if kp_idx not in bucket["outside_keypoint_indices"]:
                bucket["outside_keypoint_indices"].append(kp_idx)
                bucket["outside_keypoint_count"] += 1

        for record in without_masks:
            key = (os.path.normcase(str(record["image_path"])), int(record["person_index"]))
            bucket = without_mask_person_map.setdefault(
                key,
                {
                    "image_path": record["image_path"],
                    "person_index": int(record["person_index"]),
                    "outside_keypoint_count": 0,
                    "outside_keypoint_indices": [],
                },
            )
            kp_idx = int(record["keypoint_index"])
            if kp_idx not in bucket["outside_keypoint_indices"]:
                bucket["outside_keypoint_indices"].append(kp_idx)
                bucket["outside_keypoint_count"] += 1

        outside_persons = sorted(
            outside_person_map.values(),
            key=lambda item: (
                os.path.normcase(str(item.get("image_path"))),
                int(item.get("person_index", -1)),
            ),
        )
        without_mask_persons = sorted(
            without_mask_person_map.values(),
            key=lambda item: (
                os.path.normcase(str(item.get("image_path"))),
                int(item.get("person_index", -1)),
            ),
        )

        return {
            "outside_masks": outside_masks,
            "without_masks": without_masks,
            "outside_persons": outside_persons,
            "without_mask_persons": without_mask_persons,
            "outside_person_count": len(outside_persons),
            "without_mask_person_count": len(without_mask_persons),
        }
