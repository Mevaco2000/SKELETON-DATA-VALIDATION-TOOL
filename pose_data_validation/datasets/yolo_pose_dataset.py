"""YOLO Pose dataset loading and management."""

import os
from typing import Union, Dict, Any, List, Optional, Sequence, Tuple
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class YOLOSample:
    """Sample from YOLOPoseDataset with image and keypoints."""
    image: np.ndarray
    keypoints: np.ndarray
    image_path: str
    label_path: str
    all_keypoints: List[np.ndarray]
    person_count: int
    
    def __getitem__(self, key):
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)

    def get(self, key, default=None):
        """Allow dict-like get access for backward compatibility."""
        return getattr(self, key, default)


def _first_person_keypoints(all_keypoints: List[np.ndarray]) -> np.ndarray:
    """Return first person's keypoints for backward-compatible single-person APIs."""
    return all_keypoints[0] if all_keypoints else np.array([])


def mirrored_label_rel_path(image_rel_path: str) -> str:
    """Return the label path, relative to ``labels/``, for an image path relative to the dataset.

    A leading ``images`` folder is dropped, anything else is kept, so distinct
    images never share a label: ``images/train/a.jpg`` -> ``train/a.txt``,
    ``seq1/images/a.jpg`` -> ``seq1/images/a.txt``. Converters write labels
    with this rule and ``_candidate_label_paths`` reads them with it.
    """
    parts = [part for part in image_rel_path.replace('\\', '/').split('/') if part and part != '.']
    if len(parts) > 1 and parts[0].lower() == 'images':
        parts = parts[1:]
    return os.path.join(*parts[:-1], os.path.splitext(parts[-1])[0] + '.txt')


def ultralytics_label_path(image_path: str) -> Optional[str]:
    """Return the Ultralytics label path: the last ``images`` folder replaced by ``labels``.

    ``.../data/images/train/a.jpg`` -> ``.../data/labels/train/a.txt``; ``None``
    when the path has no ``images`` folder.
    """
    dir_parts = os.path.normpath(os.path.dirname(image_path)).split(os.sep)
    lowered_parts = [part.lower() for part in dir_parts]
    if 'images' not in lowered_parts:
        return None
    dir_parts[len(lowered_parts) - 1 - lowered_parts[::-1].index('images')] = 'labels'
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(os.sep.join(dir_parts), stem + '.txt')


def _candidate_label_paths(image_rel_path: str, labels_dir: str, dataset_dir: Optional[str] = None) -> List[str]:
    """Return likely label paths for an image path from a split file, in lookup order.

    1. flat ``labels/<name>.txt``;
    2. ``labels/<mirrored_label_rel_path>`` (relative image paths only);
    3. the Ultralytics location next to the image (needs ``dataset_dir`` for
       relative image paths).
    """
    candidate_paths = [
        os.path.join(labels_dir, os.path.splitext(os.path.basename(image_rel_path.replace('\\', '/')))[0] + '.txt'),
    ]
    if os.path.isabs(image_rel_path):
        image_path = image_rel_path
    else:
        candidate_paths.append(os.path.join(labels_dir, mirrored_label_rel_path(image_rel_path)))
        image_path = os.path.join(dataset_dir, image_rel_path) if dataset_dir else None
    if image_path:
        ultralytics_path = ultralytics_label_path(image_path)
        if ultralytics_path:
            candidate_paths.append(ultralytics_path)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(os.path.abspath(path) for path in candidate_paths))


class YOLOPoseImage:
    """
    Represents a single image in YOLO Pose format with keypoint annotations.
    
    Handles loading and managing a single image with its corresponding label file.
    Provides a clean interface for per-image operations and feature extraction.
    
    For feature extraction (LBP, visibility checking, distances), use:
    - YPImageValidation: Wraps YOLOPoseImage with LBP and distance methods
    
    Attributes:
        image_path: Absolute path to image file
        label_path: Absolute path to label file (.txt)
        image: Loaded image as numpy array (H x W x 3 or H x W)
        keypoints: Keypoint array (N, 2) with normalized [x, y] OR (N, 3) with [x, y, visibility]
        image_name: Filename of the image
        label_name: Filename of the label file
    
    Example 1 - Direct usage:
        >>> img = YOLOPoseImage("path/to/image.jpg", "path/to/image.txt")
        >>> print(f"Keypoints: {img.keypoints.shape}")  # (N, 2) or (N, 3)
        >>> cv2.imshow("Image", img.image)
    
    Example 2 - With feature extraction:
        >>> from pose_data_validation.validation import YPImageValidation
        >>> yolo_img = YOLOPoseImage("image.jpg", "image.txt")
        >>> validator = YPImageValidation(yolo_img.image, yolo_img.keypoints)
        >>> decimals = validator.compute_lbp_decimals(patch_size=5)
        >>> distances = validator.sequential_distances(visibility_threshold=0.5)
    
    Example 3 - Get as dict (compatible with YOLOPoseDataset):
        >>> yolo_img = YOLOPoseImage("image.jpg", "image.txt")
        >>> sample = yolo_img.to_dict()
        >>> image = sample['image']
        >>> keypoints = sample['keypoints']
    """
    
    def __init__(self, image_path: Union[str, np.ndarray], label_path: str = None, 
                 include_visibility: bool = True, keypoint_format: str = 'auto'):
        """
        Initialize YOLOPoseImage.
        
        Args:
            image_path: Path to image file or numpy array
            label_path: Path to YOLO label file (.txt). Required if image_path is path.
                       If None and image_path is str, infers label_path from image path.
            include_visibility: If True, load visibility scores as 3rd dimension.
                              If False, load only [x, y] coordinates.
            keypoint_format: One of ``'auto'``, ``'xy'``, or ``'xyv'``.
        
        Raises:
            FileNotFoundError: If image or label file doesn't exist
            ValueError: If image cannot be loaded
        """
        if isinstance(image_path, str):
            self.image_path = os.path.abspath(image_path)
            if not os.path.exists(self.image_path):
                raise FileNotFoundError(f"Image not found: {self.image_path}")
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                raise ValueError(f"Could not load image: {self.image_path}")
            self.image_name = os.path.basename(self.image_path)
        else:
            # Image provided as numpy array
            self.image = image_path
            self.image_path = None
            self.image_name = None
        
        if label_path is None:
            if self.image_path is None:
                raise ValueError("label_path required when image_path is numpy array")
            # Infer label path from image path, supporting mirrored labels/train/... layouts.
            image_dir = os.path.dirname(self.image_path)
            image_root = image_dir
            while os.path.basename(image_root).lower() != 'images':
                parent_dir = os.path.dirname(image_root)
                if parent_dir == image_root:
                    break
                image_root = parent_dir

            if os.path.basename(image_root).lower() == 'images':
                dataset_root = os.path.dirname(image_root)
                image_rel_path = os.path.relpath(self.image_path, dataset_root)
                label_candidates = _candidate_label_paths(
                    image_rel_path, os.path.join(dataset_root, 'labels'), dataset_root
                )
                label_path = next(
                    (candidate for candidate in label_candidates if os.path.exists(candidate)),
                    label_candidates[0],
                )
            else:
                img_name = os.path.splitext(self.image_name)[0]
                label_path = os.path.join(image_dir, '..', 'labels', img_name + '.txt')
        
        self.label_path = os.path.abspath(label_path)
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file not found: {self.label_path}")
        
        self.label_name = os.path.basename(self.label_path)
        self.keypoint_format = keypoint_format
        
        self._load_keypoints(include_visibility)
    
    @classmethod
    def from_yaml_index(cls, data_yaml: str, index: int, split: str = 'train',
                       include_visibility: bool = True,
                       keypoint_format: str = 'auto') -> 'YOLOPoseImage':
        """
        Load image by index from a data.yaml config.
        
        Args:
            data_yaml: Path to data.yaml file
            index: Index of image in the split
            split: Which split to load ('train', 'val', 'test'). Default 'train'.
            include_visibility: If True, load visibility scores as 3rd dimension.
            keypoint_format: One of ``'auto'``, ``'xy'``, or ``'xyv'``.
            
        Returns:
            YOLOPoseImage instance
            
        Raises:
            FileNotFoundError: If data.yaml or dataset files not found
            IndexError: If index is out of range
            
        Example:
            >>> img = YOLOPoseImage.from_dataset_index("data.yaml", index=0, split="train")
            >>> distances = img.get_sequential_distances()
        """
        dataset = YOLOPoseDataset(data_yaml, keypoint_format=keypoint_format)
        sample = dataset[index]
        
        return cls(
            image_path=sample['image_path'],
            label_path=sample['label_path'],
            include_visibility=include_visibility,
            keypoint_format=keypoint_format,
        )
    
    @classmethod
    def from_txt_index(cls, file_list_path: str, index: int, labels_dir: str = None,
                      include_visibility: bool = True,
                      keypoint_format: str = 'auto') -> 'YOLOPoseImage':
        """
        Load image by index from a file list (train.txt, val.txt, etc.).
        
        Args:
            file_list_path: Path to file list (.txt) with image paths, one per line
            index: Index of image in the file list
            labels_dir: Path to labels directory. If None, inferred from file_list_path parent.
                       Defaults to: os.path.dirname(file_list_path)/../labels/
            include_visibility: If True, load visibility scores as 3rd dimension.
            keypoint_format: One of ``'auto'``, ``'xy'``, or ``'xyv'``.
            
        Returns:
            YOLOPoseImage instance
            
        Raises:
            FileNotFoundError: If file list or dataset files not found
            IndexError: If index is out of range
            
        Example:
            >>> img = YOLOPoseImage.from_file_index("train.txt", index=42)
            >>> distances = img.get_sequential_distances()
            >>> 
            >>> # From val.txt with custom labels directory
            >>> img = YOLOPoseImage.from_file_index("val.txt", 0, labels_dir="custom/labels")
        """
        dataset = YOLOPoseDataset(file_list_path, labels_dir=labels_dir, keypoint_format=keypoint_format)
        sample = dataset[index]
        
        return cls(
            image_path=sample['image_path'],
            label_path=sample['label_path'],
            include_visibility=include_visibility,
            keypoint_format=keypoint_format,
        )
    
    def _load_keypoints(self, include_visibility: bool = True):
        """Load keypoints from label file."""
        from ..validation.image_validation import YPImageValidation
        self.all_keypoints = YPImageValidation.load_all_keypoints_from_file(
            self.label_path,
            include_visibility=include_visibility,
            keypoint_format=self.keypoint_format,
        )
        self.person_count = len(self.all_keypoints)
        self.keypoints = _first_person_keypoints(self.all_keypoints)
    
    @property
    def num_keypoints(self) -> int:
        """Get number of keypoints."""
        return len(self.keypoints)
    
    @property
    def image_size(self) -> tuple:
        """Get image dimensions as (height, width)."""
        return self.image.shape[:2]
    
    @property
    def has_visibility(self) -> bool:
        """Check if keypoints include visibility scores."""
        return self.keypoints.shape[1] >= 3
    
    def get_sequential_distances(
        self,
        visibility_threshold: float = 2.0,
        distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Get distances between visible keypoint pairs.
        
        By default computes consecutive distances. When ``distance_connections``
        is provided, computes distances for that explicit ordered list of pairs.
        
        Args:
            visibility_threshold: Minimum visibility score (0=out, 1=hidden, 2=visible).
                                Default 2.0 (only fully visible keypoints).
            distance_connections: Optional ordered list of ``(start_idx, end_idx)``
                                keypoint pairs to measure instead of consecutive pairs.
        
        Returns:
            Array of distances between consecutive visible keypoints.
            Returns empty array if fewer than 2 visible keypoints.
            
        Example:
            >>> yolo_img = YOLOPoseImage("image.jpg", "label.txt")
            >>> distances = yolo_img.get_sequential_distances(visibility_threshold=0.5)
            >>> print(distances)  # [0.045, 0.052, 0.031, ...]
            >>> print(f"Mean distance: {distances.mean():.4f}")
        """
        from ..validation.image_validation import YPImageValidation
        validator = YPImageValidation(self.image, self.keypoints)
        return validator.sequential_distances(
            visibility_threshold=visibility_threshold,
            distance_connections=distance_connections,
        )

    def get_sequential_angles(self, visibility_threshold: float = 2.0) -> np.ndarray:
        """Get segment angles between consecutive visible keypoints.

        Args:
            visibility_threshold: Minimum visibility score (0=out, 1=hidden, 2=visible).
                Default 2.0 (only fully visible keypoints).

        Returns:
            Structured array with fields ``angle_index`` and ``angle`` in degrees.
        """
        from ..validation.image_validation import YPImageValidation
        validator = YPImageValidation(self.image, self.keypoints)
        return validator.sequential_angles(visibility_threshold=visibility_threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format compatible with YOLOPoseDataset.
        
        Returns:
            Dictionary with keys: 'image', 'keypoints', 'image_path', 'label_path'
        """
        return {
            'image': self.image,
            'keypoints': self.keypoints,
            'image_path': self.image_path,
            'label_path': self.label_path,
            'all_keypoints': self.all_keypoints,
            'person_count': self.person_count,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"YOLOPoseImage(image={self.image_name}, "
                f"persons={self.person_count}, "
                f"keypoints={self.num_keypoints}, "
                f"visibility={self.has_visibility})")


class YOLOPoseDataset:
    """
    Class for working with YOLO Pose 1.0 format datasets.
    
    Handles loading, managing, and iterating through YOLO Pose datasets with keypoint annotations.
    
    For feature extraction, use:
    - YPImageValidation: For single image operations
    - YPSetValidation: For dataset-level operations (LBP histograms, decimals, dataframes)
    
    Supports multiple input formats:
    - data.yaml (YOLO dataset config) - auto-detects train/val/test split
    - args.yaml (training config) - loads from training directory  
    - train.txt (image list) - direct path to train.txt file
    
    Attributes:
        dataset_file: Path to train.txt file
        dataset_dir: Directory containing images
        labels_dir: Directory containing label files

    USAGE PATTERNS:
    
    Example 1 - Load and iterate (basic):
        >>> dataset = YOLOPoseDataset("data.yaml")  # or "train.txt"
        >>> for sample in dataset:
        ...     image = sample['image']  # numpy array
        ...     keypoints = sample['keypoints']  # (N, 2) array
        ...     image_path = sample['image_path']
    
    Example 2 - Get LBP features (use YPSetValidation):
        >>> from pose_data_validation.validation import YPSetValidation
        >>> validator = YPSetValidation(dataset)
        >>> decimals, counts, paths = validator.get_all_lbp_decimals(patch_size=7)
    
    Example 3 - Single image features (use YPImageValidation):
        >>> from pose_data_validation.validation import YPImageValidation
        >>> sample = dataset[0]
        >>> validator = YPImageValidation(sample['image'], sample['keypoints'])
        >>> decimals = validator.compute_lbp_decimals(patch_size=5)
    
    Example 4 - Use YOLOPoseImage for per-image operations:
        >>> from pose_data_validation.datasets import YOLOPoseImage
        >>> yolo_img = YOLOPoseImage("image.jpg", "image.txt")
        >>> print(f"Keypoints: {yolo_img.keypoints.shape}")
    """
    
    def __init__(self, dataset_file: str, labels_dir: str = None, keypoint_format: str = 'auto'):
        """
        Initialize YOLO Pose dataset.
        
        Args:
            dataset_file: Path to a dataset YAML file or train.txt file
                - any .yaml/.yml file: dataset config (auto-detects train/val/test split)
                - train.txt: File with relative image paths, one per line
            labels_dir: Directory with label files. If None, infers from dataset_file parent
            keypoint_format: One of ``'auto'``, ``'xy'``, or ``'xyv'``.
        """
        dataset_file_abs = os.path.abspath(dataset_file)
        
        # Treat any YAML file as a dataset config.
        if dataset_file_abs.lower().endswith(('.yaml', '.yml')):
            self.dataset_file, self.dataset_dir = self._load_from_data_yaml(dataset_file_abs, labels_dir)
        else:
            self.dataset_file = dataset_file_abs
            self.dataset_dir = os.path.dirname(self.dataset_file)
        
        if labels_dir is None:
            self.labels_dir = os.path.join(self.dataset_dir, "labels")
        else:
            self.labels_dir = os.path.abspath(labels_dir)
        self.keypoint_format = keypoint_format
        
        # utf-8-sig: every writer in this package uses UTF-8; also tolerate a BOM.
        with open(self.dataset_file, "r", encoding="utf-8-sig") as f:
            self.image_rel_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        self._valid_pairs = self._validate_pairs()
    
    @staticmethod
    def _load_from_data_yaml(data_yaml_path: str, labels_dir: str = None) -> tuple:
        """
        Parse data.yaml and find train.txt, val.txt, or test.txt automatically.
        
        Args:
            data_yaml_path: Absolute path to data.yaml
            labels_dir: Optional explicit labels directory
            
        Returns:
            (dataset_file_path, dataset_dir) tuple
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for data.yaml support. Install: pip install pyyaml")
        
        with open(data_yaml_path, 'r', encoding='utf-8-sig') as f:
            data_config = yaml.safe_load(f)
        
        # Get dataset root path from data.yaml
        dataset_root = data_config.get('path', os.path.dirname(data_yaml_path))
        if not os.path.isabs(dataset_root):
            dataset_root = os.path.abspath(os.path.join(os.path.dirname(data_yaml_path), dataset_root))
        
        # Try to find train.txt, val.txt, test.txt in priority order
        split_priorities = ['train', 'val', 'test']
        found_file = None
        found_dir = None
        
        for split in split_priorities:
            split_value = data_config.get(split)
            candidate_paths = []

            if split_value:
                resolved_split_path = split_value
                if not os.path.isabs(resolved_split_path):
                    resolved_split_path = os.path.join(dataset_root, resolved_split_path)
                candidate_paths.append(os.path.abspath(resolved_split_path))
            else:
                candidate_paths.append(os.path.join(dataset_root, split))

            candidate_txt_files = []
            for candidate_path in candidate_paths:
                candidate_path_abs = os.path.abspath(candidate_path)
                if candidate_path_abs.lower().endswith('.txt'):
                    candidate_txt_files.append(candidate_path_abs)
                else:
                    candidate_txt_files.append(os.path.join(candidate_path_abs, f'{split}.txt'))

            candidate_txt_files.extend([
                os.path.join(dataset_root, f'{split}.txt'),
                os.path.join(os.path.dirname(data_yaml_path), f'{split}.txt'),
            ])

            seen_txt_files = set()
            for txt_file in candidate_txt_files:
                txt_file_abs = os.path.abspath(txt_file)
                if txt_file_abs in seen_txt_files:
                    continue
                seen_txt_files.add(txt_file_abs)

                if os.path.exists(txt_file_abs):
                    found_file = txt_file_abs
                    found_dir = os.path.dirname(txt_file_abs)
                    print(f"Loading dataset from: {found_file}")
                    break

            if found_file is not None:
                break
        
        if not found_file:
            raise FileNotFoundError(
                f"Could not find train.txt, val.txt, or test.txt in {dataset_root}\n"
                f"Checked paths based on data.yaml: {data_config}"
            )
        
        return found_file, found_dir

    @staticmethod
    def generate_train_txt_from_images(
        dataset_root: str,
        images_dir: str = 'images',
        output_file: str = None,
        recursive: bool = True,
    ) -> str:
        """Generate train.txt by listing image files under an images directory.

        Args:
            dataset_root: Root directory of the dataset.
            images_dir: Images directory path, relative to ``dataset_root`` or absolute.
            output_file: Output train.txt path. Defaults to ``dataset_root/train.txt``.
            recursive: Whether to scan nested image subdirectories.

        Returns:
            Absolute path to the generated train.txt file.

        Example:
            >>> train_txt = YOLOPoseDataset.generate_train_txt_from_images(
            ...     dataset_root="./my_dataset",
            ...     images_dir="images",
            ... )
            >>> dataset = YOLOPoseDataset(train_txt)
        """
        from .operations import generate_train_txt_from_images

        return generate_train_txt_from_images(
            dataset_root=dataset_root,
            images_dir=images_dir,
            output_file=output_file,
            recursive=recursive,
        )
    
    def _validate_pairs(self):
        """Check which image/label pairs exist and are valid."""
        from ..validation.image_validation import YPImageValidation
        
        valid = []
        for rel_path in self.image_rel_paths:
            image_path = os.path.join(self.dataset_dir, rel_path)
            if not os.path.exists(image_path):
                continue

            label_path = None
            for candidate_label_path in _candidate_label_paths(rel_path, self.labels_dir, self.dataset_dir):
                if os.path.exists(candidate_label_path):
                    label_path = candidate_label_path
                    break

            if label_path is None:
                continue
            
            # Try loading to ensure validity
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                all_keypoints = YPImageValidation.load_all_keypoints_from_file(
                    label_path,
                    keypoint_format=self.keypoint_format,
                )
                if not any(kpts is not None and kpts.size > 0 for kpts in all_keypoints):
                    continue
                valid.append((image_path, label_path))
            except Exception:
                continue
        
        return valid
    
    @property
    def stats(self) -> dict:
        """Get dataset statistics.
        
        Counts ALL persons per image (not just the first one).
        This matches get_sequential_distances() behavior.
        """
        total_images = len(self._valid_pairs)
        
        # Count total keypoints and visible keypoints across ALL persons
        total_keypoints = 0
        total_visible_keypoints = 0
        total_persons = 0
        
        for _, label_path in self._valid_pairs:
            try:
                from ..validation.image_validation import YPImageValidation
                all_kpts = YPImageValidation.load_all_keypoints_from_file(
                    label_path,
                    include_visibility=True,
                    keypoint_format=self.keypoint_format,
                )
                
                for kpts in all_kpts:
                    if kpts is not None and len(kpts) >= 2:  # Only count persons with at least 2 keypoints
                        total_persons += 1
                        total_keypoints += len(kpts)
                        # Count visible keypoints (visibility >= 2.0)
                        if kpts.shape[1] >= 3:
                            visibility = kpts[:, 2]
                            visible_count = np.sum(visibility >= 2.0)
                            total_visible_keypoints += visible_count
                        else:
                            # If no visibility info, count all as visible
                            total_visible_keypoints += len(kpts)
            except Exception:
                pass
        
        return {
            'total_images': total_images,
            'total_persons': total_persons,
            'total_keypoints': total_keypoints,
            'total_visible_keypoints': total_visible_keypoints,
            'avg_keypoints_per_image': total_keypoints / total_images if total_images > 0 else 0,
            'avg_keypoints_per_person': total_keypoints / total_persons if total_persons > 0 else 0,
            'avg_visible_keypoints_per_image': total_visible_keypoints / total_images if total_images > 0 else 0,
            'avg_visible_keypoints_per_person': total_visible_keypoints / total_persons if total_persons > 0 else 0,
            'dataset_file': self.dataset_file,
            'labels_dir': self.labels_dir
        }
    
    def get_keypoint_distribution(self, verbose: bool = True) -> dict:
        """Get distribution of visible keypoints per person.
        
        Shows how many persons have each count of visible keypoints.
        
        Args:
            verbose: Print formatted distribution table
            
        Returns:
            Dictionary mapping num_visible_keypoints -> count of persons with that many
            
        Example:
            >>> dist = dataset.get_keypoint_distribution()
            >>> print(dist)  # {5: 10, 6: 45, 7: 120, 8: 234, 9: 3349}
        """
        from collections import Counter
        
        visible_keypoint_counts = []
        
        for _, label_path in self._valid_pairs:
            try:
                from ..validation.image_validation import YPImageValidation
                all_kpts = YPImageValidation.load_all_keypoints_from_file(
                    label_path,
                    include_visibility=True,
                    keypoint_format=self.keypoint_format,
                )
                
                for kpts in all_kpts:
                    if kpts is not None and len(kpts) >= 2:
                        if kpts.shape[1] >= 3:
                            visibility = kpts[:, 2]
                            visible_count = np.sum(visibility >= 2.0)
                        else:
                            visible_count = len(kpts)
                        visible_keypoint_counts.append(int(visible_count))
            except Exception:
                pass
        
        distribution = dict(sorted(Counter(visible_keypoint_counts).items()))
        
        if verbose:
            print("Distribution of Visible Keypoints per Person:")
            print("-" * 60)
            print(f"{'Visible Keypoints':>20} | {'Count':>10} | {'Percentage':>10}")
            print("-" * 60)
            
            total = sum(distribution.values())
            for num_keypoints in sorted(distribution.keys()):
                count = distribution[num_keypoints]
                percentage = 100 * count / total if total > 0 else 0
                print(f"{num_keypoints:>20} | {count:>10} | {percentage:>10.2f}%")
            
            print("-" * 60)
            print(f"{'Total Persons':>20} | {total:>10}")
            print()
        
        return distribution
    
    def get_keypoint_statistics(self, verbose: bool = True) -> dict:
        """Get visibility statistics for each keypoint index.
        
        Shows how many times each keypoint (0-8) appears, is visible, or is occluded.
        
        Args:
            verbose: Print formatted statistics table
            
        Returns:
            Dictionary mapping keypoint_index -> {'total': count, 'visible': count, 'occluded': count}
            
        Example:
            >>> stats = dataset.get_keypoint_statistics()
            >>> print(stats[0])  # {'total': 4758, 'visible': 4702, 'occluded': 56}
        """
        stats = {}
        
        for _, label_path in self._valid_pairs:
            try:
                from ..validation.image_validation import YPImageValidation
                all_kpts = YPImageValidation.load_all_keypoints_from_file(
                    label_path,
                    include_visibility=True,
                    keypoint_format=self.keypoint_format,
                )
                
                for kpts in all_kpts:
                    if kpts is not None and len(kpts) >= 2:
                        for keypoint_idx in range(len(kpts)):
                            if keypoint_idx not in stats:
                                stats[keypoint_idx] = {'total': 0, 'visible': 0, 'occluded': 0}
                            
                            stats[keypoint_idx]['total'] += 1
                            
                            if kpts.shape[1] >= 3:
                                visibility = kpts[keypoint_idx, 2]
                                if visibility >= 2.0:
                                    stats[keypoint_idx]['visible'] += 1
                                else:
                                    stats[keypoint_idx]['occluded'] += 1
                            else:
                                stats[keypoint_idx]['visible'] += 1
            except Exception:
                pass
        
        if verbose:
            print("Keypoint Statistics (Visibility across entire dataset):")
            print("-" * 85)
            print(f"{'Keypoint':>10} | {'Total':>10} | {'Visible':>10} | {'Occluded':>10} | {'Visibility %':>12}")
            print("-" * 85)
            
            for kp_idx in sorted(stats.keys()):
                total = stats[kp_idx]['total']
                visible = stats[kp_idx]['visible']
                occluded = stats[kp_idx]['occluded']
                visibility_pct = 100 * visible / total if total > 0 else 0
                
                print(f"{kp_idx:>10} | {total:>10} | {visible:>10} | {occluded:>10} | {visibility_pct:>11.2f}%")
            
            print("-" * 85)
        
        return stats
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self._valid_pairs)
    
    def __iter__(self):
        """Iterate through dataset, yielding samples WITHOUT LBP features.
        
        Use iterate_with_lbp(patch_size) if you need LBP features.
        """
        from ..validation.image_validation import YPImageValidation
        
        for image_path, label_path in self._valid_pairs:
            try:
                image = cv2.imread(image_path)
                all_keypoints = YPImageValidation.load_all_keypoints_from_file(
                    label_path,
                    keypoint_format=self.keypoint_format,
                )
                keypoints = _first_person_keypoints(all_keypoints)
                
                if image is None or keypoints is None or keypoints.size == 0:
                    continue
                
                yield {
                    'image': image,
                    'keypoints': keypoints,
                    'image_path': image_path,
                    'label_path': label_path,
                    'all_keypoints': all_keypoints,
                    'person_count': len(all_keypoints),
                }
            except Exception:
                continue
    
    def __getitem__(self, idx: int) -> YOLOSample:
        """Get sample by index without computing features.
        
        Returns a YOLOSample object with attributes: image, keypoints, image_path, label_path
        
        Also supports dict-like access for backward compatibility:
        - sample.image or sample['image']
        - sample.keypoints or sample['keypoints']
        - sample.image_path or sample['image_path']
        - sample.label_path or sample['label_path']
        """
        from ..validation.image_validation import YPImageValidation
        
        if idx >= len(self._valid_pairs):
            raise IndexError(f"Sample index {idx} out of range {len(self._valid_pairs)}")
        
        image_path, label_path = self._valid_pairs[idx]
        
        image = cv2.imread(image_path)
        all_keypoints = YPImageValidation.load_all_keypoints_from_file(
            label_path,
            keypoint_format=self.keypoint_format,
        )
        keypoints = _first_person_keypoints(all_keypoints)
        
        return YOLOSample(
            image=image,
            keypoints=keypoints,
            image_path=image_path,
            label_path=label_path,
            all_keypoints=all_keypoints,
            person_count=len(all_keypoints),
        )    
    def get_sequential_distances(self, idx: int, visibility_threshold: float = 2.0) -> np.ndarray:
        """
        Get sequential distances for a single sample by index.
        
        Computes Euclidean distances between consecutive visible keypoints.
        Skips distances involving hidden/occluded keypoints or out-of-frame keypoints.
        
        Args:
            idx: Sample index in dataset
            visibility_threshold: Minimum visibility score (0=out, 1=hidden, 2=visible).
                                Default 2.0 (only fully visible keypoints).
        
        Returns:
            Array of distances between consecutive visible keypoints.
            Returns empty array if fewer than 2 visible keypoints.
            
        Raises:
            IndexError: If idx is out of range
            
        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> distances = dataset.get_sequential_distances(0, visibility_threshold=0.5)
            >>> print(distances)  # [0.045, 0.052, 0.031, ...]
            >>> print(f"Mean distance: {distances.mean():.4f}")
        """
        sample = self[idx]
        image = sample['image']
        keypoints = sample['keypoints']
        
        from ..validation.image_validation import YPImageValidation
        validator = YPImageValidation(image, keypoints)
        return validator.sequential_distances(visibility_threshold=visibility_threshold)

    def get_sequential_angles(self, idx: int, visibility_threshold: float = 2.0) -> np.ndarray:
        """Get segment angles for a single sample by index.

        Args:
            idx: Sample index in dataset.
            visibility_threshold: Minimum visibility score (0=out, 1=hidden, 2=visible).
                Default 2.0 (only fully visible keypoints).

        Returns:
            Structured array with fields ``angle_index`` and ``angle`` in degrees.
        """
        sample = self[idx]
        image = sample['image']
        keypoints = sample['keypoints']

        from ..validation.image_validation import YPImageValidation
        validator = YPImageValidation(image, keypoints)
        return validator.sequential_angles(visibility_threshold=visibility_threshold)
    
    def get_all_keypoints(self) -> list:
        """
        Get all keypoints from dataset as a list.
        
        Returns:
            List of keypoint arrays where each element is shape (N, 2) or (N, 3)
            
        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> all_kpts = dataset.get_all_keypoints()
            >>> print(f"Total images: {len(all_kpts)}")
            >>> print(f"First image keypoints: {all_kpts[0].shape}")
        """
        return [sample['keypoints'] for sample in self]
    
    def get_all_images(self) -> list:
        """
        Get all images from dataset as a list.
        
        WARNING: Loads all images into memory. Only use for small/medium datasets.
        
        Returns:
            List of image numpy arrays (H, W, 3 in BGR format)
            
        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> all_imgs = dataset.get_all_images()
            >>> print(f"Total images: {len(all_imgs)}")
            >>> print(f"First image shape: {all_imgs[0].shape}")
        """
        return [sample['image'] for sample in self]
    
    def get_all_image_paths(self) -> list:
        """
        Get all image file paths from dataset.
        
        Returns:
            List of absolute image file paths
            
        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> paths = dataset.get_all_image_paths()
            >>> print(f"Total images: {len(paths)}")
            >>> print(f"First image: {paths[0]}")
        """
        return [path for path, _ in self._valid_pairs]
    
    def get_all_label_paths(self) -> list:
        """
        Get all label file paths from dataset.
        
        Returns:
            List of absolute label file paths (.txt)
            
        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> paths = dataset.get_all_label_paths()
            >>> print(f"Total labels: {len(paths)}")
            >>> print(f"First label: {paths[0]}")
        """
        return [path for _, path in self._valid_pairs]
    
    def get_all_samples(self) -> list:
        """
        Get all samples from dataset.
        
        Returns:
            List of dicts with keys: 'image', 'keypoints', 'image_path', 'label_path'
            
        WARNING: Loads all images into memory. Only use for small/medium datasets.
        
        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> samples = dataset.get_all_samples()
            >>> print(f"Total samples: {len(samples)}")
            >>> print(f"First sample keys: {samples[0].keys()}")
        """
        return [sample for sample in self]
    
    

    def replace_angle_preserving_keypoints(
        self,
        sample_count: int,
        max_keypoints_to_shift: int,
        max_endpoint_shift: float,
        min_shift: float = 0.0,
        output_dataset_path: str = None,
        seed: int = None,
    ) -> list:
        """Replace existing keypoints in dataset labels while preserving segment angles.

        This method modifies existing label files in place. It does not create
        additional dataset samples.

        Args:
            sample_count: Number of existing person annotations to replace.
            max_keypoints_to_shift: Maximum number of randomly selected keypoints
                to shift in one replaced annotation.
            max_endpoint_shift: Maximum shift magnitude for the first and last
                keypoint in the sequence.
            min_shift: Minimum shift magnitude to apply. If this value is larger
                than what a selected keypoint block allows, the available maximum
                value is used.
            output_dataset_path: Optional path where a copied dataset with
                modified labels should be written.
            seed: Optional RNG seed for reproducibility.

        Returns:
            List of replacement result objects.
        """
        from .operations import replace_angle_preserving_keypoints

        return replace_angle_preserving_keypoints(
            self,
            sample_count=sample_count,
            max_keypoints_to_shift=max_keypoints_to_shift,
            max_endpoint_shift=max_endpoint_shift,
            min_shift=min_shift,
            output_dataset_path=output_dataset_path,
            seed=seed,
        )

    def replace_rigid_keypoints(
        self,
        sample_count: int,
        max_rotation_degrees: float,
        max_translation_distance: float,
        transform_mode: str = 'both',
        output_dataset_path: str = None,
        seed: int = None,
    ) -> list:
        """Replace existing keypoints using translation, rotation, or both.

        This method modifies existing label files in place. It applies a rigid
        transform to all keypoints of selected persons, preserving their mutual
        geometry.

        Args:
            sample_count: Number of existing person annotations to replace.
            max_rotation_degrees: Maximum absolute rotation angle in degrees.
            max_translation_distance: Maximum translation magnitude in normalized units.
            transform_mode: One of ``'translate'``, ``'rotate'`` or ``'both'``.
            output_dataset_path: Optional path where a copied dataset with
                modified labels should be written.
            seed: Optional RNG seed for reproducibility.

        Returns:
            List of rigid replacement result objects.
        """
        from .operations import replace_rigid_keypoints

        return replace_rigid_keypoints(
            self,
            sample_count=sample_count,
            max_rotation_degrees=max_rotation_degrees,
            max_translation_distance=max_translation_distance,
            transform_mode=transform_mode,
            output_dataset_path=output_dataset_path,
            seed=seed,
        )

    def replace_axis_aligned_keypoints(
        self,
        sample_count: int,
        max_keypoints_to_shift: int,
        max_shift_distance: float,
        direction_mode: str = 'both',
        output_dataset_path: str = None,
        seed: int = None,
    ) -> list:
        """Replace existing keypoints by shifting a random subset along axes.

        This method modifies existing label files in place. Each selected person
        gets a random subset of keypoints shifted horizontally, vertically, or in
        both axes.

        Args:
            sample_count: Number of existing person annotations to replace.
            max_keypoints_to_shift: Maximum number of keypoints to shift per person.
            max_shift_distance: Maximum per-axis shift magnitude in normalized units.
            direction_mode: One of ``'horizontal'``, ``'vertical'`` or ``'both'``.
            output_dataset_path: Optional path where a copied dataset with
                modified labels should be written.
            seed: Optional RNG seed for reproducibility.

        Returns:
            List of axis-aligned replacement result objects.
        """
        from .operations import replace_axis_aligned_keypoints

        return replace_axis_aligned_keypoints(
            self,
            sample_count=sample_count,
            max_keypoints_to_shift=max_keypoints_to_shift,
            max_shift_distance=max_shift_distance,
            direction_mode=direction_mode,
            output_dataset_path=output_dataset_path,
            seed=seed,
        )

    def export_random_subset(
        self,
        sample_count: int,
        output_dataset_path: str,
        seed: int = None,
    ) -> str:
        """Export a random subset of dataset samples into a new dataset directory.

        The new dataset contains only the selected images, their corresponding
        labels, a rewritten split file, and a copied ``data.yaml`` when the
        source dataset provides one.

        Args:
            sample_count: Number of unique samples to export.
            output_dataset_path: Destination directory for the new dataset.
            seed: Optional RNG seed for reproducible sampling.

        Returns:
            Absolute path to the generated split file in the new dataset.

        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> subset_train = dataset.export_random_subset(
            ...     sample_count=100,
            ...     output_dataset_path="./subset_dataset",
            ...     seed=123,
            ... )
            >>> print(subset_train)
        """
        from .operations import export_random_dataset_subset

        return export_random_dataset_subset(
            self,
            sample_count=sample_count,
            output_dataset_path=output_dataset_path,
            seed=seed,
        )

    def export_dataset_partitions(
        self,
        part_count: int,
        output_root_dir: str,
        seed: int = None,
        shuffle: bool = True,
    ) -> List[str]:
        """Export the dataset into a chosen number of parts.

        Args:
            part_count: Number of parts to create.
            output_root_dir: Destination directory for generated part folders.
            seed: Optional RNG seed for reproducible shuffling.
            shuffle: Whether to shuffle samples before partitioning.

        Returns:
            List of generated split file paths, one per part.

        Example:
            >>> dataset = YOLOPoseDataset("train.txt")
            >>> split_files = dataset.export_dataset_partitions(
            ...     part_count=5,
            ...     output_root_dir="./dataset_parts",
            ...     shuffle=False,
            ... )
            >>> print(split_files)
        """
        from .operations import export_dataset_partitions

        return export_dataset_partitions(
            self,
            part_count=part_count,
            output_root_dir=output_root_dir,
            seed=seed,
            shuffle=shuffle,
        )

    def remove_duplicates(
        self,
        duplicates: Sequence[Sequence[Any]],
        output_dataset_path: Optional[str] = None,
        keep: str = 'first',
    ) -> Dict[str, Any]:
        """Remove duplicate samples described by ``find_near_duplicates`` output.

        Args:
            duplicates: Iterable of duplicate tuples like
                ``(file_a, file_b, similarity)``.
            output_dataset_path: Optional destination for a cleaned dataset copy.
                When omitted, the current split file is rewritten in place.
            keep: Which sample to keep from each duplicate group: ``'first'`` or
                ``'last'`` according to current dataset order.

        Returns:
            Summary dictionary with removed and kept sample paths.
        """
        from .operations import remove_duplicate_samples_from_dataset

        return remove_duplicate_samples_from_dataset(
            self,
            duplicates=duplicates,
            output_dataset_path=output_dataset_path,
            keep=keep,
        )

    def remove_samples_by_image_paths(
        self,
        image_paths: Sequence[str],
        delete_files: bool = True,
    ) -> Dict[str, Any]:
        """Remove samples identified by absolute or split-relative image paths.

        The split file is rewritten in place. When ``delete_files`` is true,
        matching images and their label files are also deleted from disk.

        Args:
            image_paths: Iterable of image paths to remove.
            delete_files: Whether to delete image and label files from disk.

        Returns:
            Summary dictionary with removal counts and unmatched paths.
        """
        from .operations import remove_samples_by_image_paths

        return remove_samples_by_image_paths(
            self,
            image_paths=image_paths,
            delete_files=delete_files,
        )