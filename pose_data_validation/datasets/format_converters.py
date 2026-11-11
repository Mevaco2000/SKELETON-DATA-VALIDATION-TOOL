"""Converters from custom keypoint dataset formats to YOLO Pose format."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List, Optional

import numpy as np

from .dataset_formats import FORMAT_REGISTRY, KeypointAnnotation, bbox_from_keypoints
from .operations import generate_train_txt_from_images
from .yolo_pose_dataset import mirrored_label_rel_path


def _ensure_yolo_bbox(annotation: KeypointAnnotation) -> List[float]:
    bbox = annotation.bbox_xywh if annotation.bbox_xywh is not None else bbox_from_keypoints(annotation.keypoints)
    return [float(v) for v in bbox]


def _annotation_to_yolo_line(annotation: KeypointAnnotation) -> str:
    class_id = int(annotation.class_id)
    bbox = _ensure_yolo_bbox(annotation)

    keypoints = np.asarray(annotation.keypoints, dtype=float)
    if keypoints.size == 0:
        return " ".join([str(class_id)] + [f"{v:.6f}" for v in bbox])

    if keypoints.ndim != 2 or keypoints.shape[1] not in (2, 3):
        raise ValueError(f"Invalid keypoints shape {keypoints.shape}; expected (N,2) or (N,3)")

    coords = np.clip(keypoints[:, :2], 0.0, 1.0)
    if keypoints.shape[1] == 3:
        visibility = keypoints[:, 2:3]
    else:
        visibility = np.full((coords.shape[0], 1), 2.0, dtype=float)

    keypoints_xyv = np.hstack([coords, visibility]).reshape(-1)

    serialized = [str(class_id)]
    serialized.extend(f"{float(v):.6f}" for v in bbox)
    serialized.extend(f"{float(v):.6f}" for v in keypoints_xyv)
    return " ".join(serialized)


def convert_registered_format_to_yolo_pose(
    format_name: str,
    input_root: str,
    output_root: str,
    split_name: str = "train",
    copy_images: bool = True,
    generate_split_file: bool = True,
) -> Dict[str, object]:
    """Convert a registered format to YOLO Pose directory structure.

    Output layout:
      output_root/images/...  (only with ``copy_images=True``)
      output_root/labels/...
      output_root/<split_name>.txt (optional)

    Labels are placed where ``YOLOPoseDataset`` looks for them: with copied
    images at ``labels/<mirrored_label_rel_path>`` (``images/train/x.jpg`` ->
    ``labels/train/x.txt``); with ``copy_images=False`` the split file lists the
    absolute source image paths and labels are flat, ``labels/<name>.txt``.

    Raises:
        ValueError: When two images would get the same label file.
    """
    format_impl = FORMAT_REGISTRY.get(format_name)

    input_root_abs = os.path.abspath(input_root)
    output_root_abs = os.path.abspath(output_root)
    images_root = os.path.join(output_root_abs, "images")
    labels_root = os.path.join(output_root_abs, "labels")
    if copy_images:
        os.makedirs(images_root, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)

    converted_images = 0
    converted_annotations = 0
    skipped_missing_images = 0
    split_entries: List[str] = []
    image_by_label_key: Dict[str, str] = {}

    for sample in format_impl.iter_samples(input_root_abs):
        rel_image_path = os.path.normpath(sample.relative_image_path.replace('\\', '/')).replace('\\', '/')
        source_image_path = sample.image_path

        if not os.path.isfile(source_image_path):
            skipped_missing_images += 1
            continue

        if os.path.isabs(rel_image_path) or rel_image_path == ".." or rel_image_path.startswith("../"):
            raise ValueError(
                f"Image path '{sample.relative_image_path}' leaves the dataset root; cannot convert it."
            )

        if copy_images:
            label_rel_path = mirrored_label_rel_path(rel_image_path)
        else:
            label_rel_path = os.path.splitext(os.path.basename(rel_image_path))[0] + ".txt"
        destination_label_path = os.path.join(labels_root, label_rel_path)
        label_key = os.path.normcase(os.path.abspath(destination_label_path))
        if label_key in image_by_label_key:
            raise ValueError(
                f"Images '{image_by_label_key[label_key]}' and '{rel_image_path}' would share the label "
                f"file '{destination_label_path}'."
                + ("" if copy_images else " Use copy_images=True to keep the folder layout.")
            )
        image_by_label_key[label_key] = rel_image_path
        os.makedirs(os.path.dirname(destination_label_path), exist_ok=True)

        if copy_images:
            destination_image_path = os.path.join(output_root_abs, rel_image_path)
            os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
            shutil.copy2(source_image_path, destination_image_path)
            split_entries.append(rel_image_path)
        else:
            split_entries.append(os.path.abspath(source_image_path).replace('\\', '/'))

        lines: List[str] = []
        for ann in sample.annotations:
            lines.append(_annotation_to_yolo_line(ann))
            converted_annotations += 1

        with open(destination_label_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))

        converted_images += 1

    split_path: Optional[str] = None
    if generate_split_file:
        if not split_entries:
            raise ValueError(f"No images were converted from: {input_root_abs}")
        # Written from the converted samples rather than by scanning images/,
        # so it also works without copies and for paths outside images/.
        split_path = os.path.join(output_root_abs, f"{split_name}.txt")
        with open(split_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(split_entries) + "\n")

    return {
        "format": format_name,
        "input_root": input_root_abs,
        "output_root": output_root_abs,
        "split_file": split_path,
        "converted_images": converted_images,
        "converted_annotations": converted_annotations,
        "skipped_missing_images": skipped_missing_images,
    }


def convert_coco_keypoints_with_ultralytics(
    coco_annotations_dir: str,
    output_root: str,
    copy_images_from: Optional[str] = None,
    split_name: str = "train",
) -> Dict[str, object]:
    """Use Ultralytics COCO converter for keypoints when available.

    This wraps Ultralytics conversion utility and then optionally copies images
    into ``output_root/images`` and regenerates ``split_name`` txt file.
    """
    try:
        from ultralytics.data.converter import convert_coco
    except Exception as exc:
        raise ImportError(
            "Ultralytics converter is unavailable. Install ultralytics to use this path."
        ) from exc

    coco_annotations_dir_abs = os.path.abspath(coco_annotations_dir)
    output_root_abs = os.path.abspath(output_root)
    labels_dst_abs = os.path.join(output_root_abs, "labels")

    os.makedirs(output_root_abs, exist_ok=True)

    # convert_coco renames a save_dir that already exists (X -> X2) and writes
    # labels to save_dir/labels/<json name>/<file_name>.txt. Convert into a
    # staging folder that does not exist yet, then move the labels to
    # output_root/labels/<file_name>.txt so they line up with output_root/images.
    staging_root = tempfile.mkdtemp(prefix="coco_convert_", dir=output_root_abs)
    staging_save_dir = os.path.join(staging_root, "converted")
    source_json_by_label: Dict[str, str] = {}
    try:
        convert_coco(
            labels_dir=coco_annotations_dir_abs,
            save_dir=staging_save_dir,
            use_segments=False,
            use_keypoints=True,
        )
        if not os.path.isdir(staging_save_dir):
            raise RuntimeError(f"Ultralytics convert_coco did not write to {staging_save_dir}")

        staged_labels_root = os.path.join(staging_save_dir, "labels")
        if os.path.isdir(staged_labels_root):
            for json_dir_name in sorted(os.listdir(staged_labels_root)):
                json_labels_dir = os.path.join(staged_labels_root, json_dir_name)
                if not os.path.isdir(json_labels_dir):
                    continue
                for root, _, files in os.walk(json_labels_dir):
                    for file_name in files:
                        if not file_name.lower().endswith(".txt"):
                            continue
                        src = os.path.join(root, file_name)
                        dst = os.path.join(labels_dst_abs, os.path.relpath(src, json_labels_dir))
                        dst_key = os.path.normcase(dst)
                        if dst_key in source_json_by_label:
                            raise ValueError(
                                f"Annotation files '{source_json_by_label[dst_key]}' and '{json_dir_name}' "
                                f"both label '{os.path.relpath(src, json_labels_dir)}'; convert them separately."
                            )
                        source_json_by_label[dst_key] = json_dir_name
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        os.replace(src, dst)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    copied_images = 0
    if copy_images_from is not None:
        images_src_abs = os.path.abspath(copy_images_from)
        images_dst_abs = os.path.join(output_root_abs, "images")
        os.makedirs(images_dst_abs, exist_ok=True)

        for root, _, files in os.walk(images_src_abs):
            for file_name in files:
                extension = os.path.splitext(file_name)[1].lower()
                if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                    continue
                src = os.path.join(root, file_name)
                rel = os.path.relpath(src, images_src_abs)
                dst = os.path.join(images_dst_abs, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                copied_images += 1

    split_path = generate_train_txt_from_images(
        dataset_root=output_root_abs,
        images_dir="images",
        output_file=os.path.join(output_root_abs, f"{split_name}.txt"),
        recursive=True,
    )

    return {
        "converter": "ultralytics.convert_coco",
        "annotations_dir": coco_annotations_dir_abs,
        "output_root": output_root_abs,
        "converted_labels": len(source_json_by_label),
        "copied_images": copied_images,
        "split_file": split_path,
    }
