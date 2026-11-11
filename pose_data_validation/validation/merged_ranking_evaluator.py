"""Lightweight container for validation experiments and ad-hoc checks."""

import ast
import os
import re
from collections.abc import Mapping
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


class MergedRankingEvaluator:
    def __init__(
        self,
        validator,
        report_paths,
        top_k=450,
        output_path=None,
        distance_model_name: str = "random_forest",
        distance_visibility_threshold: float = 1.2,
        distance_threshold_percentile: float = 95.0,
        distance_sort_by: str = "mean",
        distance_connections: Optional[Iterable[Tuple[int, int]]] = None,
        segmentation_model_name: Optional[str] = "yolo26m-seg",
        segmentation_target_class: Union[str, int] = "person",
        segmentation_score_threshold: float = 0.5,
        segmentation_mask_tolerance_px: int = 12,
    ):
        self.validator = validator
        self.report_paths = report_paths
        self.top_k = int(top_k)
        self.output_path = output_path
        self.distance_model_name = str(distance_model_name)
        self.distance_visibility_threshold = float(distance_visibility_threshold)
        self.distance_threshold_percentile = float(distance_threshold_percentile)
        self.distance_sort_by = str(distance_sort_by)
        self.distance_connections = list(distance_connections) if distance_connections is not None else None
        self.segmentation_model_name = segmentation_model_name
        self.segmentation_target_class = segmentation_target_class
        self.segmentation_score_threshold = float(segmentation_score_threshold)
        self.segmentation_mask_tolerance_px = int(segmentation_mask_tolerance_px)
        self._prepared = False
        self._prepared_components = set()

    @staticmethod
    def _is_segmentation_disabled(segmentation_model_name: Optional[str]) -> bool:
        if segmentation_model_name is None:
            return True
        normalized = str(segmentation_model_name).strip().lower()
        return normalized in {"", "none", "off", "disabled", "no"}

    def _resolved_segmentation_model_name(self) -> Optional[str]:
        if self._is_segmentation_disabled(self.segmentation_model_name):
            return None
        return str(self.segmentation_model_name)

    def _load_reference_report_entries(self):
        loaded = {name: parse_results_txt(path) for name, path in self.report_paths.items()}
        source_report_names = [name for name in loaded.keys() if name != "merged"]
        if not source_report_names and "merged" in loaded:
            source_report_names = ["merged"]

        merged = []
        for report_name in source_report_names:
            merged.extend(loaded.get(report_name, []))
        loaded["merged"] = merged
        return loaded

    def _normalize_ranked_entries(self, rank_source):
        extracted_entries = YPValidation_Test._extract_distance_anomaly_entries(rank_source)
        return YPValidation_Test._normalize_distance_anomaly_entries(extracted_entries)

    def _sample_key(self, image_path, person_id):
        return (os.path.normcase(str(image_path)), int(person_id))

    def _infer_unique_sample_key_by_image(self, image_path):
        """Resolve a unique (image_path, person_id) key when person_id is missing.

        Distance-only anomaly outputs may provide only image_path. In such cases,
        map the entry to dataset records by image path and use it only when the
        image contains exactly one person.
        """
        if image_path is None:
            return None

        normalized_path = os.path.normcase(str(image_path))
        candidate_keys = [
            key
            for key in getattr(self, "all_samples", {}).keys()
            if os.path.normcase(str(key[0])) == normalized_path
        ]
        if len(candidate_keys) == 1:
            return candidate_keys[0]
        return None

    def _reverse_position_powered(self, rank, rank_total):
        if rank is None or rank_total is None or rank_total <= 0:
            return 0.0
        reverse_position = max(0.0, float(rank_total) - float(rank) + 1.0)
        return reverse_position ** 2

    def _build_rank_lookup(self, rank_source):
        lookup = {}
        entries = self._normalize_ranked_entries(rank_source)
        for entry in entries:
            person_id = entry.get("person_id")
            if person_id is None:
                person_id = entry.get("person_index")
            if person_id is None:
                person_id = entry.get("person_idx")
            if person_id is None:
                inferred_key = self._infer_unique_sample_key_by_image(entry.get("image_path"))
                if inferred_key is None:
                    continue
                lookup[inferred_key] = dict(entry)
            else:
                lookup[self._sample_key(entry.get("image_path"), person_id)] = dict(entry)
        return lookup, len(entries)

    def _build_outside_lookup(self, raw_outside_keypoints):
        if isinstance(raw_outside_keypoints, dict):
            outside_entries = raw_outside_keypoints.get("outside_masks")
            if outside_entries is None:
                outside_entries = raw_outside_keypoints.get("outside_keypoints")
            if outside_entries is None:
                outside_entries = raw_outside_keypoints.get("records", [])
        elif isinstance(raw_outside_keypoints, list):
            outside_entries = raw_outside_keypoints
        else:
            outside_entries = []

        grouped = {}
        for entry in outside_entries:
            person_id = entry.get("person_id")
            if person_id is None:
                person_id = entry.get("person_index")
            if person_id is None:
                person_id = entry.get("person_idx")
            if person_id is None:
                continue

            key = self._sample_key(entry.get("image_path"), person_id)
            group = grouped.setdefault(
                key,
                {
                    "image_path": entry.get("image_path"),
                    "person_id": int(person_id),
                    "outside_mask_detected": True,
                    "outside_keypoint_count": 0,
                    "outside_keypoint_indices": [],
                },
            )

            keypoint_index = entry.get("keypoint_index")
            if keypoint_index is not None and keypoint_index not in group["outside_keypoint_indices"]:
                group["outside_keypoint_indices"].append(int(keypoint_index))

            group["outside_keypoint_count"] = len(group["outside_keypoint_indices"])

        return grouped

    def _summarize_segmentation_masks(self, raw_masks):
        stats = {
            "segmentation_total_images": None,
            "segmentation_images_with_mask": None,
            "segmentation_images_without_mask": None,
        }
        if not isinstance(raw_masks, dict):
            return stats

        mask_list = raw_masks.get("masks")
        if not isinstance(mask_list, list):
            return stats

        total_images = len(mask_list)
        images_with_mask = 0
        for mask in mask_list:
            try:
                if np.any(np.asarray(mask) > 0):
                    images_with_mask += 1
            except Exception:
                continue

        stats["segmentation_total_images"] = int(total_images)
        stats["segmentation_images_with_mask"] = int(images_with_mask)
        stats["segmentation_images_without_mask"] = int(max(total_images - images_with_mask, 0))
        return stats

    def _build_reference_key_set(self, reference_entries):
        key_set = set()
        for entry in reference_entries:
            person_id = YPValidation_Test._extract_person_identifier(entry)
            image_path = entry.get("image_path")
            if image_path is None or person_id is None:
                continue
            key_set.add(self._sample_key(image_path, person_id))
        return key_set

    def _build_reference_image_path_set(self, reference_entries):
        image_path_set = set()
        for entry in reference_entries:
            image_path = entry.get("image_path")
            if image_path is None:
                continue
            image_path_set.add(os.path.normcase(str(image_path)))
        return image_path_set

    def _build_reference_image_name_set(self, reference_entries):
        image_name_set = set()
        for entry in reference_entries:
            image_path = entry.get("image_path")
            if image_path is None:
                continue
            image_name_set.add(os.path.basename(os.path.normcase(str(image_path))))
        return image_name_set

    def _compare_reference_entries_to_anomalies(self, reference_entries, ranked_source):
        ranked_entries = self._normalize_ranked_entries(ranked_source)
        _, matches = YPValidation_Test._match_reference_entries(reference_entries, ranked_entries, self.top_k)

        def matched_ranks(position):
            return [
                int(match_tuple[position]["rank"])
                for match_tuple in matches
                if match_tuple[position] is not None and match_tuple[position].get("rank") is not None
            ]

        overall_ranks = matched_ranks(1)
        top_ranks = matched_ranks(2)
        return {
            "matched_top_k_count": len(top_ranks),
            "total_reference_entries": len(reference_entries),
            "matched_overall_count": len(overall_ranks),
            "avg_rank_all_reference": sum(overall_ranks) / len(overall_ranks) if overall_ranks else None,
            "last_matched_rank_in_top_k": max(top_ranks) if top_ranks else None,
        }

    def _summarize_comparison(self, comparison):
        avg_rank = comparison["avg_rank_all_reference"]
        avg_rank_text = f"{avg_rank:.2f}" if avg_rank is not None else "brak"
        return (
            f"{comparison['matched_top_k_count']}/{comparison['total_reference_entries']} | "
            f"overall={comparison['matched_overall_count']}/{comparison['total_reference_entries']} | "
            f"avg_rank_all={avg_rank_text}"
        )

    def _set_segmentation_disabled_results(self):
        self.segmentation_masks = {
            "masks": [],
            "image_paths": [],
            "model_name": "none",
            "model_family": "disabled",
            "target_class_id": None,
            "target_class_name": None,
            "score_threshold": None,
        }
        self.outside_keypoints = {
            "outside_masks": [],
            "without_masks": [],
            "outside_persons": [],
            "without_mask_persons": [],
            "outside_person_count": 0,
            "without_mask_person_count": 0,
            "mask_tolerance_px": int(self.segmentation_mask_tolerance_px),
        }
        self.results_mask_distance = []

    def _compute_segmentation_results(self):
        masks = self.validator.segment_dataset_with_model(
            model_name=self._resolved_segmentation_model_name(),
            target_class=self.segmentation_target_class,
            score_threshold=self.segmentation_score_threshold,
        )
        masks["score_threshold"] = float(self.segmentation_score_threshold)
        self.segmentation_masks = masks

        results_mask = self.validator.evaluate_dataset_keypoints_against_masks(
            masks["masks"],
            masks["image_paths"],
            mask_tolerance_px=int(self.segmentation_mask_tolerance_px),
        )
        self.outside_keypoints = self.validator.collect_visible_keypoints_outside_masks(results_mask)
        self.outside_keypoints["mask_tolerance_px"] = int(self.segmentation_mask_tolerance_px)
        self.results_mask_distance = self.validator.build_keypoint_mask_distance_datasets(results_mask, variant=0)

    def _refresh_segmentation_lookups(self):
        self.segmentation_stats = self._summarize_segmentation_masks(self.segmentation_masks)
        self.mask_lookup, self.mask_total = self._build_rank_lookup(self.results_mask_distance)
        self.outside_lookup = self._build_outside_lookup(self.outside_keypoints)
        self.outside_person_total = int(
            (self.outside_keypoints or {}).get("outside_person_count", len(self.outside_lookup))
        )
        self.segmentation_target_class_name = (self.segmentation_masks or {}).get("target_class_name")
        self.segmentation_target_class_id = (self.segmentation_masks or {}).get("target_class_id")
        self.segmentation_model_name_resolved = (self.segmentation_masks or {}).get(
            "model_name",
            self.segmentation_model_name,
        )

    def _prepare(self, include_distance: bool = True, include_lbp: bool = True, include_segmentation: bool = True):
        """Compute the ranking inputs that are still missing.

        Each component is computed at most once and then kept. A later call that
        includes a component skipped earlier computes it then, so the result of
        ``run()`` does not depend on the weights of the first call.
        """
        if not self._prepared:
            self.reference_report_entries = self._load_reference_report_entries()
            self.merged_reference_keys = self._build_reference_key_set(self.reference_report_entries["merged"])
            self.merged_reference_image_paths = self._build_reference_image_path_set(self.reference_report_entries["merged"])
            self.merged_reference_image_names = self._build_reference_image_name_set(self.reference_report_entries["merged"])

            # all_samples must exist before any rank lookup is built, because
            # entries without person_id are resolved through it.
            self.all_samples = {}
            for sample in self.validator.dataset:
                image_path = sample.get("image_path")
                person_keypoints = self.validator._get_sample_person_keypoints(sample)
                for person_id, _ in enumerate(person_keypoints):
                    self.all_samples[self._sample_key(image_path, person_id)] = {
                        "image_path": image_path,
                        "person_id": int(person_id),
                    }

            self.results_distance = []
            self.results_lbp = []
            self.distance_lookup, self.distance_total = self._build_rank_lookup(self.results_distance)
            self.lbp_lookup, self.lbp_total = self._build_rank_lookup(self.results_lbp)
            self._set_segmentation_disabled_results()
            self._refresh_segmentation_lookups()
            self._prepared_components = set()
            self._prepared = True

        if include_distance and "distance" not in self._prepared_components:
            distance_model, valid_indices = self.validator.train_distance_models(
                self.distance_model_name,
                distance_connections=self.distance_connections,
                visibility_threshold=self.distance_visibility_threshold,
            )
            self.results_distance = self.validator.predict_distance_anomalies(
                distance_model,
                valid_indices,
                distance_connections=self.distance_connections,
                visibility_threshold=self.distance_visibility_threshold,
                threshold_percentile=self.distance_threshold_percentile,
                sort_by=self.distance_sort_by,
            )
            self.distance_lookup, self.distance_total = self._build_rank_lookup(self.results_distance)
            self._prepared_components.add("distance")

        if include_lbp and "lbp" not in self._prepared_components:
            self.results_lbp = self.validator.predict_lbp_anomalies_by_embedding_groups(patch_size=32)
            self.lbp_lookup, self.lbp_total = self._build_rank_lookup(self.results_lbp)
            self._prepared_components.add("lbp")

        if (
            include_segmentation
            and not self._is_segmentation_disabled(self.segmentation_model_name)
            and "segmentation" not in self._prepared_components
        ):
            self._compute_segmentation_results()
            self._refresh_segmentation_lookups()
            self._prepared_components.add("segmentation")

    def _build_combined_ranking(self, distance_weight, lbp_weight, segmentation_weight):
        if self._is_segmentation_disabled(self.segmentation_model_name):
            segmentation_weight = 0.0
        weight_sum = float(distance_weight) + float(lbp_weight) + float(segmentation_weight)
        if weight_sum <= 0.0:
            print("Suma wag musi byc dodatnia.")
            return [], {}

        outside_keys = {
            key
            for key, meta in self.outside_lookup.items()
            if bool(meta.get("outside_mask_detected", False))
        }
        model_mentioned_keys = set(self.distance_lookup) | set(self.lbp_lookup) | set(self.mask_lookup)

        def build_record(key, stage_name):
            sample_record = self.all_samples.get(key)
            if sample_record is None:
                return None

            outside_record = self.outside_lookup.get(key, {})
            distance_rank = self.distance_lookup.get(key, {}).get("rank")
            lbp_rank = self.lbp_lookup.get(key, {}).get("rank")
            mask_rank = self.mask_lookup.get(key, {}).get("rank")

            distance_score = self._reverse_position_powered(distance_rank, self.distance_total)
            lbp_score = self._reverse_position_powered(lbp_rank, self.lbp_total)
            mask_score = self._reverse_position_powered(mask_rank, self.mask_total)
            weighted_score = (
                distance_weight * distance_score
                + lbp_weight * lbp_score
                + segmentation_weight * mask_score
            ) / weight_sum

            return {
                "image_path": sample_record["image_path"],
                "person_id": int(sample_record["person_id"]),
                "outside_mask_detected": bool(outside_record),
                "outside_keypoint_count": int(outside_record.get("outside_keypoint_count", 0)),
                "distance_powered_score": float(distance_score),
                "lbp_powered_score": float(lbp_score),
                "mask_powered_score": float(mask_score),
                "weighted_score": float(weighted_score),
                "stage": stage_name,
            }

        combined = []
        for key in sorted(set(self.all_samples)):
            record = build_record(key, "ranked_by_weighted_score")
            if record is not None:
                combined.append(record)

        combined.sort(
            key=lambda record: (
                -record["weighted_score"],
                str(record["image_path"]),
                int(record["person_id"]),
            )
        )

        promoted_outside = [
            record
            for record in combined
            if bool(record.get("outside_mask_detected", False))
        ]
        ranked_rest = [
            record
            for record in combined
            if not bool(record.get("outside_mask_detected", False))
        ]
        for record in promoted_outside:
            record["stage"] = "promoted_outside_mask"
        combined = promoted_outside + ranked_rest

        for rank, record in enumerate(combined, start=1):
            record["rank"] = int(rank)

        outside_mentioned_person = len(outside_keys & self.merged_reference_keys)
        outside_image_paths = {os.path.normcase(str(key[0])) for key in outside_keys}
        outside_mentioned_image = len(outside_image_paths & self.merged_reference_image_paths)
        outside_image_names = {os.path.basename(path) for path in outside_image_paths}
        outside_mentioned_name = len(outside_image_names & self.merged_reference_image_names)
        outside_mentioned_in_merged = (
            outside_mentioned_person
            if outside_mentioned_person > 0
            else (outside_mentioned_image if outside_mentioned_image > 0 else outside_mentioned_name)
        )

        meta = {
            "promoted_outside_count": len(promoted_outside),
            "ranked_after_outside_count": len(combined),
            "outside_total": len(outside_keys),
            "outside_person_total": int(getattr(self, "outside_person_total", len(outside_keys))),
            "outside_in_3_models": len(outside_keys & model_mentioned_keys),
            "outside_mentioned_in_merged": int(outside_mentioned_in_merged),
            "segmentation_model_name": self.segmentation_model_name_resolved,
            "segmentation_target_class_name": self.segmentation_target_class_name,
            "segmentation_target_class_id": self.segmentation_target_class_id,
            "segmentation_mask_tolerance_px": int(self.segmentation_mask_tolerance_px),
            # Number of ranking records produced by mask-distance pipeline.
            "mask_ranked_records_total": int(self.mask_total),
            # Real segmentation outcomes based on binary masks.
            "segmentation_total_images": self.segmentation_stats.get("segmentation_total_images"),
            "segmentation_images_with_mask": self.segmentation_stats.get("segmentation_images_with_mask"),
            "segmentation_images_without_mask": self.segmentation_stats.get("segmentation_images_without_mask"),
        }
        return combined, meta

    def run(self, distance_weight, lbp_weight, segmentation_weight):
        # Segmentation is prepared even at weight 0: _build_combined_ranking always
        # promotes persons with keypoints outside the mask to the top. Disable it
        # with segmentation_model_name=None.
        self._prepare(
            include_distance=float(distance_weight) > 0.0,
            include_lbp=float(lbp_weight) > 0.0,
            include_segmentation=True,
        )
        combined_ranking, ranking_meta = self._build_combined_ranking(
            distance_weight,
            lbp_weight,
            segmentation_weight,
        )
        if not combined_ranking:
            return None

        # Recompute outside_mentioned_in_merged excluding entries already captured by top-k,
        # so that matched_top_k and outside_mentioned_in_merged are always disjoint.
        # Use anchored identifiers (path from images/ onward) so reference reports stored in a
        # different directory than the dataset still match (same rule as report comparisons).
        def _anchored_key(image_path, person_id):
            normalized_person_id = YPValidation_Test._extract_person_identifier({"person_id": person_id})
            return (
                YPValidation_Test._normalize_results_identifier_path(image_path),
                normalized_person_id,
            )

        top_k_keys = {
            _anchored_key(entry["image_path"], entry["person_id"])
            for entry in combined_ranking
            if entry.get("rank", float("inf")) <= self.top_k
        }
        top_k_sample_keys = {
            self._sample_key(entry["image_path"], entry["person_id"])
            for entry in combined_ranking
            if entry.get("rank", float("inf")) <= self.top_k
        }
        outside_sample_keys = {
            self._sample_key(meta.get("image_path"), meta.get("person_id"))
            for meta in self.outside_lookup.values()
            if bool(meta.get("outside_mask_detected", False))
            and meta.get("image_path") is not None
            and meta.get("person_id") is not None
        }
        ranking_meta["outside_persons_in_top_k"] = len(outside_sample_keys & top_k_sample_keys)
        reference_keys_anchored = {
            _anchored_key(entry.get("image_path"), YPValidation_Test._extract_person_identifier(entry))
            for entry in self.reference_report_entries["merged"]
            if entry.get("image_path") is not None
        }
        outside_keys_in_reference = {
            _anchored_key(meta.get("image_path"), meta.get("person_id"))
            for meta in self.outside_lookup.values()
            if bool(meta.get("outside_mask_detected", False))
        } & reference_keys_anchored
        ranking_meta["outside_mentioned_in_merged"] = len(outside_keys_in_reference - top_k_keys)

        report_summaries = {}
        report_details = {}
        lines = [
            f"REPORT: single_run | weights={float(distance_weight):.3f}/{float(lbp_weight):.3f}/{float(segmentation_weight):.3f}",
            (
                f"ranking_meta: promoted_outside={ranking_meta['promoted_outside_count']} | "
                f"ranked_after_outside={ranking_meta['ranked_after_outside_count']} | "
                f"outside_person_total={ranking_meta['outside_person_total']} | "
                f"outside_persons_in_top_k={ranking_meta['outside_persons_in_top_k']} | "
                f"outside_in_3_models={ranking_meta['outside_in_3_models']} | "
                f"outside_mentioned_in_merged={ranking_meta['outside_mentioned_in_merged']} | "
                f"seg_model={ranking_meta['segmentation_model_name']} | "
                f"seg_target={ranking_meta['segmentation_target_class_name']}({ranking_meta['segmentation_target_class_id']}) | "
                f"seg_mask_tolerance_px={ranking_meta['segmentation_mask_tolerance_px']} | "
                f"mask_ranked_records_total={ranking_meta['mask_ranked_records_total']} | "
                f"segmentation_images_with_mask={ranking_meta['segmentation_images_with_mask']} | "
                f"segmentation_images_without_mask={ranking_meta['segmentation_images_without_mask']}"
            ),
        ]

        for report_name, report_entries in self.reference_report_entries.items():
            comparison = self._compare_reference_entries_to_anomalies(report_entries, combined_ranking)
            report_details[report_name] = comparison
            report_summaries[report_name] = self._summarize_comparison(comparison)
            lines.append(f"{report_name}: {report_summaries[report_name]}")

        final_output_path = self.output_path
        if final_output_path:
            with open(final_output_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("\n".join(lines) + "\n")

        merged_summary = report_summaries.get("merged", "N/A")
        other_report_parts = [
            f"{name}={summary}"
            for name, summary in report_summaries.items()
            if name != "merged"
        ]

        summary_text = (
            f"weights={float(distance_weight):.3f}/{float(lbp_weight):.3f}/{float(segmentation_weight):.3f} | "
            f"merged={merged_summary}"
        )
        if other_report_parts:
            summary_text += " | " + " | ".join(other_report_parts)

        print(summary_text)
        print(
            f"CACHE READY | all_samples={len(self.all_samples)} | distance={self.distance_total} | "
            f"lbp={self.lbp_total} | mask_ranked_records={self.mask_total} | "
            f"segmentation_with_mask={ranking_meta.get('segmentation_images_with_mask')} | "
            f"segmentation_without_mask={ranking_meta.get('segmentation_images_without_mask')} | "
            f"outside={len(self.outside_lookup)} | "
            f"outside_person_total={ranking_meta.get('outside_person_total')} | "
            f"outside_persons_in_top_k={ranking_meta.get('outside_persons_in_top_k')} | "
            f"outside_mentioned_in_merged={ranking_meta['outside_mentioned_in_merged']}"
        )

        return {
            "weights": (float(distance_weight), float(lbp_weight), float(segmentation_weight)),
            "output_path": final_output_path,
            "ranking_meta": ranking_meta,
            "combined_ranking": combined_ranking,
            "report_summaries": report_summaries,
            "report_details": report_details,
        }

class YPValidation_Test:
    """Store and summarize results from manual validation tests.

    This class is intentionally small. It provides a stable place for
    temporary validation logic without coupling it to YPImageValidation or
    YPSetValidation.
    """

    @staticmethod
    def parse_results_txt_value(raw_value: str) -> Any:
        """Convert a raw results.txt field to a Python object when possible."""
        value = raw_value.strip()
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    # Console line of rigid / axis-aligned results: "<image_path> <person_id> <extra>",
    # where extra is the rotation angle or the list of changed keypoint indices.
    _CONSOLE_RESULT_LINE = re.compile(
        r"^(?P<path>.+?\.(?:jpe?g|png|bmp|tiff?|webp))\s+(?P<person_id>-?\d+)(?:\s+(?P<extra>.+))?$",
        re.IGNORECASE,
    )
    _DETAILS_HEADER_LINE = re.compile(r"^--- details \d+ ---$")
    _DETAILS_FIELD_LINE = re.compile(r"^(?P<key>[A-Za-z_]\w*): (?P<value>.*)$")

    @classmethod
    def parse_results_txt_line(cls, line: str) -> Dict[str, Any]:
        """Parse a single line from a validation results.txt file."""
        stripped = line.strip()
        if not stripped:
            raise ValueError("Empty results.txt line cannot be parsed")

        console_match = cls._CONSOLE_RESULT_LINE.match(stripped)
        if console_match and " | " not in stripped and not stripped.startswith("{"):
            parsed_console: Dict[str, Any] = {
                "image_path": console_match.group("path"),
                "person_id": int(console_match.group("person_id")),
            }
            extra = console_match.group("extra")
            if extra is not None:
                extra_value = cls.parse_results_txt_value(extra)
                if isinstance(extra_value, (list, tuple)):
                    parsed_console["changed_keypoint_indices"] = list(extra_value)
                elif isinstance(extra_value, (int, float)):
                    parsed_console["rotation_angle_degrees"] = float(extra_value)
                else:
                    parsed_console["extra"] = extra_value
            return parsed_console

        parts = [part.strip() for part in stripped.split(" | ") if part.strip()]
        parsed: Dict[str, Any] = {}

        first_part = parts[0]
        if first_part.startswith("{") and first_part.endswith("}"):
            parsed.update(cls.parse_results_txt_value(first_part))
            parts = parts[1:]

        for part in parts:
            if ": " not in part:
                continue
            key, raw_value = part.split(": ", 1)
            parsed[key.strip()] = cls.parse_results_txt_value(raw_value)

        if not parsed:
            raise ValueError(f"Could not parse results.txt line: {line[:120]}")

        return parsed

    @classmethod
    def parse_results_txt(cls, results_txt_path: str) -> List[Dict[str, Any]]:
        """Parse a validation results.txt file into structured records."""
        parsed_entries: List[Dict[str, Any]] = []
        in_details_block = False

        with open(results_txt_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                # Multi-line details written by save_results_console_output:
                # "--- details N ---" followed by "key: value" lines that belong
                # to the entry above them.
                if cls._DETAILS_HEADER_LINE.match(stripped):
                    in_details_block = bool(parsed_entries)
                    continue
                if in_details_block:
                    # A " | " line is a one-line record of its own, not a detail field.
                    field_match = None if " | " in stripped else cls._DETAILS_FIELD_LINE.match(stripped)
                    if field_match:
                        parsed_entries[-1][field_match.group("key")] = cls.parse_results_txt_value(
                            field_match.group("value")
                        )
                        continue
                    in_details_block = False

                try:
                    entry = cls.parse_results_txt_line(stripped)
                except ValueError as exc:
                    raise ValueError(
                        f"Failed to parse {results_txt_path} at line {line_number}: {exc}"
                    ) from exc

                entry["line_number"] = line_number
                parsed_entries.append(entry)

        return parsed_entries

    @staticmethod
    def _normalize_results_identifier_path(path: Any) -> Any:
        """Normalize a stored path for stable file matching."""
        if path is None:
            return None

        normalized_path = os.path.normcase(os.path.normpath(str(path))).replace('\\', '/')

        for anchor in ('/images/', '/labels/'):
            anchor_position = normalized_path.rfind(anchor)
            if anchor_position != -1:
                return normalized_path[anchor_position + 1:]

        return normalized_path

    @staticmethod
    def _extract_person_identifier(entry: Mapping[str, Any]) -> Any:
        """Return a normalized person identifier from supported field names."""
        person_identifier = entry.get("person_id")
        if person_identifier is None:
            person_identifier = entry.get("person_index")
        if person_identifier is None:
            person_identifier = entry.get("person_idx")

        if person_identifier is None:
            return None

        try:
            return int(person_identifier)
        except (TypeError, ValueError):
            return person_identifier

    @classmethod
    def _distance_anomaly_entry_identifier(
        cls,
        entry: Dict[str, Any],
        match_mode: str,
    ) -> Any:
        """Return the identifier used when matching ranked anomaly entries."""
        image_path = cls._normalize_results_identifier_path(entry.get("image_path"))
        if match_mode == "image_path_person_id":
            return image_path, cls._extract_person_identifier(entry)
        return image_path

    @classmethod
    def _match_reference_entries(
        cls,
        reference_entries: List[Dict[str, Any]],
        ranked_entries: List[Dict[str, Any]],
        top_k: int,
        match_mode: str = "auto",
    ) -> Tuple[str, List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]]:
        """Pair each reference entry with its first unused match in the ranking and in its top ``top_k``.

        ``match_mode="auto"`` matches on (image path, person id) when both sides
        carry person ids, otherwise on image path. Returns the resolved match mode
        and one ``(reference_entry, report_match, top_k_match)`` tuple per
        reference entry, with ``None`` where nothing matched.
        """
        if match_mode == "auto":
            def has_person_ids(entries):
                return bool(entries) and all(cls._extract_person_identifier(entry) is not None for entry in entries)

            match_mode = (
                "image_path_person_id"
                if has_person_ids(ranked_entries) and has_person_ids(reference_entries)
                else "image_path"
            )

        ranked_entries_sorted = sorted(ranked_entries, key=lambda entry: entry.get("rank", float("inf")))
        report_buckets: Dict[Any, List[Dict[str, Any]]] = {}
        top_buckets: Dict[Any, List[Dict[str, Any]]] = {}
        for position, entry in enumerate(ranked_entries_sorted):
            identifier = cls._distance_anomaly_entry_identifier(entry, match_mode)
            report_buckets.setdefault(identifier, []).append(entry)
            if position < top_k:
                top_buckets.setdefault(identifier, []).append(entry)

        matches = []
        for reference_entry in reference_entries:
            identifier = cls._distance_anomaly_entry_identifier(reference_entry, match_mode)
            report_bucket = report_buckets.get(identifier)
            top_bucket = top_buckets.get(identifier)
            matches.append((
                reference_entry,
                report_bucket.pop(0) if report_bucket else None,
                top_bucket.pop(0) if top_bucket else None,
            ))
        return match_mode, matches

    @staticmethod
    def _extract_distance_anomaly_entries(raw_object: Any) -> Optional[List[Dict[str, Any]]]:
        """Return a list of anomaly entries when a supported object shape is provided."""
        if isinstance(raw_object, dict):
            if isinstance(raw_object.get("images"), list):
                return [dict(entry) for entry in raw_object["images"] if isinstance(entry, dict)]
            if isinstance(raw_object.get("records"), list):
                return [dict(entry) for entry in raw_object["records"] if isinstance(entry, dict)]
            if "image_path" in raw_object:
                return [dict(raw_object)]

        if isinstance(raw_object, list):
            if all(isinstance(entry, dict) for entry in raw_object):
                return [dict(entry) for entry in raw_object]

        return None

    @classmethod
    def _normalize_distance_anomaly_entries(
        cls,
        entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize ranked anomaly entries to a consistent serializable schema."""
        normalized_entries: List[Dict[str, Any]] = []

        for fallback_rank, entry in enumerate(entries, start=1):
            normalized_entry = dict(entry)
            normalized_entry["image_path"] = entry.get("image_path")
            normalized_entry["person_id"] = cls._extract_person_identifier(entry)

            if normalized_entry.get("rank") is None:
                normalized_entry["rank"] = fallback_rank
            else:
                try:
                    normalized_entry["rank"] = int(normalized_entry["rank"])
                except (TypeError, ValueError):
                    normalized_entry["rank"] = fallback_rank

            if entry.get("person_index") is not None:
                try:
                    normalized_entry["person_index"] = int(entry.get("person_index"))
                except (TypeError, ValueError):
                    normalized_entry["person_index"] = entry.get("person_index")

            if entry.get("person_idx") is not None:
                try:
                    normalized_entry["person_idx"] = int(entry.get("person_idx"))
                except (TypeError, ValueError):
                    normalized_entry["person_idx"] = entry.get("person_idx")

            if normalized_entry.get("error") is not None:
                try:
                    normalized_entry["error"] = float(normalized_entry["error"])
                except (TypeError, ValueError):
                    pass

            for score_key in ("max_anomaly_score", "mean_anomaly_score", "sort_score"):
                if normalized_entry.get(score_key) is None:
                    continue
                try:
                    normalized_entry[score_key] = float(normalized_entry[score_key])
                except (TypeError, ValueError):
                    pass

            if normalized_entry.get("is_anomaly") is not None:
                normalized_entry["is_anomaly"] = bool(normalized_entry["is_anomaly"])

            normalized_entries.append(normalized_entry)

        normalized_entries.sort(key=lambda entry: entry.get("rank", float("inf")))
        for fallback_rank, entry in enumerate(normalized_entries, start=1):
            entry["rank"] = fallback_rank if entry.get("rank") is None else int(entry["rank"])

        return normalized_entries

    @classmethod
    def parse_distance_anomaly_report(cls, distance_anomaly_report_path: str) -> List[Dict[str, Any]]:
        """Parse a saved distance-anomaly ranking file into ranked entries.

        Supported formats:
        - a Python literal dict containing an ``images`` list
        - a Python literal list of dicts
        - one dict per line
        - one key-value line per entry in the style ``key: value | key: value``
        """
        with open(distance_anomaly_report_path, "r", encoding="utf-8") as file_handle:
            raw_text = file_handle.read().strip()

        if not raw_text:
            return []

        try:
            parsed_object = ast.literal_eval(raw_text)
        except (ValueError, SyntaxError):
            parsed_object = None

        extracted_entries = cls._extract_distance_anomaly_entries(parsed_object)
        if extracted_entries is not None:
            return cls._normalize_distance_anomaly_entries(extracted_entries)

        parsed_entries: List[Dict[str, Any]] = []
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                if stripped.startswith("{") and stripped.endswith("}"):
                    parsed_line = ast.literal_eval(stripped)
                    if not isinstance(parsed_line, dict):
                        raise ValueError("Parsed line is not a dictionary")
                else:
                    parsed_line = cls.parse_results_txt_line(stripped)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    f"Failed to parse distance anomaly report {distance_anomaly_report_path} at line {line_number}: {exc}"
                ) from exc

            parsed_line["line_number"] = line_number
            parsed_entries.append(parsed_line)

        return cls._normalize_distance_anomaly_entries(parsed_entries)

    @classmethod
    def save_distance_anomaly_report(
        cls,
        distance_anomaly_results: Dict[str, Any],
        output_path: str,
        only_anomalies: bool = False,
    ) -> str:
        """Save ranked distance-anomaly results to a line-based text file.

        The saved format is designed to be parseable by
        ``parse_distance_anomaly_report()`` and stable across notebooks.
        """
        entries = cls._extract_distance_anomaly_entries(distance_anomaly_results)
        if entries is None:
            raise TypeError(
                "distance_anomaly_results must contain an 'images' list, a 'records' list, "
                "a single record dictionary, or be a list of dictionaries"
            )

        normalized_entries = cls._normalize_distance_anomaly_entries(entries)

        output_path_abs = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path_abs), exist_ok=True)

        serialized_lines: List[str] = []
        for fallback_rank, image_entry in enumerate(normalized_entries, start=1):
            if only_anomalies and not bool(image_entry.get("is_anomaly")):
                continue

            person_identifier = cls._extract_person_identifier(image_entry)
            error_value = image_entry.get("error")
            if error_value is None:
                error_value = image_entry.get("sort_score")
            if error_value is None:
                error_value = image_entry.get("max_anomaly_score")
            if error_value is None:
                error_value = image_entry.get("mean_anomaly_score")

            serialized_error = None
            if error_value is not None:
                serialized_error = float(error_value)
                if not np.isfinite(serialized_error):
                    serialized_error = None

            serialized_entry = {
                "rank": int(image_entry.get("rank", fallback_rank)),
                "image_path": image_entry.get("image_path"),
                "person_id": person_identifier,
                "error": serialized_error,
                "is_anomaly": bool(image_entry.get("is_anomaly")),
            }
            if image_entry.get("person_index") is not None:
                serialized_entry["person_index"] = image_entry.get("person_index")
            if image_entry.get("person_idx") is not None:
                serialized_entry["person_idx"] = image_entry.get("person_idx")
            if image_entry.get("max_anomaly_score") is not None:
                max_anomaly_score = float(image_entry["max_anomaly_score"])
                serialized_entry["max_anomaly_score"] = (
                    max_anomaly_score if np.isfinite(max_anomaly_score) else None
                )
            if image_entry.get("mean_anomaly_score") is not None:
                mean_anomaly_score = float(image_entry["mean_anomaly_score"])
                serialized_entry["mean_anomaly_score"] = (
                    mean_anomaly_score if np.isfinite(mean_anomaly_score) else None
                )
            serialized_lines.append(repr(serialized_entry))

        with open(output_path_abs, "w", encoding="utf-8") as file_handle:
            file_handle.write("\n".join(serialized_lines) + ("\n" if serialized_lines else ""))

        return output_path_abs

    @classmethod
    def compare_results_txt_to_distance_anomalies(
        cls,
        reference_results_txt_path: str,
        distance_anomaly_report_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        top_k: int,
        match_mode: str = "auto",
        include_reference_fields: bool = False,
        reference_field_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare a reference ``results.txt`` file with a ranked anomaly report.

        Args:
            reference_results_txt_path: Path to the reference ``results.txt`` file.
            distance_anomaly_report_path: Path to a saved anomaly ranking or an
                in-memory ranking object compatible with ``parse_distance_anomaly_report``.
            top_k: Number of top-ranked anomaly entries to inspect.
            match_mode: ``auto``, ``image_path``, or ``image_path_person_id``.
            include_reference_fields: When ``True``, add per-reference entries
                with selected fields from ``results.txt`` and flags describing
                whether the entry was found in the report and inside ``top_k``.
            reference_field_names: Optional subset of ``results.txt`` fields to
                include when ``include_reference_fields`` is enabled. When omitted,
                all reference fields except identifiers are included.

        Returns:
            Serializable dictionary containing the overlap summary and the exact
            matched entries. The ``summary_text`` field contains a ready-to-print
            sentence in Polish.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if match_mode not in {"auto", "image_path", "image_path_person_id"}:
            raise ValueError("match_mode must be 'auto', 'image_path', or 'image_path_person_id'")

        reference_entries = cls.parse_results_txt(reference_results_txt_path)
        ranked_entries: List[Dict[str, Any]]
        if isinstance(distance_anomaly_report_path, (str, os.PathLike)):
            ranked_entries = cls.parse_distance_anomaly_report(os.fspath(distance_anomaly_report_path))
            ranked_report_reference = os.path.abspath(os.fspath(distance_anomaly_report_path))
        else:
            extracted_entries = cls._extract_distance_anomaly_entries(distance_anomaly_report_path)
            if extracted_entries is None:
                raise TypeError(
                    "distance_anomaly_report_path must be a path or an anomaly result object "
                    "containing 'images'/'records' entries"
                )
            ranked_entries = cls._normalize_distance_anomaly_entries(extracted_entries)
            ranked_report_reference = "<in-memory>"

        resolved_match_mode, matches = cls._match_reference_entries(
            reference_entries, ranked_entries, top_k, match_mode
        )

        requested_reference_fields = list(reference_field_names or [])
        reference_report_entries: List[Dict[str, Any]] = []
        matched_top_entries: List[Dict[str, Any]] = []
        matched_overall_entries: List[Dict[str, Any]] = []

        for reference_entry, report_match, top_match in matches:
            extracted_entry = {
                "image_path": reference_entry.get("image_path"),
                "person_id": cls._extract_person_identifier(reference_entry),
                "found_in_report": report_match is not None,
                "found_in_top_k": top_match is not None,
                "report_rank": report_match.get("rank") if report_match is not None else None,
                "report_error": report_match.get("error") if report_match is not None else None,
                "report_is_anomaly": report_match.get("is_anomaly") if report_match is not None else None,
                "reference_line_number": reference_entry.get("line_number"),
            }

            if requested_reference_fields:
                for field_name in requested_reference_fields:
                    extracted_entry[field_name] = reference_entry.get(field_name)
            else:
                for field_name, field_value in reference_entry.items():
                    if field_name in {"image_path", "person_id", "line_number"}:
                        continue
                    extracted_entry[field_name] = field_value

            reference_report_entries.append(extracted_entry)

            if top_match is not None:
                matched_top_entries.append(
                    {
                        "rank": top_match.get("rank"),
                        "image_path": reference_entry.get("image_path"),
                        "person_id": cls._extract_person_identifier(reference_entry),
                        "error": top_match.get("error"),
                        "is_anomaly": top_match.get("is_anomaly"),
                    }
                )

            if report_match is not None:
                matched_overall_entries.append(
                    {
                        "rank": report_match.get("rank"),
                        "image_path": reference_entry.get("image_path"),
                        "person_id": cls._extract_person_identifier(reference_entry),
                        "error": report_match.get("error"),
                        "is_anomaly": report_match.get("is_anomaly"),
                    }
                )

        matched_top_count = len(matched_top_entries)
        last_matched_rank_in_top_k = max(
            (int(entry["rank"]) for entry in matched_top_entries if entry.get("rank") is not None),
            default=None,
        )
        best_rank_overall = min(
            (int(entry["rank"]) for entry in matched_overall_entries if entry.get("rank") is not None),
            default=None,
        )
        worst_rank_overall = max(
            (int(entry["rank"]) for entry in matched_overall_entries if entry.get("rank") is not None),
            default=None,
        )
        matched_overall_count = len(matched_overall_entries)
        avg_rank_all_reference = (
            sum(int(entry["rank"]) for entry in matched_overall_entries if entry.get("rank") is not None)
            / matched_overall_count
            if matched_overall_count > 0
            else None
        )

        if last_matched_rank_in_top_k is None:
            summary_text = (
                f"W top {top_k} pozycji rankingu anomalii znajduje się 0 poprawnych plików "
                f"(plików z results.txt); żaden z nich nie występuje w badanym zakresie."
            )
        else:
            avg_str = f"{avg_rank_all_reference:.1f}" if avg_rank_all_reference is not None else "brak"
            summary_text = (
                f"W top {top_k} pozycji rankingu anomalii znajduje się {matched_top_count} poprawnych plików "
                f"(plików z results.txt), a ostatni z nich zajmuje pozycję {last_matched_rank_in_top_k}. "
                f"Średnia pozycja wszystkich poszukiwanych: {avg_str}."
            )

        comparison = {
            "reference_results_txt_path": os.path.abspath(reference_results_txt_path),
            "distance_anomaly_report_path": ranked_report_reference,
            "top_k": int(top_k),
            "match_mode": resolved_match_mode,
            "total_reference_entries": len(reference_entries),
            "total_ranked_entries": len(ranked_entries),
            "matched_top_k_count": matched_top_count,
            "last_matched_rank_in_top_k": last_matched_rank_in_top_k,
            "matched_top_k_entries": matched_top_entries,
            "matched_overall_count": matched_overall_count,
            "avg_rank_all_reference": avg_rank_all_reference,
            "best_rank_overall": best_rank_overall,
            "worst_rank_overall": worst_rank_overall,
            "matched_overall_entries": matched_overall_entries,
            "summary_text": summary_text,
        }

        if include_reference_fields:
            comparison["reference_report_entries"] = reference_report_entries
            comparison["reference_report_found_count"] = sum(
                1 for entry in reference_report_entries if entry["found_in_report"]
            )
            comparison["reference_top_k_found_count"] = sum(
                1 for entry in reference_report_entries if entry["found_in_top_k"]
            )

        return comparison

    @classmethod
    def compare_results_txt_to_distance_anomalies_with_outside_keypoints(
        cls,
        reference_results_txt_path: str,
        distance_anomaly_report_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        outside_keypoint_records: Union[List[Dict[str, Any]], Dict[str, Any]],
        top_k: int,
        match_mode: str = "auto",
    ) -> Dict[str, Any]:
        """Compare top-k anomaly matches and then add matches from outside-mask keypoints.

        This variant preserves the default ranking-based comparison and only adds
        supplemental matches for persons present in ``outside_keypoint_records``
        that were not already matched within ``top_k``.
        """
        base_comparison = cls.compare_results_txt_to_distance_anomalies(
            reference_results_txt_path=reference_results_txt_path,
            distance_anomaly_report_path=distance_anomaly_report_path,
            top_k=top_k,
            match_mode=match_mode,
        )

        if isinstance(outside_keypoint_records, dict):
            if isinstance(outside_keypoint_records.get("outside_keypoints"), list):
                raw_outside_entries = outside_keypoint_records["outside_keypoints"]
            elif isinstance(outside_keypoint_records.get("outside_masks"), list):
                # Compatibility with collect_visible_keypoints_outside_masks output.
                raw_outside_entries = outside_keypoint_records["outside_masks"]
            elif isinstance(outside_keypoint_records.get("records"), list):
                raw_outside_entries = outside_keypoint_records["records"]
            else:
                raise TypeError(
                    "outside_keypoint_records must be a list of dictionaries or contain "
                    "an 'outside_keypoints'/'outside_masks'/'records' list"
                )
        elif isinstance(outside_keypoint_records, list):
            raw_outside_entries = outside_keypoint_records
        else:
            raise TypeError(
                "outside_keypoint_records must be a list of dictionaries or a dictionary"
            )

        reference_entries = cls.parse_results_txt(reference_results_txt_path)
        resolved_match_mode = str(base_comparison["match_mode"])
        reference_identifiers = [
            cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            for entry in reference_entries
        ]
        remaining_counts = Counter(reference_identifiers)

        for entry in base_comparison.get("matched_top_k_entries", []):
            identifier = cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            if remaining_counts[identifier] > 0:
                remaining_counts[identifier] -= 1

        grouped_outside_entries: Dict[Any, Dict[str, Any]] = {}
        for entry in raw_outside_entries:
            if not isinstance(entry, dict):
                continue

            identifier = cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            if identifier is None:
                continue

            grouped_entry = grouped_outside_entries.get(identifier)
            if grouped_entry is None:
                grouped_entry = {
                    "image_path": entry.get("image_path"),
                    "person_id": cls._extract_person_identifier(entry),
                    "outside_keypoint_indices": [],
                    "outside_keypoint_count": 0,
                }
                grouped_outside_entries[identifier] = grouped_entry

            keypoint_index = entry.get("keypoint_index")
            if keypoint_index is not None:
                try:
                    normalized_index = int(keypoint_index)
                except (TypeError, ValueError):
                    normalized_index = keypoint_index
                if normalized_index not in grouped_entry["outside_keypoint_indices"]:
                    grouped_entry["outside_keypoint_indices"].append(normalized_index)
            grouped_entry["outside_keypoint_count"] = len(grouped_entry["outside_keypoint_indices"])

        supplemental_outside_matches: List[Dict[str, Any]] = []
        for identifier, entry in grouped_outside_entries.items():
            if remaining_counts[identifier] <= 0:
                continue
            remaining_counts[identifier] -= 1
            supplemental_outside_matches.append(entry)

        supplemental_outside_matches.sort(
            key=lambda entry: (
                -int(entry.get("outside_keypoint_count", 0)),
                str(entry.get("image_path")),
                str(entry.get("person_id")),
            )
        )

        matched_top_k_plus_outside_count = (
            int(base_comparison.get("matched_top_k_count", 0))
            + len(supplemental_outside_matches)
        )

        summary_text = str(base_comparison.get("summary_text", ""))
        if supplemental_outside_matches:
            summary_text += (
                f" Po doliczeniu {len(supplemental_outside_matches)} rekordów z visible keypoints "
                f"outside mask liczba trafień rośnie do {matched_top_k_plus_outside_count}."
            )

        summary_text += (
            f" Łącznie wykryto {matched_top_k_plus_outside_count} plików z "
            f"{len(reference_entries)} plików referencyjnych."
        )

        comparison = dict(base_comparison)
        comparison["matched_top_k_plus_outside_keypoints_count"] = matched_top_k_plus_outside_count
        comparison["detected_files_count"] = matched_top_k_plus_outside_count
        comparison["reference_files_count"] = len(reference_entries)
        comparison["supplemental_outside_keypoint_matches"] = supplemental_outside_matches
        comparison["summary_text"] = summary_text
        return comparison



def parse_results_txt(results_txt_path: str) -> List[Dict[str, Any]]:
    """Module-level wrapper for YPValidation_Test.parse_results_txt."""
    return YPValidation_Test.parse_results_txt(results_txt_path)


def parse_distance_anomaly_report(distance_anomaly_report_path: str) -> List[Dict[str, Any]]:
    """Module-level wrapper for YPValidation_Test.parse_distance_anomaly_report."""
    return YPValidation_Test.parse_distance_anomaly_report(distance_anomaly_report_path)


def save_distance_anomaly_report(
    distance_anomaly_results: Dict[str, Any],
    output_path: str,
    only_anomalies: bool = False,
) -> str:
    """Module-level wrapper for YPValidation_Test.save_distance_anomaly_report."""
    return YPValidation_Test.save_distance_anomaly_report(
        distance_anomaly_results,
        output_path,
        only_anomalies=only_anomalies,
    )


def compare_results_txt_to_distance_anomalies(
    reference_results_txt_path: str,
    distance_anomaly_report_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    top_k: int,
    match_mode: str = "auto",
    include_reference_fields: bool = False,
    reference_field_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Module-level wrapper for YPValidation_Test.compare_results_txt_to_distance_anomalies."""
    return YPValidation_Test.compare_results_txt_to_distance_anomalies(
        reference_results_txt_path,
        distance_anomaly_report_path,
        top_k,
        match_mode=match_mode,
        include_reference_fields=include_reference_fields,
        reference_field_names=reference_field_names,
    )


def compare_results_txt_to_distance_anomalies_with_outside_keypoints(
    reference_results_txt_path: str,
    distance_anomaly_report_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    outside_keypoint_records: Union[List[Dict[str, Any]], Dict[str, Any]],
    top_k: int,
    match_mode: str = "auto",
) -> Dict[str, Any]:
    """Module-level wrapper for the outside-keypoint-aware anomaly comparison."""
    return YPValidation_Test.compare_results_txt_to_distance_anomalies_with_outside_keypoints(
        reference_results_txt_path,
        distance_anomaly_report_path,
        outside_keypoint_records,
        top_k,
        match_mode=match_mode,
    )


