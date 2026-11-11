
import os
from typing import Union, List, Optional, Sequence, Tuple
import numpy as np
import cv2


def _normalize_keypoint_format(keypoint_format: str) -> str:
    """Validate and normalize keypoint format selector."""
    normalized_format = keypoint_format.lower().strip()
    if normalized_format not in {"auto", "xy", "xyv"}:
        raise ValueError("keypoint_format must be one of: 'auto', 'xy', 'xyv'")
    return normalized_format


def _looks_like_visibility_values(values: List[float]) -> bool:
    """Return whether provided values resemble YOLO visibility flags."""
    return all(abs(value - round(value)) < 1e-6 and int(round(value)) in {0, 1, 2} for value in values)


def _resolve_keypoint_format(values: List[float], keypoint_format: str) -> str:
    """Resolve the keypoint serialization format for one YOLO label row."""
    normalized_format = _normalize_keypoint_format(keypoint_format)
    if normalized_format != "auto":
        return normalized_format

    value_count = len(values)
    if value_count == 0:
        return "xyv"

    supports_xyv = value_count % 3 == 0
    supports_xy = value_count % 2 == 0

    if supports_xyv and not supports_xy:
        return "xyv"
    if supports_xy and not supports_xyv:
        return "xy"
    if supports_xyv and supports_xy:
        visibility_values = values[2::3]
        return "xyv" if _looks_like_visibility_values(visibility_values) else "xy"

    raise ValueError(
        "Could not infer keypoint format from label row. "
        "Use keypoint_format='xy' or keypoint_format='xyv'."
    )


def _parse_yolo_keypoint_values(values: List[float], include_visibility: bool, keypoint_format: str) -> np.ndarray:
    """Parse YOLO keypoint values encoded as either xy or xyv."""
    resolved_format = _resolve_keypoint_format(values, keypoint_format)
    step = 3 if resolved_format == "xyv" else 2
    keypoints = []

    for value_index in range(0, len(values), step):
        if resolved_format == "xyv":
            if value_index + 2 >= len(values):
                continue
            if include_visibility:
                keypoints.append([
                    values[value_index],
                    values[value_index + 1],
                    values[value_index + 2],
                ])
            else:
                keypoints.append([values[value_index], values[value_index + 1]])
        else:
            if value_index + 1 >= len(values):
                continue
            keypoints.append([values[value_index], values[value_index + 1]])

    if not keypoints:
        column_count = 3 if include_visibility and resolved_format == "xyv" else 2
        return np.empty((0, column_count), dtype=float)

    return np.asarray(keypoints, dtype=float)


def _parse_yolo_pose_label_line(
    line: str,
    include_visibility: bool,
    keypoint_format: str,
) -> np.ndarray:
    """Parse one YOLO pose label line into keypoints array."""
    parts = line.split()
    values = list(map(float, parts[5:]))  # Skip class_id, x, y, width, height
    return _parse_yolo_keypoint_values(values, include_visibility, keypoint_format)


def labelled_keypoint_mask(keypoints: np.ndarray) -> np.ndarray:
    """Return a boolean mask of keypoints that carry a labelled position.

    YOLO pose labels store unlabelled keypoints as ``(0, 0)`` / ``(0, 0, 0)`` or
    with visibility 0; such points have no position to move, draw or compare.
    """
    keypoints = np.asarray(keypoints, dtype=float)
    if keypoints.ndim != 2 or keypoints.shape[0] == 0:
        return np.zeros(len(keypoints), dtype=bool)
    labelled = ~np.all(np.isclose(keypoints[:, :2], 0.0), axis=1)
    if keypoints.shape[1] >= 3:
        labelled &= keypoints[:, 2] > 0
    return labelled


def normalize_distance_connections(
    distance_connections: Optional[Sequence[Tuple[int, int]]]
) -> Optional[List[Tuple[int, int]]]:
    """Validate and normalize an optional list of keypoint index pairs."""
    if distance_connections is None:
        return None

    normalized_connections: List[Tuple[int, int]] = []
    for connection_index, connection in enumerate(distance_connections):
        if len(connection) != 2:
            raise ValueError(
                "Each distance connection must contain exactly two keypoint indices; "
                f"got {connection!r} at position {connection_index}."
            )

        start_idx = int(connection[0])
        end_idx = int(connection[1])
        if start_idx < 0 or end_idx < 0:
            raise ValueError(
                "Distance connection indices must be non-negative; "
                f"got {connection!r} at position {connection_index}."
            )
        if start_idx == end_idx:
            raise ValueError(
                "Distance connection cannot reference the same keypoint twice; "
                f"got {connection!r} at position {connection_index}."
            )

        normalized_connections.append((start_idx, end_idx))

    if not normalized_connections:
        raise ValueError("distance_connections cannot be empty")

    return normalized_connections


class YPImageValidation:
    """
    Validation methods for a single image with annotations.
    
    Handles LBP computation and binary decimal representations
    for individual images with keypoint annotations.
    
    Use this class for per-image or per-keypoint feature extraction.
    Use YPSetValidation for dataset-level operations.
    
    USAGE PATTERNS:
    
    Example 1 - Compute LBP for a single image:
        >>> from pose_data_validation.validation import YPImageValidation
        >>> import cv2
        >>> 
        >>> image = cv2.imread('image.jpg')
        >>> keypoints = np.array([[100, 50], [150, 200]])  # (N, 2)
        >>> 
        >>> validator = YPImageValidation(image, keypoints)
        >>> decimals = validator.compute_lbp_decimals(patch_size=7)  # (N,)
    
    Example 2 - Process individual keypoints:
        >>> validator = YPImageValidation(image, keypoints)
        >>> for i, (x, y) in enumerate(keypoints):
        ...     decimal = validator.compute_lbp_decimal_single(int(x), int(y))
    
    Example 3 - Visualize single image:
        >>> validator = YPImageValidation(image, keypoints)
        >>> result = validator.visualize(patch_size=5)
    """
    
    def __init__(self, image: Union[np.ndarray, str], keypoints: np.ndarray):
        """
        Initialize image validator.
        
        Args:
            image: Image as numpy array (H x W or H x W x 3) or path to image file
            keypoints: Array of keypoints with shape (N, 2) with [x, y] OR (N, 3) with [x, y, visibility].
                      If (N, 3), visibility is checked when filtering keypoints.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.image = image
        self.h, self.w = image.shape[:2]
        self.keypoints = np.asarray(keypoints, dtype=float).copy()
        
        # Keep original (normalized) keypoints for distance computation.
        # Distances are scale-independent and must use the 0-1 YOLO coordinate space.
        self.keypoints_norm = self.keypoints.copy()
        
        # Denormalize keypoints to pixel coordinates for LBP (needs real image pixels).
        # Check only x, y columns (columns 0-1), not visibility if present
        if self.keypoints.size > 0 and self.keypoints[:, :2].max() <= 1.0:
            self.keypoints[:, 0] *= self.w
            self.keypoints[:, 1] *= self.h
    
    def sequential_distances(
        self,
        visibility_threshold: float = 2.0,
        distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Calculate Euclidean distances between visible keypoint pairs.
        
        By default computes the distance from keypoint[i] to keypoint[i+1] for
        consecutive visible pairs. When ``distance_connections`` is provided,
        computes distances for those explicit keypoint pairs instead.
        
        Args:
            visibility_threshold: Minimum visibility score (0=out, 1=hidden, 2=visible).
                                Default 2.0 (only fully visible keypoints).
            distance_connections: Optional ordered list of ``(start_idx, end_idx)``
                                keypoint pairs to measure instead of consecutive pairs.
        
        Returns:
            Structured array with fields 'distance_index' (int) and 'distance' (float)
            for each valid measured pair. Distance indices always refer to the
            position in the active connection list. With the default behavior,
            this still matches ``distance[i] = keypoint[i] -> keypoint[i+1]``.
            
        Example:
            >>> validator = YPImageValidation(image, keypoints)
            >>> distances = validator.sequential_distances(visibility_threshold=2.0)
            >>> print(distances['distance_index'])  # [0, 1, 3, 5, ...] (absolute indices, may skip values if keypoints hidden)
            >>> print(distances['distance'])        # distance values
        """
        return YPImageValidation.compute_sequential_distances(
            self.keypoints_norm,
            visibility_threshold=visibility_threshold,
            distance_connections=distance_connections,
        )

    def sequential_angles(self, visibility_threshold: float = 2.0) -> np.ndarray:
        """Calculate angles of consecutive visible keypoint segments.

        Computes the orientation of the segment from keypoint[i] to keypoint[i+1]
        for consecutive visible pairs. Angles are reported in degrees using the
        standard mathematical convention: ``0`` points right, ``90`` points up,
        ``-90`` points down.

        Args:
            visibility_threshold: Minimum visibility score (0=out, 1=hidden, 2=visible).
                Default 2.0 (only fully visible keypoints).

        Returns:
            Structured array with fields ``angle_index`` (int) and ``angle`` (float)
            for each valid consecutive keypoint pair.
        """
        return YPImageValidation.compute_sequential_angles(
            self.keypoints_norm,
            visibility_threshold=visibility_threshold,
        )
    
    @staticmethod
    def compute_sequential_distances(
        keypoints: np.ndarray,
        visibility_threshold: float = 2.0,
        distance_connections: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Calculate Euclidean distances between visible keypoint pairs (static).
        
        Static method for computing default consecutive distances or distances for
        an explicit ordered list of keypoint pairs without creating an instance.
        
        Args:
            keypoints: Array of shape (N, 2) with [x, y] OR (N, 3) with [x, y, visibility].
            visibility_threshold: Minimum visibility score (0=out, 1=hidden, 2=visible).
                                Default 2.0 (only fully visible keypoints). Keypoints below this are skipped.
            distance_connections: Optional ordered list of ``(start_idx, end_idx)``
                                keypoint pairs to measure. If omitted, uses consecutive
                                pairs ``(0, 1), (1, 2), ...``.
        
        Returns:
            Structured array with fields 'distance_index' (int) and 'distance' (float)
            for each visible keypoint pair. Distance indices refer to the position
            within the active connection list.
            
        Example:
            >>> kpts = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
            >>> distances = YPImageValidation.compute_sequential_distances(kpts, visibility_threshold=2.0)
            >>> print(distances['distance_index'])  # [0, 2, ...] (absolute indices: skips 1 since keypoint[1] hidden)
            >>> print(distances['distance'])        # distance values
        """
        keypoints = np.asarray(keypoints)
        coords = keypoints[:, :2]
        normalized_connections = normalize_distance_connections(distance_connections)
        
        if keypoints.shape[1] >= 3:
            visibility = keypoints[:, 2]
            visible_mask = visibility >= visibility_threshold
        else:
            visible_mask = np.ones(len(keypoints), dtype=bool)

        if normalized_connections is None:
            normalized_connections = [(i, i + 1) for i in range(len(keypoints) - 1)]

        for connection_index, (start_idx, end_idx) in enumerate(normalized_connections):
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                raise ValueError(
                    "distance_connections contains an out-of-range keypoint index; "
                    f"got {(start_idx, end_idx)} at position {connection_index} for {len(keypoints)} keypoints."
                )
        
        distances = []
        distance_indices = []
        
        for connection_index, (start_idx, end_idx) in enumerate(normalized_connections):
            if visible_mask[start_idx] and visible_mask[end_idx]:
                dist = np.linalg.norm(coords[start_idx] - coords[end_idx])
                distances.append(dist)
                distance_indices.append(connection_index)
        
        if distances:
            result = np.array(
                list(zip(distance_indices, distances)),
                dtype=[('distance_index', 'i4'), ('distance', 'f8')]
            )
        else:
            result = np.array([], dtype=[('distance_index', 'i4'), ('distance', 'f8')])
        
        return result

    @staticmethod
    def compute_sequential_angles(keypoints: np.ndarray, visibility_threshold: float = 2.0) -> np.ndarray:
        """Calculate angles of consecutive visible keypoint segments (static).

        The angle is computed for the vector from keypoint[i] to keypoint[i+1].
        To preserve the standard mathematical sign convention on images, the
        y-axis delta is negated before calling ``arctan2``.

        Args:
            keypoints: Array of shape (N, 2) with [x, y] OR (N, 3) with
                [x, y, visibility].
            visibility_threshold: Minimum visibility score (0=out, 1=hidden, 2=visible).
                Default 2.0 (only fully visible keypoints). Keypoints below this are skipped.

        Returns:
            Structured array with fields ``angle_index`` (int) and ``angle`` (float)
            where ``angle_index`` is the absolute start index of the segment and
            ``angle`` is reported in degrees in the range ``[-180, 180]``.
        """
        keypoints = np.asarray(keypoints)
        coords = keypoints[:, :2]

        if keypoints.shape[1] >= 3:
            visibility = keypoints[:, 2]
            visible_mask = visibility >= visibility_threshold
        else:
            visible_mask = np.ones(len(keypoints), dtype=bool)

        angles = []
        angle_indices = []

        for i in range(len(keypoints) - 1):
            if visible_mask[i] and visible_mask[i + 1]:
                delta = coords[i + 1] - coords[i]
                angle = np.degrees(np.arctan2(-delta[1], delta[0]))
                angles.append(float(angle))
                angle_indices.append(i)

        if angles:
            result = np.array(
                list(zip(angle_indices, angles)),
                dtype=[('angle_index', 'i4'), ('angle', 'f8')]
            )
        else:
            result = np.array([], dtype=[('angle_index', 'i4'), ('angle', 'f8')])

        return result
    
    @staticmethod
    def compute_keypoint_mask_metrics(
        keypoints: np.ndarray,
        mask: np.ndarray,
        image_shape: tuple,
        visibility_threshold: Optional[float] = 2.0,
        mask_tolerance_px: int = 12,
    ) -> np.ndarray:
        """Compute mask-membership and horizontal edge distances for keypoints.

        For each keypoint, this method checks whether the point lies inside the
        binary mask and computes the horizontal distance to the left and right
        edges of the connected mask segment that contains the keypoint.

        Distances are normalized by the width of the local mask span in that row:
        ``row_width = right_edge - left_edge``. When no foreground exists in the
        keypoint row, the keypoint lies outside the image, or the keypoint is not
        inside the mask, normalized distances are returned as ``NaN``.

        Args:
            keypoints: Array of shape (N, 2) or (N, 3) with ``[x, y]`` or
                ``[x, y, visibility]``. Coordinates may be normalized or pixel-based.
            mask: Binary mask of shape ``(H, W)`` or multi-channel mask convertible
                to a foreground map via ``mask > 0``.
            image_shape: Image shape used to denormalize keypoints when needed.
            visibility_threshold: If provided and keypoints include visibility,
                keypoints below this threshold return ``inside_mask=0`` and
                ``NaN`` distances.
            mask_tolerance_px: Pixel tolerance used when checking if a keypoint
                is inside the mask. A value of 0 requires an exact pixel hit.
                Values > 0 mark a keypoint as inside when any foreground pixel
                exists in the local ``(2*t+1) x (2*t+1)`` neighborhood.

        Returns:
            Structured array with fields:
            - ``keypoint_index``: Original keypoint index
            - ``is_visible``: ``1`` if the keypoint passed visibility filtering else ``0``
            - ``inside_mask``: ``1`` if the keypoint lies inside the mask else ``0``
            - ``left_distance``: Normalized horizontal distance to left mask edge
            - ``right_distance``: Normalized horizontal distance to right mask edge
        """
        keypoints = np.asarray(keypoints, dtype=float)
        if keypoints.size == 0:
            return np.array(
                [],
                dtype=[
                    ('keypoint_index', 'i4'),
                    ('is_visible', 'i4'),
                    ('inside_mask', 'i4'),
                    ('left_distance', 'f8'),
                    ('right_distance', 'f8'),
                ],
            )

        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = np.any(mask_array > 0, axis=2)
        else:
            mask_array = mask_array > 0

        height, width = image_shape[:2]
        if mask_array.shape != (height, width):
            raise ValueError(
                f"Mask shape {mask_array.shape} does not match image shape {(height, width)}."
            )

        coords = keypoints[:, :2].copy()
        if coords.size > 0 and np.nanmax(coords) <= 1.0:
            coords[:, 0] *= width
            coords[:, 1] *= height

        tolerance_px = max(int(mask_tolerance_px), 0)

        results = []
        for keypoint_index, (x_coord, y_coord) in enumerate(coords):
            is_visible = 1
            if visibility_threshold is not None and keypoints.shape[1] >= 3:
                if keypoints[keypoint_index, 2] < visibility_threshold:
                    is_visible = 0
                    results.append((keypoint_index, is_visible, 0, np.nan, np.nan))
                    continue

            if not (0 <= x_coord < width and 0 <= y_coord < height):
                results.append((keypoint_index, is_visible, 0, np.nan, np.nan))
                continue

            pixel_x = int(np.clip(np.round(x_coord), 0, width - 1))
            pixel_y = int(np.clip(np.round(y_coord), 0, height - 1))
            if tolerance_px > 0:
                x_min = max(0, pixel_x - tolerance_px)
                x_max = min(width - 1, pixel_x + tolerance_px)
                y_min = max(0, pixel_y - tolerance_px)
                y_max = min(height - 1, pixel_y + tolerance_px)
                inside_mask = int(np.any(mask_array[y_min:y_max + 1, x_min:x_max + 1]))
            else:
                inside_mask = int(mask_array[pixel_y, pixel_x])

            row_y = pixel_y
            row_indices = np.flatnonzero(mask_array[row_y])
            if row_indices.size == 0 and tolerance_px > 0:
                for offset in range(1, tolerance_px + 1):
                    for candidate_y in (pixel_y - offset, pixel_y + offset):
                        if 0 <= candidate_y < height:
                            candidate_indices = np.flatnonzero(mask_array[candidate_y])
                            if candidate_indices.size > 0:
                                row_y = candidate_y
                                row_indices = candidate_indices
                                break
                    if row_indices.size > 0:
                        break

            if inside_mask:
                if row_indices.size == 0:
                    left_distance = np.nan
                    right_distance = np.nan
                    results.append((keypoint_index, is_visible, inside_mask, left_distance, right_distance))
                    continue

                if mask_array[row_y, pixel_x]:
                    seed_x = pixel_x
                else:
                    nearest_idx = int(np.argmin(np.abs(row_indices.astype(float) - float(x_coord))))
                    seed_x = int(row_indices[nearest_idx])

                left_idx = seed_x
                right_idx = seed_x

                while left_idx > 0 and mask_array[row_y, left_idx - 1]:
                    left_idx -= 1
                while right_idx < width - 1 and mask_array[row_y, right_idx + 1]:
                    right_idx += 1

                left_edge = float(left_idx)
                right_edge = float(right_idx)
                row_width = max(right_edge - left_edge, 1.0)
                left_distance = abs(float(x_coord) - left_edge) / row_width
                right_distance = abs(right_edge - float(x_coord)) / row_width
            else:
                left_distance = np.nan
                right_distance = np.nan

            results.append((keypoint_index, is_visible, inside_mask, left_distance, right_distance))

        return np.array(
            results,
            dtype=[
                ('keypoint_index', 'i4'),
                ('is_visible', 'i4'),
                ('inside_mask', 'i4'),
                ('left_distance', 'f8'),
                ('right_distance', 'f8'),
            ],
        )

    def evaluate_keypoints_against_mask(
        self,
        mask: np.ndarray,
        visibility_threshold: Optional[float] = 2.0,
        mask_tolerance_px: int = 12,
    ) -> np.ndarray:
        """Evaluate all image keypoints against a binary mask.

        Args:
            mask: Binary mask with the same height and width as the image.
            visibility_threshold: Optional visibility threshold. If provided,
                hidden keypoints return ``inside_mask=0`` and ``NaN`` distances.
            mask_tolerance_px: Pixel tolerance used for inside-mask checks.

        Returns:
            Structured array with ``keypoint_index``, ``is_visible``, ``inside_mask``,
            ``left_distance`` and ``right_distance``.
        """
        return YPImageValidation.compute_keypoint_mask_metrics(
            keypoints=self.keypoints_norm,
            mask=mask,
            image_shape=(self.h, self.w),
            visibility_threshold=visibility_threshold,
            mask_tolerance_px=mask_tolerance_px,
        )
    
    @staticmethod
    def _patch_to_binary_decimal(patch: np.ndarray, center_value: float = None) -> dict:
        """Convert patch to binary decimal representation."""
        if center_value is None:
            h, w = patch.shape
            center_idx_h, center_idx_w = h // 2, w // 2
            center_value = patch[center_idx_h, center_idx_w]
        
        binary_matrix = (patch >= center_value).astype(np.uint8)
        binary_string = ''.join(binary_matrix.flatten().astype(str))
        decimal_value = int(binary_string, 2)
        
        return {
            'binary_matrix': binary_matrix,
            'binary_string': binary_string,
            'decimal': decimal_value,
            'patch_shape': patch.shape
        }
    
    def compute_lbp_decimal_single(self, x: int, y: int, patch_size: int = 5) -> int:
        """
        Compute binary decimal representation for single keypoint.
        
        Args:
            x: X coordinate of keypoint
            y: Y coordinate of keypoint
            patch_size: Size of patch around keypoint
            
        Returns:
            Binary decimal value (arbitrary precision int)
        """
        half_size = patch_size // 2
        
        x_start = max(0, int(x) - half_size)
        y_start = max(0, int(y) - half_size)
        
        # For odd patch_size: include +1; for even: don't
        if patch_size % 2 == 1:
            x_end = min(self.w, int(x) + half_size + 1)
            y_end = min(self.h, int(y) + half_size + 1)
        else:
            x_end = min(self.w, int(x) + half_size)
            y_end = min(self.h, int(y) + half_size)
        
        patch = self.image[y_start:y_end, x_start:x_end].astype(np.uint8)
        
        if patch.size == 0:
            return np.uint64(0)
        
        # Handle undersized patches by padding (for edge cases)
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            pad_h = patch_size - patch.shape[0]
            pad_w = patch_size - patch.shape[1]
            patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='edge')
        
        result = self._patch_to_binary_decimal(patch)
        return int(result['decimal'])
    
    def compute_lbp_decimals(self, patch_size: int = 5, visibility_threshold: float = 2.0) -> np.ndarray:
        """
        Compute binary decimal representations for all visible keypoints in the image.
        
        Skips keypoints with visibility below threshold (hidden/occluded/out-of-frame keypoints).
        
        Args:
            patch_size: Size of patch around each keypoint
            visibility_threshold: Minimum visibility score (0=out, 1=hidden, 2=visible).
                                Default 2.0 (only fully visible keypoints).
            
        Returns:
            Array of shape (N_visible,) - arbitrary precision int values, one per visible keypoint.
            N_visible <= N if keypoints include visibility information.
        """
        if self.keypoints.size == 0:
            return np.array([], dtype=object)
        
        # Filter by visibility if available
        if self.keypoints.shape[1] >= 3:
            visibility = self.keypoints[:, 2]
            visible_indices = np.where(visibility >= visibility_threshold)[0]
            coords = self.keypoints[visible_indices, :2]
        else:
            coords = self.keypoints
        
        decimal_values = []
        for x, y in coords:
            decimal_values.append(self.compute_lbp_decimal_single(x, y, patch_size))
        
        return np.array(decimal_values, dtype=object)
    
    def draw_keypoints(self, color: tuple = (0, 255, 0), keypoint_size: int = 4) -> np.ndarray:
        """
        Draw keypoints as circles on image.
        
        Args:
            color: RGB color tuple (R, G, B) in range [0, 255].
                  Default (0, 255, 0) for green.
            keypoint_size: Radius of drawn keypoints in pixels.
        
        Returns:
            Image with keypoints drawn as circles
            
        Example:
            >>> validator = YPImageValidation(image, keypoints)
            >>> result = validator.draw_keypoints(color=(0, 255, 0), keypoint_size=4)
            >>> cv2.imshow("With keypoints", result)
        """
        return YPImageValidation.draw_keypoints_on_image(
            self.image, self.keypoints, color=color, keypoint_size=keypoint_size
        )
    
    @staticmethod
    def draw_keypoints_on_image(
        image: np.ndarray,
        keypoints: np.ndarray,
        color: tuple = (0, 255, 0),
        keypoint_size: int = 4,
    ) -> np.ndarray:
        """
        Draw keypoints as circles on image (static method).
        
        Static method for drawing keypoints without creating an instance.
        
        Args:
            image: Image as numpy array (H x W or H x W x 3)
            keypoints: Array of keypoints (N, 2) or (N, 3). Coordinates may be
                normalized (all values <= 1) or already in pixels.
            color: RGB color tuple (R, G, B) in range [0, 255]
            keypoint_size: Radius of drawn keypoints in pixels.

        Returns:
            Image with keypoints drawn

        Example:
            >>> result = YPImageValidation.draw_keypoints_on_image(
            ...     image, keypoints, color=(0, 255, 0), keypoint_size=4
            ... )
        """
        result = image.copy()
        h, w = image.shape[:2]

        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        for px, py in YPImageValidation._drawable_pixel_coords(keypoints, w, h):
            cv2.circle(result, (px, py), keypoint_size, color, -1)
        return result

    @staticmethod
    def _drawable_pixel_coords(keypoints: np.ndarray, width: int, height: int) -> List[Tuple[int, int]]:
        """Return pixel coordinates of labelled keypoints.

        Same normalization rule as ``__init__``: coordinates are scaled to
        pixels only when all of them are <= 1. Unlabelled ``(0, 0)`` keypoints
        and keypoints with visibility 0 are skipped.
        """
        keypoints = np.asarray(keypoints, dtype=float)
        if keypoints.size == 0:
            return []
        coords = keypoints[:, :2].copy()
        if coords.max() <= 1.0:
            coords[:, 0] *= width
            coords[:, 1] *= height
        labelled = labelled_keypoint_mask(keypoints)
        return [(int(x), int(y)) for (x, y), keep in zip(coords, labelled) if keep]
    
    def draw_keypoints_with_patches(self, patch_size: int = 32,
                                    keypoint_color: tuple = (0, 255, 0),
                                    patch_color: tuple = (255, 0, 0),
                                    thickness: int = 2) -> np.ndarray:
        """
        Draw keypoints and surrounding patch rectangles on image.
        
        For each keypoint, draws:
        1. A rectangle (patch_size × patch_size) centered on the keypoint
        2. A circle at the keypoint center
        
        Args:
            patch_size: Size of square patch to draw around each keypoint (pixels)
            keypoint_color: RGB color tuple for keypoint circles (default: green)
            patch_color: RGB color tuple for patch rectangles (default: blue)
            thickness: Line thickness for rectangles. -1 to fill.
        
        Returns:
            Image with keypoints and patches drawn
            
        Example:
            >>> validator = YPImageValidation(image, keypoints)
            >>> result = validator.draw_keypoints_with_patches(patch_size=32)
            >>> cv2.imshow("With patches", result)
        """
        import copy
        result = copy.copy(self.image)
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        half_patch = patch_size // 2

        # self.keypoints is already in pixels (see __init__).
        for px, py in self._drawable_pixel_coords(self.keypoints, self.w, self.h):
            top_left = (max(0, px - half_patch), max(0, py - half_patch))
            bottom_right = (min(self.w, px + half_patch), min(self.h, py + half_patch))
            cv2.rectangle(result, top_left, bottom_right, patch_color, thickness)
            
            cv2.circle(result, (px, py), 4, keypoint_color, -1)
        
        return result
    
    @classmethod
    def load_keypoints(
        cls,
        label_path: str,
        include_visibility: bool = True,
        keypoint_format: str = "auto",
    ) -> np.ndarray:
        """
        Load keypoints for the first person from a YOLO pose label file.
        
        Class method for loading keypoints without creating an instance.
        
        Args:
            label_path: Path to .txt label file in YOLO pose format
            include_visibility: If True, returns (N, 3) with visibility.
                              If False, returns (N, 2) only coordinates.
            keypoint_format: One of ``'auto'``, ``'xy'``, or ``'xyv'``.
                ``'auto'`` detects whether labels store point pairs or
                point-visibility triplets.
        
        Returns:
            Array of keypoints (N, 2) or (N, 3) for the first person only.
            Use load_all_keypoints_from_file() to process every person in the image.
            
        Example:
            >>> keypoints = YPImageValidation.load_keypoints("image.txt")
            >>> print(keypoints.shape)  # (9, 3) - 9 keypoints with visibility
        """
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        with open(label_path, 'r', encoding='utf-8-sig') as f:
            line = f.readline().strip()
        
        if not line:
            return np.array([])
        
        return _parse_yolo_pose_label_line(line, include_visibility, keypoint_format)
    
    @classmethod
    def load_all_keypoints_from_file(
        cls,
        label_path: str,
        include_visibility: bool = True,
        keypoint_format: str = "auto",
    ) -> List[np.ndarray]:
        """
        Load ALL keypoints from a YOLO pose label file (one person per line).
        
        Class method for loading multiple people from one label file.
        
        Args:
            label_path: Path to .txt label file in YOLO pose format
            include_visibility: If True, returns (N, 3) with visibility.
                              If False, returns (N, 2) only coordinates.
            keypoint_format: One of ``'auto'``, ``'xy'``, or ``'xyv'``.
                ``'auto'`` detects whether labels store point pairs or
                point-visibility triplets.
        
        Returns:
            List of keypoint arrays, one per person. Each array is (N_keypoints, 2) or (N_keypoints, 3)
            
        Example:
            >>> # Label file has 3 people
            >>> all_kpts = YPImageValidation.load_all_keypoints_from_file("image.txt")
            >>> print(len(all_kpts))  # 3
            >>> print(all_kpts[0].shape)  # (9, 3) - first person, 9 keypoints
        """
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        all_keypoints = []
        with open(label_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                keypoints = _parse_yolo_pose_label_line(line, include_visibility, keypoint_format)
                if keypoints.size > 0:
                    all_keypoints.append(keypoints)
        
        return all_keypoints

    
    def visualize(self, patch_size: int = 5):
        """
        Visualize image with keypoints and patches.
        
        Args:
            patch_size: Size of patches to draw
            
        Returns:
            Image with keypoints and patches drawn
        """
        return self.draw_keypoints_with_patches(patch_size=patch_size)

