"""Deterministic, deploy-time-safe primitives for generated 2-D simulators.

The helpers in this module are deliberately task agnostic.  They provide the
reliable numerical plumbing (appearance masking, component geometry, simple
identified dynamics, and planar rigid contact) while leaving object identity,
state design, action semantics, and objectives to the generated simulator.

They are local and deterministic: no network, dataset, model, or WorldAPI
access is performed, so the same functions are available in P1 and P2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


LOCAL_2D_TOOLBOX_VERSION = "4"
# Version 4 only adds helpers/metadata and preserves the numerical behaviour
# of all version-3 methods. Existing generated simulators may therefore run
# against v4, while their original generation version remains recorded.
LOCAL_2D_TOOLBOX_COMPATIBLE_VERSIONS = frozenset({"3", "4"})


def _rgb_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"image must have shape (H, W, >=3), got {array.shape}")
    return np.clip(array[..., :3], 0, 255).astype(np.uint8)


def _vec(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array


def clean_mask(
    mask: np.ndarray,
    *,
    min_area: int = 1,
    open_radius: int = 0,
    close_radius: int = 1,
    largest: bool = True,
) -> np.ndarray:
    """Morphologically clean a mask and optionally retain its largest component."""
    import cv2

    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {binary.shape}")
    if int(min_area) < 1:
        raise ValueError("min_area must be >= 1")

    work = binary.astype(np.uint8) * 255
    if int(open_radius) > 0:
        radius = int(open_radius)
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
    if int(close_radius) > 0:
        radius = int(close_radius)
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (work > 0).astype(np.uint8),
        connectivity=8,
    )
    candidates = [
        index
        for index in range(1, int(count))
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_area)
    ]
    if not candidates:
        return np.zeros_like(binary)
    if largest:
        selected = max(
            candidates,
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        )
        return labels == int(selected)
    return np.isin(labels, np.asarray(candidates, dtype=np.int32))


def select_connected_component(
    mask: np.ndarray,
    *,
    min_area: int = 1,
    reference_centroid: Sequence[float] | np.ndarray | None = None,
    expected_area: float | None = None,
    connectivity: int = 8,
) -> dict[str, Any]:
    """Select one coherent component using optional geometric expectations.

    Components smaller than ``min_area`` are never eligible. Eligible
    components are ranked by centroid distance and/or log-area error when the
    corresponding references are supplied; without references this retains the
    largest-component behaviour. Ties are resolved deterministically by area,
    centroid, then connected-component label.

    ``confidence`` is in ``[0, 1]`` and combines absolute agreement with the
    references and ambiguity among eligible candidates. All metadata values
    are ordinary Python scalars/lists/``None`` for straightforward
    serialization; only ``mask`` is a boolean NumPy array.
    """
    import cv2

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {array.shape}")
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError("mask must contain boolean or numeric values")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError("mask must contain only finite values")
    if isinstance(min_area, bool):
        raise ValueError("min_area must be an integer >= 1")
    try:
        minimum = int(min_area)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("min_area must be an integer >= 1") from exc
    if minimum < 1 or minimum != min_area:
        raise ValueError("min_area must be an integer >= 1")
    if isinstance(connectivity, bool) or connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")

    reference = (
        None
        if reference_centroid is None
        else _vec("reference_centroid", reference_centroid, 2)
    )
    target_area = None if expected_area is None else float(expected_area)
    if target_area is not None and (
        not np.isfinite(target_area) or target_area <= 0.0
    ):
        raise ValueError("expected_area must be finite and positive")

    binary = array.astype(bool)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=int(connectivity),
    )
    component_count = max(0, int(count) - 1)
    diagonal = max(float(np.hypot(*binary.shape)), 1.0)
    candidates: list[dict[str, Any]] = []
    for label_index in range(1, int(count)):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < minimum:
            continue
        centroid = np.asarray(centroids[label_index], dtype=np.float64)
        score = 0.0
        if reference is not None:
            score += float(np.linalg.norm(centroid - reference) / diagonal)
        if target_area is not None:
            score += float(abs(np.log(area / target_area)))
        candidates.append(
            {
                "label": int(label_index),
                "area": area,
                "centroid": centroid,
                "score": score,
            }
        )

    if not candidates:
        return {
            "mask": np.zeros_like(binary),
            "valid": False,
            "confidence": 0.0,
            "area": 0,
            "centroid": None,
            "component_count": component_count,
            "candidate_count": 0,
        }

    uses_reference = reference is not None or target_area is not None
    if uses_reference:
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(item["score"]),
                -int(item["area"]),
                float(item["centroid"][1]),
                float(item["centroid"][0]),
                int(item["label"]),
            ),
        )
        selected = ranked[0]
        relative_weights = np.exp(
            -np.asarray(
                [float(item["score"]) - float(selected["score"]) for item in ranked],
                dtype=np.float64,
            )
        )
        confidence = float(
            np.exp(-float(selected["score"])) / np.sum(relative_weights)
        )
    else:
        ranked = sorted(
            candidates,
            key=lambda item: (
                -int(item["area"]),
                float(item["centroid"][1]),
                float(item["centroid"][0]),
                int(item["label"]),
            ),
        )
        selected = ranked[0]
        confidence = float(
            int(selected["area"])
            / sum(int(item["area"]) for item in ranked)
        )

    centroid = np.asarray(selected["centroid"], dtype=np.float64)
    return {
        "mask": labels == int(selected["label"]),
        "valid": True,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "area": int(selected["area"]),
        "centroid": [float(centroid[0]), float(centroid[1])],
        "component_count": component_count,
        "candidate_count": len(candidates),
    }


def mask_from_appearance(
    image: np.ndarray,
    model: Mapping[str, Any],
    *,
    min_area: int = 1,
    open_radius: int = 0,
    close_radius: int = 1,
    largest: bool = True,
) -> np.ndarray:
    """Segment pixels with a robust Lab appearance model.

    ``model`` is a JSON/Python-literal-friendly mapping with:

    * ``center_lab``: three-channel Lab centre;
    * ``scale_lab``: robust per-channel scale (strictly positive);
    * ``max_distance``: maximum scaled Euclidean distance.

    The code-generation tool ``derive_visual_model`` emits this exact format.
    """
    import cv2

    rgb = _rgb_image(image)
    center = _vec("model['center_lab']", model.get("center_lab"), 3)
    scale = _vec("model['scale_lab']", model.get("scale_lab"), 3)
    if np.any(scale <= 0):
        raise ValueError("model['scale_lab'] must be strictly positive")
    max_distance = float(model.get("max_distance", 3.0))
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("model['max_distance'] must be finite and positive")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    distance = np.linalg.norm((lab - center.reshape(1, 1, 3)) / scale, axis=2)
    mask = distance <= max_distance
    return clean_mask(
        mask,
        min_area=int(min_area),
        open_radius=int(open_radius),
        close_radius=int(close_radius),
        largest=bool(largest),
    )


def foreground_from_border(
    image: np.ndarray,
    *,
    lab_distance: float = 12.0,
    border_width: int = 3,
    min_area: int = 1,
    open_radius: int = 0,
    close_radius: int = 1,
    largest: bool = False,
) -> np.ndarray:
    """Segment non-background pixels using the robust median border colour."""
    import cv2

    rgb = _rgb_image(image)
    h, w = rgb.shape[:2]
    width = max(1, min(int(border_width), h // 2, w // 2))
    border = np.concatenate(
        [
            rgb[:width].reshape(-1, 3),
            rgb[-width:].reshape(-1, 3),
            rgb[:, :width].reshape(-1, 3),
            rgb[:, -width:].reshape(-1, 3),
        ],
        axis=0,
    )
    background_rgb = np.median(border, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    background_lab = cv2.cvtColor(
        background_rgb.reshape(1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(3).astype(np.float64)
    mask = np.linalg.norm(lab - background_lab.reshape(1, 1, 3), axis=2) >= float(
        lab_distance
    )
    return clean_mask(
        mask,
        min_area=int(min_area),
        open_radius=int(open_radius),
        close_radius=int(close_radius),
        largest=bool(largest),
    )


def component_geometry(
    mask: np.ndarray,
    *,
    simplify_px: float = 1.0,
    max_vertices: int = 64,
) -> dict[str, Any]:
    """Return centroid, bounds, PCA orientation, and a simplified outer contour."""
    import cv2

    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {binary.shape}")
    ys, xs = np.where(binary)
    if xs.size == 0:
        return {
            "area": 0,
            "centroid": None,
            "bbox": None,
            "pca_angle": None,
            "contour": [],
        }

    points = np.column_stack([xs, ys]).astype(np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(1, len(points))
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    pca_angle = float(np.arctan2(axis[1], axis[0]))

    contours, _ = cv2.findContours(
        binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    outer = max(contours, key=cv2.contourArea)
    epsilon = max(0.0, float(simplify_px))
    approx = cv2.approxPolyDP(outer, epsilon, closed=True).reshape(-1, 2)
    while len(approx) > int(max_vertices) and epsilon < 64.0:
        epsilon = max(0.5, epsilon * 1.5 if epsilon else 0.5)
        approx = cv2.approxPolyDP(outer, epsilon, closed=True).reshape(-1, 2)

    return {
        "area": int(xs.size),
        "centroid": [float(centroid[0]), float(centroid[1])],
        "bbox": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        "pca_angle": pca_angle,
        "contour": approx.astype(float).tolist(),
    }


def template_from_mask(
    mask: np.ndarray,
    *,
    simplify_px: float = 1.0,
    max_vertices: int = 64,
) -> list[np.ndarray]:
    """Convert a segmented rigid component into a centroid-centred polygon template."""
    geometry = component_geometry(
        mask,
        simplify_px=simplify_px,
        max_vertices=max_vertices,
    )
    if not geometry["contour"] or geometry["centroid"] is None:
        raise ValueError("cannot derive a template from an empty mask")
    contour = np.asarray(geometry["contour"], dtype=np.float64)
    centroid = np.asarray(geometry["centroid"], dtype=np.float64)
    return [contour - centroid]


def calibrate_component_geometry(
    masks: Sequence[np.ndarray],
    *,
    min_area: int = 1,
    open_radius: int = 0,
    close_radius: int = 0,
    largest: bool = True,
    simplify_px: float = 1.0,
    max_vertices: int = 64,
    orientation_anisotropy_threshold: float = 0.05,
) -> dict[str, Any]:
    """Calibrate a rigid component's geometry from one or more binary masks.

    Mask coordinates are reported unchanged as ``(x=column, y=row)``; this
    helper performs no implicit axis flip. Each frame record contains area,
    centroid, bounding-box dimensions, and area-equivalent radius. Aggregate
    values are robust medians over non-empty frames.

    ``canonical_template`` is a one-polygon, centroid-centred template selected
    from the frame nearest the median geometry and scaled to the median area.
    An anisotropic silhouette is PCA-aligned with deterministic 180-degree sign
    disambiguation. Near-isotropic components (notably circular pushers) are
    intentionally left in their observed orientation and flagged as unstable.

    The returned mapping contains only Python scalars, lists, dictionaries, and
    ``None``, so it can be serialized directly as JSON.
    """
    mask_list = list(masks)
    if not mask_list:
        raise ValueError("masks must contain at least one mask")
    if isinstance(min_area, bool) or int(min_area) < 1:
        raise ValueError("min_area must be an integer >= 1")
    if isinstance(max_vertices, bool) or int(max_vertices) < 3:
        raise ValueError("max_vertices must be an integer >= 3")
    simplify = float(simplify_px)
    threshold = float(orientation_anisotropy_threshold)
    if not np.isfinite(simplify) or simplify < 0:
        raise ValueError("simplify_px must be finite and non-negative")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "orientation_anisotropy_threshold must be finite and in [0, 1]"
        )

    frame_records: list[dict[str, Any]] = []
    canonical_candidates: list[dict[str, Any]] = []
    for frame_index, mask in enumerate(mask_list):
        binary = np.asarray(mask).astype(bool)
        if binary.ndim != 2:
            raise ValueError(
                f"every mask must be 2-D; mask {frame_index} has shape {binary.shape}"
            )
        cleaned = clean_mask(
            binary,
            min_area=int(min_area),
            open_radius=int(open_radius),
            close_radius=int(close_radius),
            largest=bool(largest),
        )
        geometry = component_geometry(
            cleaned,
            simplify_px=simplify,
            max_vertices=int(max_vertices),
        )
        if geometry["centroid"] is None:
            frame_records.append(
                {
                    "frame_index": int(frame_index),
                    "area": 0,
                    "centroid": None,
                    "bbox": None,
                    "bbox_dimensions": None,
                    "area_equivalent_radius": 0.0,
                    "pca_angle": None,
                    "orientation_anisotropy": None,
                    "orientation_stable": False,
                    "relative_scale": None,
                }
            )
            continue

        area = int(geometry["area"])
        bbox = [int(value) for value in geometry["bbox"]]
        dimensions = [int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]
        ys, xs = np.where(cleaned)
        points = np.column_stack([xs, ys]).astype(np.float64)
        centroid = np.asarray(geometry["centroid"], dtype=np.float64)
        centered_points = points - centroid
        covariance = centered_points.T @ centered_points / max(1, len(points))
        eigenvalues = np.linalg.eigvalsh(covariance)
        largest_eigenvalue = float(max(eigenvalues[-1], 0.0))
        anisotropy = (
            float((eigenvalues[-1] - eigenvalues[0]) / largest_eigenvalue)
            if largest_eigenvalue > 1e-12
            else 0.0
        )
        orientation_stable = bool(anisotropy >= threshold)
        record = {
            "frame_index": int(frame_index),
            "area": area,
            "centroid": [float(value) for value in centroid],
            "bbox": bbox,
            "bbox_dimensions": dimensions,
            "area_equivalent_radius": float(np.sqrt(area / np.pi)),
            "pca_angle": float(geometry["pca_angle"]),
            "orientation_anisotropy": anisotropy,
            "orientation_stable": orientation_stable,
            "relative_scale": None,
        }
        frame_records.append(record)
        contour = np.asarray(geometry["contour"], dtype=np.float64)
        if contour.shape[0] >= 3:
            canonical_candidates.append(
                {
                    "frame_index": int(frame_index),
                    "contour": contour,
                    "centroid": centroid,
                    "centered_points": centered_points,
                    "pca_angle": float(geometry["pca_angle"]),
                    "orientation_stable": orientation_stable,
                }
            )

    valid_records = [record for record in frame_records if record["area"] > 0]
    if not valid_records:
        raise ValueError("at least one mask must contain a component")
    if not canonical_candidates:
        raise ValueError("at least one component must have a polygonal contour")

    areas = np.asarray([record["area"] for record in valid_records], dtype=np.float64)
    centroids = np.asarray(
        [record["centroid"] for record in valid_records],
        dtype=np.float64,
    )
    bbox_dimensions = np.asarray(
        [record["bbox_dimensions"] for record in valid_records],
        dtype=np.float64,
    )
    median_area = float(np.median(areas))
    median_centroid = np.median(centroids, axis=0)
    median_dimensions = np.median(bbox_dimensions, axis=0)
    median_radius = float(np.sqrt(median_area / np.pi))

    relative_scales = np.sqrt(areas / median_area)
    for record, relative_scale in zip(valid_records, relative_scales):
        record["relative_scale"] = float(relative_scale)

    def candidate_score(candidate: dict[str, Any]) -> tuple[float, int]:
        record = frame_records[int(candidate["frame_index"])]
        dimensions = np.asarray(record["bbox_dimensions"], dtype=np.float64)
        area_error = abs(np.log(float(record["area"]) / median_area))
        dimension_error = np.sum(
            np.abs(dimensions - median_dimensions) / np.maximum(median_dimensions, 1.0)
        )
        return float(area_error + dimension_error), int(candidate["frame_index"])

    selected = min(canonical_candidates, key=candidate_score)
    canonical = np.asarray(selected["contour"], dtype=np.float64) - np.asarray(
        selected["centroid"],
        dtype=np.float64,
    )
    if bool(selected["orientation_stable"]):
        angle = float(selected["pca_angle"])
        axes = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float64,
        )
        canonical = canonical @ axes
        aligned_points = np.asarray(selected["centered_points"]) @ axes
        characteristic_length = max(
            float(np.sqrt(np.mean(np.sum(aligned_points**2, axis=1)))),
            1.0,
        )
        odd_moments = (
            float(np.mean(aligned_points[:, 0] ** 3)),
            float(np.mean(aligned_points[:, 0] * aligned_points[:, 1] ** 2)),
        )
        sign_measure = next(
            (
                value
                for value in odd_moments
                if abs(value) > 1e-9 * characteristic_length**3
            ),
            0.0,
        )
        if sign_measure < 0.0:
            canonical = -canonical

    selected_record = frame_records[int(selected["frame_index"])]
    canonical_scale = float(np.sqrt(median_area / float(selected_record["area"])))
    canonical *= canonical_scale
    signed_area = 0.5 * float(
        np.sum(
            canonical[:, 0] * np.roll(canonical[:, 1], -1)
            - canonical[:, 1] * np.roll(canonical[:, 0], -1)
        )
    )
    if signed_area < 0.0:
        canonical = canonical[::-1]
    rounded = np.round(canonical, decimals=12)
    start = int(np.lexsort((rounded[:, 1], rounded[:, 0]))[0])
    canonical = np.roll(canonical, -start, axis=0)

    median_relative_scale = float(np.median(relative_scales))
    mad_relative_scale = float(
        np.median(np.abs(relative_scales - median_relative_scale))
    )
    return {
        "n_frames": len(frame_records),
        "n_valid_frames": len(valid_records),
        "frames": frame_records,
        "median_area": median_area,
        "median_centroid": [float(value) for value in median_centroid],
        "median_bbox_dimensions": [
            float(value) for value in median_dimensions
        ],
        "median_area_equivalent_radius": median_radius,
        "canonical_template": [
            [[float(value) for value in point] for point in canonical]
        ],
        "canonical_frame_index": int(selected["frame_index"]),
        "canonical_scale": canonical_scale,
        "canonical_orientation_stable": bool(selected["orientation_stable"]),
        "scale_diagnostics": {
            "relative_scales": [float(value) for value in relative_scales],
            "median_relative_scale": median_relative_scale,
            "mad_relative_scale": mad_relative_scale,
            "min_relative_scale": float(np.min(relative_scales)),
            "max_relative_scale": float(np.max(relative_scales)),
        },
    }


def fit_articulated_chain(
    mask: np.ndarray,
    base_xy: np.ndarray,
    link_lengths: np.ndarray,
    *,
    initial_angles: np.ndarray | None = None,
    relative_angles: bool = True,
    n_starts: int = 128,
    prior_weight: float = 0.02,
) -> dict[str, Any]:
    """Fit a planar fixed-length serial chain to a binary link mask.

    The task-agnostic objective is a symmetric Chamfer distance between mask
    pixels and densely sampled chain segments. ``initial_angles`` supplies a
    temporal-continuity prior when refitting video frames. Angles are relative
    joint angles by default and use image coordinates (positive y downward).
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {binary.shape}")
    ys, xs = np.where(binary)
    if xs.size < 3:
        raise ValueError("mask must contain at least three foreground pixels")
    base = _vec("base_xy", base_xy, 2)
    lengths = np.asarray(link_lengths, dtype=np.float64).reshape(-1)
    if lengths.size == 0 or np.any(lengths <= 0) or not np.all(np.isfinite(lengths)):
        raise ValueError("link_lengths must be a non-empty positive finite vector")
    count = int(lengths.size)
    initial = (
        None
        if initial_angles is None
        else _vec("initial_angles", initial_angles, count)
    )
    starts = max(1, int(n_starts))

    mask_points = np.column_stack([xs, ys]).astype(np.float64)
    # Bound cost while retaining the whole silhouette spatially.
    if len(mask_points) > 2000:
        indices = np.linspace(0, len(mask_points) - 1, 2000).astype(int)
        mask_points = mask_points[indices]
    mask_tree = cKDTree(mask_points)

    def joints_for(angles: np.ndarray) -> np.ndarray:
        joints = np.empty((count + 1, 2), dtype=np.float64)
        joints[0] = base
        headings = np.cumsum(angles) if relative_angles else angles
        for index, (heading, length) in enumerate(
            zip(headings, lengths),
            start=1,
        ):
            joints[index] = joints[index - 1] + float(length) * np.array(
                [np.cos(heading), np.sin(heading)],
                dtype=np.float64,
            )
        return joints

    def chain_samples(joints: np.ndarray) -> np.ndarray:
        rows = []
        for index, length in enumerate(lengths):
            samples = max(8, int(np.ceil(float(length) * 1.5)))
            alpha = np.linspace(0.0, 1.0, samples, endpoint=True)[:, None]
            rows.append(
                (1.0 - alpha) * joints[index] + alpha * joints[index + 1]
            )
        return np.concatenate(rows, axis=0)

    def segment_distance_squared(points: np.ndarray, start, end) -> np.ndarray:
        direction = end - start
        denom = max(float(direction @ direction), 1e-12)
        alpha = np.clip(((points - start) @ direction) / denom, 0.0, 1.0)
        closest = start + alpha[:, None] * direction
        return np.sum((points - closest) ** 2, axis=1)

    def visual_objective(angles: np.ndarray) -> float:
        wrapped = np.arctan2(np.sin(angles), np.cos(angles))
        joints = joints_for(wrapped)
        pixel_to_chain = np.min(
            np.column_stack(
                [
                    segment_distance_squared(
                        mask_points,
                        joints[index],
                        joints[index + 1],
                    )
                    for index in range(count)
                ]
            ),
            axis=1,
        )
        samples = chain_samples(joints)
        chain_to_pixel, _ = mask_tree.query(samples, k=1)
        return float(
            np.mean(np.minimum(pixel_to_chain, 100.0))
            + np.mean(np.minimum(chain_to_pixel**2, 100.0))
        )

    def objective(angles: np.ndarray) -> float:
        wrapped = np.arctan2(np.sin(angles), np.cos(angles))
        value = visual_objective(wrapped)
        if initial is not None and float(prior_weight) > 0:
            delta = np.arctan2(np.sin(wrapped - initial), np.cos(wrapped - initial))
            value += float(prior_weight) * float(delta @ delta)
        return value

    candidates: list[np.ndarray] = []
    if initial is not None:
        candidates.append(initial.copy())
    if count <= 3:
        # A true Cartesian grid is important for short chains. The previous
        # correlated sequence sampled two-link starts only where theta2 was
        # theta1 or theta1+pi, which systematically missed folded branches.
        side = max(3, int(np.ceil(starts ** (1.0 / count))))
        axis = -np.pi + (np.arange(side, dtype=np.float64) + 0.5) * (
            2.0 * np.pi / side
        )
        mesh = np.meshgrid(*([axis] * count), indexing="ij")
        candidates.extend(
            np.column_stack([values.reshape(-1) for values in mesh])
        )
    else:
        # Avoid an exponential grid for longer chains while still sampling
        # dimensions independently and deterministically.
        from scipy.stats import qmc

        unit = qmc.Halton(d=count, scramble=False).random(n=starts)
        candidates.extend(-np.pi + 2.0 * np.pi * unit)

    scored = sorted(
        ((objective(candidate), candidate) for candidate in candidates),
        key=lambda row: row[0],
    )
    refined: list[tuple[float, np.ndarray]] = []
    for value, candidate in scored[: min(16, len(scored))]:
        result = minimize(
            objective,
            candidate,
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * count,
            options={"maxiter": 160, "ftol": 1e-10},
        )
        angles = np.asarray(result.x, dtype=np.float64)
        refined.append((float(result.fun), angles))

    ranked = sorted(refined + scored[: min(16, len(scored))], key=lambda row: row[0])
    distinct: list[tuple[float, np.ndarray]] = []
    for value, angles in ranked:
        wrapped = np.arctan2(np.sin(angles), np.cos(angles))
        if any(
            np.linalg.norm(
                np.arctan2(np.sin(wrapped - other), np.cos(wrapped - other))
            )
            < 0.15
            for _, other in distinct
        ):
            continue
        distinct.append((float(value), wrapped))
        if len(distinct) >= 4:
            break

    best_value, best_angles = distinct[0]
    best_angles = np.arctan2(np.sin(best_angles), np.cos(best_angles))
    joints = joints_for(best_angles)
    candidate_records = []
    for value, angles in distinct:
        candidate_joints = joints_for(angles)
        candidate_records.append(
            {
                "angles": angles.copy(),
                "joints": candidate_joints,
                "end_effector": candidate_joints[-1].copy(),
                "chamfer_rmse": float(
                    np.sqrt(max(0.0, visual_objective(angles)))
                ),
                "objective": float(value),
            }
        )
    ambiguity_margin = None
    ambiguous = False
    if len(distinct) > 1:
        ambiguity_margin = float(distinct[1][0] - distinct[0][0])
        relative_margin = ambiguity_margin / max(abs(float(best_value)), 1e-9)
        branch_distance = float(
            np.linalg.norm(
                np.arctan2(
                    np.sin(distinct[1][1] - distinct[0][1]),
                    np.cos(distinct[1][1] - distinct[0][1]),
                )
            )
        )
        ambiguous = bool(relative_margin < 0.10 and branch_distance > 0.35)
    return {
        "angles": best_angles,
        "joints": joints,
        "end_effector": joints[-1].copy(),
        "chamfer_rmse": float(
            np.sqrt(max(0.0, visual_objective(best_angles)))
        ),
        "n_mask_pixels": int(binary.sum()),
        "candidates": candidate_records,
        "ambiguity_margin": ambiguity_margin,
        "ambiguous": ambiguous,
    }


def calibrate_articulated_chain_geometry(
    masks: Sequence[np.ndarray],
    base_xy: np.ndarray,
    link_lengths: np.ndarray,
    *,
    base_search_radius: float = 12.0,
    length_scale_bounds: tuple[float, float] = (0.70, 1.30),
    n_starts: int = 64,
    iterations: int = 2,
) -> dict[str, Any]:
    """Estimate fixed chain base and link lengths from several diverse masks.

    Geometry and per-frame angles are fitted alternately. This keeps fixed
    dimensions out of single-frame ``fit()`` heuristics while remaining fully
    local and task agnostic. Call during code generation, then bake the returned
    robust constants into the deployed simulator.
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from skimage.morphology import skeletonize

    binary_masks = [
        skeletonize(np.asarray(mask).astype(bool))
        for mask in masks
    ]
    if len(binary_masks) < 2:
        raise ValueError("at least two masks are required for geometry calibration")
    if any(mask.ndim != 2 for mask in binary_masks):
        raise ValueError("every mask must be 2-D")
    if any(int(mask.sum()) < 3 for mask in binary_masks):
        raise ValueError("every mask must contain at least three foreground pixels")

    base_hint = _vec("base_xy", base_xy, 2)
    lengths_hint = np.asarray(link_lengths, dtype=np.float64).reshape(-1)
    if (
        lengths_hint.size == 0
        or np.any(lengths_hint <= 0)
        or not np.all(np.isfinite(lengths_hint))
    ):
        raise ValueError("link_lengths must be positive finite values")
    low_scale, high_scale = map(float, length_scale_bounds)
    if not (0.0 < low_scale <= 1.0 <= high_scale):
        raise ValueError("length_scale_bounds must straddle 1.0 and be positive")
    radius = float(base_search_radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("base_search_radius must be finite and non-negative")

    mask_points: list[np.ndarray] = []
    mask_trees = []
    for binary in binary_masks:
        ys, xs = np.where(binary)
        points = np.column_stack([xs, ys]).astype(np.float64)
        if len(points) > 1500:
            indices = np.linspace(0, len(points) - 1, 1500).astype(int)
            points = points[indices]
        mask_points.append(points)
        mask_trees.append(cKDTree(points))

    def joints_for(
        base: np.ndarray,
        lengths: np.ndarray,
        angles: np.ndarray,
    ) -> np.ndarray:
        joints = np.empty((len(lengths) + 1, 2), dtype=np.float64)
        joints[0] = base
        for index, (heading, length) in enumerate(
            zip(np.cumsum(angles), lengths),
            start=1,
        ):
            joints[index] = joints[index - 1] + float(length) * np.array(
                [np.cos(heading), np.sin(heading)]
            )
        return joints

    def frame_loss(
        frame_index: int,
        base: np.ndarray,
        lengths: np.ndarray,
        angles: np.ndarray,
    ) -> float:
        joints = joints_for(base, lengths, angles)
        points = mask_points[frame_index]
        distances = []
        samples = []
        for index, length in enumerate(lengths):
            start, end = joints[index], joints[index + 1]
            direction = end - start
            denom = max(float(direction @ direction), 1e-12)
            alpha = np.clip(((points - start) @ direction) / denom, 0.0, 1.0)
            closest = start + alpha[:, None] * direction
            distances.append(np.sum((points - closest) ** 2, axis=1))
            sample_count = max(8, int(np.ceil(float(length) * 1.5)))
            segment_alpha = np.linspace(0.0, 1.0, sample_count)[:, None]
            samples.append(
                (1.0 - segment_alpha) * start + segment_alpha * end
            )
        pixel_to_chain = np.min(np.column_stack(distances), axis=1)
        chain_points = np.concatenate(samples, axis=0)
        chain_to_pixel, _ = mask_trees[frame_index].query(chain_points, k=1)
        return float(
            np.mean(np.minimum(pixel_to_chain, 100.0))
            + np.mean(np.minimum(chain_to_pixel**2, 100.0))
        )

    base = base_hint.copy()
    lengths = lengths_hint.copy()
    angles_per_frame: list[np.ndarray] = []
    geometry_bounds = [
        (float(base_hint[0] - radius), float(base_hint[0] + radius)),
        (float(base_hint[1] - radius), float(base_hint[1] + radius)),
        *[
            (float(value * low_scale), float(value * high_scale))
            for value in lengths_hint
        ],
    ]

    for _ in range(max(1, int(iterations))):
        angles_per_frame = [
            np.asarray(
                fit_articulated_chain(
                    binary,
                    base,
                    lengths,
                    relative_angles=True,
                    n_starts=n_starts,
                )["angles"],
                dtype=np.float64,
            )
            for binary in binary_masks
        ]

        def geometry_objective(values: np.ndarray) -> float:
            candidate_base = np.asarray(values[:2], dtype=np.float64)
            candidate_lengths = np.asarray(values[2:], dtype=np.float64)
            losses = [
                frame_loss(index, candidate_base, candidate_lengths, angles)
                for index, angles in enumerate(angles_per_frame)
            ]
            base_regularizer = np.sum(
                ((candidate_base - base_hint) / max(radius, 1.0)) ** 2
            )
            length_regularizer = np.sum(
                ((candidate_lengths - lengths_hint) / lengths_hint) ** 2
            )
            return float(
                np.mean(losses)
                + 1e-3 * base_regularizer
                + 1e-3 * length_regularizer
            )

        result = minimize(
            geometry_objective,
            np.concatenate([base, lengths]),
            method="L-BFGS-B",
            bounds=geometry_bounds,
            options={"maxiter": 180, "ftol": 1e-10},
        )
        base = np.asarray(result.x[:2], dtype=np.float64)
        lengths = np.asarray(result.x[2:], dtype=np.float64)

    fitted_frames = [
        fit_articulated_chain(
            binary,
            base,
            lengths,
            relative_angles=True,
            n_starts=n_starts,
        )
        for binary in binary_masks
    ]
    rmses = np.asarray(
        [float(record["chamfer_rmse"]) for record in fitted_frames],
        dtype=np.float64,
    )
    return {
        "base_xy": base,
        "link_lengths": lengths,
        "geometry_delta": {
            "base": base - base_hint,
            "link_lengths": lengths - lengths_hint,
        },
        "mean_chamfer_rmse": float(np.mean(rmses)),
        "worst_chamfer_rmse": float(np.max(rmses)),
        "frames": fitted_frames,
        "n_frames": len(fitted_frames),
    }


def clamp_step(
    previous: np.ndarray,
    target: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """Move from ``previous`` toward ``target`` by at most ``max_step``."""
    previous_vec = np.asarray(previous, dtype=np.float64).reshape(-1)
    target_vec = np.asarray(target, dtype=np.float64).reshape(-1)
    if previous_vec.shape != target_vec.shape or previous_vec.size == 0:
        raise ValueError("previous and target must have the same non-empty shape")
    if not np.all(np.isfinite(previous_vec)) or not np.all(np.isfinite(target_vec)):
        raise ValueError("previous and target must be finite")
    limit = float(max_step)
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("max_step must be finite and non-negative")
    delta = target_vec - previous_vec
    distance = float(np.linalg.norm(delta))
    if distance <= limit or distance == 0.0:
        return target_vec.copy()
    return previous_vec + delta * (limit / distance)


def estimate_observed_velocity(
    current_positions: np.ndarray,
    previous_observed_positions: np.ndarray,
    elapsed_seconds: float,
    periodic: bool | Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Estimate velocity between two observations under declared topology.

    Only dimensions marked periodic use the shortest wrapped angular delta.
    Bounded and other nonperiodic dimensions retain their raw displacement.
    """
    current = np.asarray(current_positions, dtype=np.float64)
    previous = np.asarray(previous_observed_positions, dtype=np.float64)
    if current.shape != previous.shape or current.size == 0:
        raise ValueError(
            "current_positions and previous_observed_positions must have the "
            "same non-empty shape"
        )
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(previous)):
        raise ValueError("observed positions must contain only finite values")
    elapsed = float(elapsed_seconds)
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("elapsed_seconds must be finite and positive")

    periodic_array = np.asarray(periodic)
    if not (
        np.issubdtype(periodic_array.dtype, np.bool_)
        or (
            np.issubdtype(periodic_array.dtype, np.number)
            and np.all(np.isfinite(periodic_array))
            and np.all((periodic_array == 0) | (periodic_array == 1))
        )
    ):
        raise ValueError("periodic must be a boolean mask")
    try:
        periodic_mask = np.broadcast_to(
            periodic_array.astype(bool),
            current.shape,
        )
    except ValueError as exc:
        raise ValueError(
            f"periodic must be scalar or broadcastable to shape {current.shape}"
        ) from exc

    delta = current - previous
    if np.any(periodic_mask):
        wrapped_delta = np.arctan2(np.sin(delta), np.cos(delta))
        # np.where works for both zero-dimensional scalars and arrays. Boolean
        # item assignment does not: ``delta[periodic_mask] = ...`` crashes when
        # generated simulators track one scalar joint angle.
        delta = np.where(periodic_mask, wrapped_delta, delta)
    return delta / elapsed


def linear_dynamics_step(
    state: np.ndarray,
    action: np.ndarray,
    gain: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    max_delta: float | Sequence[float] | None = None,
) -> np.ndarray:
    """Apply an identified first-order model ``x_next = x + gain @ action + bias``."""
    x = np.asarray(state, dtype=np.float64).reshape(-1)
    u = np.asarray(action, dtype=np.float64).reshape(-1)
    matrix = np.asarray(gain, dtype=np.float64)
    if matrix.shape != (x.size, u.size):
        raise ValueError(
            f"gain must have shape {(x.size, u.size)}, got {matrix.shape}"
        )
    delta = matrix @ u
    if bias is not None:
        offset = np.asarray(bias, dtype=np.float64).reshape(-1)
        if offset.shape != x.shape:
            raise ValueError(f"bias must have shape {x.shape}, got {offset.shape}")
        delta = delta + offset
    if max_delta is not None:
        limit = np.asarray(max_delta, dtype=np.float64)
        if limit.ndim == 0:
            norm = float(np.linalg.norm(delta))
            scalar = float(limit)
            if scalar < 0:
                raise ValueError("max_delta must be non-negative")
            if norm > scalar > 0:
                delta = delta * (scalar / norm)
            elif scalar == 0:
                delta = np.zeros_like(delta)
        else:
            limit = np.broadcast_to(limit.reshape(-1), delta.shape)
            if np.any(limit < 0):
                raise ValueError("max_delta entries must be non-negative")
            delta = np.clip(delta, -limit, limit)
    return x + delta


def damped_dynamics_step(
    position: np.ndarray,
    velocity: np.ndarray,
    action: np.ndarray,
    gain: np.ndarray,
    *,
    damping: float | Sequence[float],
    dt: float = 1.0,
    max_velocity: float | Sequence[float] | None = None,
    position_low: float | Sequence[float] | None = None,
    position_high: float | Sequence[float] | None = None,
    periodic: bool | Sequence[bool] = False,
) -> dict[str, np.ndarray]:
    """Apply a semi-implicit damped update with optional position topology."""
    q = np.asarray(position, dtype=np.float64).reshape(-1)
    v = np.asarray(velocity, dtype=np.float64).reshape(-1)
    u = np.asarray(action, dtype=np.float64).reshape(-1)
    matrix = np.asarray(gain, dtype=np.float64)
    if q.shape != v.shape:
        raise ValueError("position and velocity must have matching shapes")
    if matrix.shape != (q.size, u.size):
        raise ValueError(
            f"gain must have shape {(q.size, u.size)}, got {matrix.shape}"
        )
    step = float(dt)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("dt must be finite and positive")
    damp = np.broadcast_to(np.asarray(damping, dtype=np.float64), q.shape)
    acceleration = matrix @ u - damp * v
    next_velocity = v + acceleration * step
    if max_velocity is not None:
        limit = np.broadcast_to(
            np.asarray(max_velocity, dtype=np.float64),
            q.shape,
        )
        if np.any(limit < 0):
            raise ValueError("max_velocity must be non-negative")
        next_velocity = np.clip(next_velocity, -limit, limit)
    next_position = q + next_velocity * step
    periodic_mask = np.broadcast_to(np.asarray(periodic, dtype=bool), q.shape)
    if np.any(periodic_mask):
        next_position = next_position.copy()
        next_position[periodic_mask] = np.arctan2(
            np.sin(next_position[periodic_mask]),
            np.cos(next_position[periodic_mask]),
        )
    if (position_low is None) != (position_high is None):
        raise ValueError("position_low and position_high must be provided together")
    if position_low is not None and position_high is not None:
        low = np.broadcast_to(np.asarray(position_low, dtype=np.float64), q.shape)
        high = np.broadcast_to(np.asarray(position_high, dtype=np.float64), q.shape)
        if np.any(np.isnan(low)) or np.any(np.isnan(high)) or np.any(low > high):
            raise ValueError("position bounds must be ordered and not NaN")
        if np.any(periodic_mask & (np.isfinite(low) | np.isfinite(high))):
            raise ValueError("periodic dimensions cannot also have finite bounds")
        below = next_position < low
        above = next_position > high
        next_position = np.clip(next_position, low, high)
        outward = (below & (next_velocity < 0.0)) | (
            above & (next_velocity > 0.0)
        )
        next_velocity = np.where(outward, 0.0, next_velocity)
    return {
        "position": next_position,
        "velocity": next_velocity,
    }


def convex_decompose_polygon(
    polygon: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> list[np.ndarray]:
    """Deterministically decompose a simple polygon into convex polygons.

    Convex inputs are returned as one polygon. Concave inputs are decomposed
    with deterministic ear clipping, currently yielding triangles. Vertices
    remain in the caller's numeric coordinate system and are never centred,
    transformed, or y-flipped, so the local coordinate origin is preserved.
    Self-intersecting and degenerate outlines are rejected.
    """
    vertices = np.asarray(polygon, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError(
            "polygon must have shape (N, 2), N >= 3; "
            f"got {vertices.shape}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("polygon must contain only finite values")
    epsilon = float(tolerance)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    vertices = vertices.copy()
    if len(vertices) > 3 and np.linalg.norm(vertices[0] - vertices[-1]) <= epsilon:
        vertices = vertices[:-1]
    deduplicated = []
    for vertex in vertices:
        if not deduplicated or np.linalg.norm(vertex - deduplicated[-1]) > epsilon:
            deduplicated.append(vertex)
    vertices = np.asarray(deduplicated, dtype=np.float64)
    if len(vertices) < 3:
        raise ValueError("polygon must contain at least three distinct vertices")

    changed = True
    while changed and len(vertices) > 3:
        changed = False
        keep = np.ones(len(vertices), dtype=bool)
        for index in range(len(vertices)):
            previous = vertices[(index - 1) % len(vertices)]
            current = vertices[index]
            following = vertices[(index + 1) % len(vertices)]
            incoming = current - previous
            outgoing = following - current
            cross = float(
                incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            )
            between = float((current - previous) @ (current - following)) <= epsilon
            if abs(cross) <= epsilon and between:
                keep[index] = False
                changed = True
        vertices = vertices[keep]
    if len(vertices) < 3:
        raise ValueError("polygon is degenerate")

    def cross_2d(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    def signed_area(points: np.ndarray) -> float:
        return 0.5 * float(
            np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - points[:, 1] * np.roll(points[:, 0], -1)
            )
        )

    area = signed_area(vertices)
    if abs(area) <= epsilon:
        raise ValueError("polygon area must be non-zero")
    orientation = 1.0 if area > 0.0 else -1.0

    def point_on_segment(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> bool:
        if abs(cross_2d(end - start, point - start)) > epsilon:
            return False
        return bool(
            np.all(point >= np.minimum(start, end) - epsilon)
            and np.all(point <= np.maximum(start, end) + epsilon)
        )

    def segments_intersect(
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
    ) -> bool:
        values = (
            cross_2d(first_end - first_start, second_start - first_start),
            cross_2d(first_end - first_start, second_end - first_start),
            cross_2d(second_end - second_start, first_start - second_start),
            cross_2d(second_end - second_start, first_end - second_start),
        )
        if (
            values[0] * values[1] < -(epsilon**2)
            and values[2] * values[3] < -(epsilon**2)
        ):
            return True
        return bool(
            (abs(values[0]) <= epsilon and point_on_segment(second_start, first_start, first_end))
            or (
                abs(values[1]) <= epsilon
                and point_on_segment(second_end, first_start, first_end)
            )
            or (
                abs(values[2]) <= epsilon
                and point_on_segment(first_start, second_start, second_end)
            )
            or (
                abs(values[3]) <= epsilon
                and point_on_segment(first_end, second_start, second_end)
            )
        )

    vertex_count = len(vertices)
    for first_index in range(vertex_count):
        first_next = (first_index + 1) % vertex_count
        for second_index in range(first_index + 1, vertex_count):
            second_next = (second_index + 1) % vertex_count
            if (
                first_index == second_index
                or first_next == second_index
                or second_next == first_index
            ):
                continue
            if segments_intersect(
                vertices[first_index],
                vertices[first_next],
                vertices[second_index],
                vertices[second_next],
            ):
                raise ValueError("polygon must be simple (non-self-intersecting)")

    corner_crosses = np.asarray(
        [
            cross_2d(
                vertices[(index + 1) % vertex_count] - vertices[index],
                vertices[(index + 2) % vertex_count]
                - vertices[(index + 1) % vertex_count],
            )
            for index in range(vertex_count)
        ]
    )
    if np.all(orientation * corner_crosses >= -epsilon):
        return [vertices.copy()]

    def point_in_triangle(
        point: np.ndarray,
        triangle: np.ndarray,
    ) -> bool:
        edge_crosses = [
            orientation
            * cross_2d(
                triangle[(index + 1) % 3] - triangle[index],
                point - triangle[index],
            )
            for index in range(3)
        ]
        return bool(min(edge_crosses) >= -epsilon)

    remaining = list(range(vertex_count))
    pieces: list[np.ndarray] = []
    while len(remaining) > 3:
        ear_found = False
        for position, current_index in enumerate(remaining):
            previous_index = remaining[(position - 1) % len(remaining)]
            next_index = remaining[(position + 1) % len(remaining)]
            triangle = vertices[[previous_index, current_index, next_index]]
            corner = cross_2d(
                vertices[current_index] - vertices[previous_index],
                vertices[next_index] - vertices[current_index],
            )
            if orientation * corner <= epsilon:
                continue
            if any(
                point_in_triangle(vertices[index], triangle)
                for index in remaining
                if index not in {previous_index, current_index, next_index}
            ):
                continue
            pieces.append(triangle.copy())
            del remaining[position]
            ear_found = True
            break
        if not ear_found:
            raise ValueError(
                "polygon could not be decomposed; check for degeneracy or "
                "self-intersection"
            )
    pieces.append(vertices[remaining].copy())
    return pieces


def planar_push_step(
    *,
    body_polygons: Sequence[np.ndarray],
    body_position: np.ndarray,
    body_angle: float,
    pusher_position: np.ndarray,
    pusher_target: np.ndarray,
    pusher_radius: float,
    body_velocity: np.ndarray | None = None,
    body_angular_velocity: float = 0.0,
    pusher_velocity: np.ndarray | None = None,
    body_mass: float = 1.0,
    friction: float = 0.0,
    damping: float = 1.0,
    kp: float = 100.0,
    kv: float = 20.0,
    dt: float = 0.01,
    substeps: int = 10,
    max_pusher_speed: float | None = None,
    bounds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Advance a circle-controlled pusher and arbitrary polygon rigid body.

    This is a generic local Pymunk primitive for 2-D contact systems. Polygon
    vertices are in the body's local ``(x, y)`` frame and may be convex pieces
    or concave simple outlines; concave outlines are decomposed
    deterministically. All positions, velocities, polygons, bounds, and
    returned values use the caller's numeric coordinates with no implicit
    y-axis flip.

    ``body_position`` is the world position of the local polygon origin. Do not
    mix a transformed/world position (for example, an image position after a
    y-flip) with untransformed local polygons. Apply any coordinate transform
    consistently to all geometric and kinematic inputs before calling.

    The return keys are exactly ``body_position``, ``body_angle``,
    ``body_velocity``, ``body_angular_velocity``, ``pusher_position``, and
    ``pusher_velocity``. Supplied linear and angular velocities initialize the
    bodies and the corresponding post-step velocities are returned.
    """
    import pymunk

    supplied_polygons = [np.asarray(poly, dtype=np.float64) for poly in body_polygons]
    if not supplied_polygons:
        raise ValueError("body_polygons must contain at least one polygon")
    polygons = [
        piece
        for polygon in supplied_polygons
        for piece in convex_decompose_polygon(polygon)
    ]

    body_pos = _vec("body_position", body_position, 2)
    pusher_pos = _vec("pusher_position", pusher_position, 2)
    target = _vec("pusher_target", pusher_target, 2)
    body_vel = (
        np.zeros(2, dtype=np.float64)
        if body_velocity is None
        else _vec("body_velocity", body_velocity, 2)
    )
    pusher_vel = (
        np.zeros(2, dtype=np.float64)
        if pusher_velocity is None
        else _vec("pusher_velocity", pusher_velocity, 2)
    )
    mass = float(body_mass)
    radius = float(pusher_radius)
    step = float(dt)
    if isinstance(substeps, bool) or not float(substeps).is_integer():
        raise ValueError("substeps must be a positive integer")
    count = int(substeps)
    angle = float(body_angle)
    angular_velocity = float(body_angular_velocity)
    coefficient_friction = float(friction)
    damping_factor = float(damping)
    proportional_gain = float(kp)
    derivative_gain = float(kv)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("body_mass must be finite and positive")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("pusher_radius must be finite and positive")
    if not np.isfinite(step) or step <= 0 or count <= 0:
        raise ValueError("dt and substeps must be positive")
    if not np.isfinite(angle) or not np.isfinite(angular_velocity):
        raise ValueError(
            "body_angle and body_angular_velocity must be finite"
        )
    if not np.isfinite(coefficient_friction) or coefficient_friction < 0.0:
        raise ValueError("friction must be finite and non-negative")
    if not np.isfinite(damping_factor) or not 0.0 <= damping_factor <= 1.0:
        raise ValueError("damping must be finite and in [0, 1]")
    if (
        not np.isfinite(proportional_gain)
        or proportional_gain < 0.0
        or not np.isfinite(derivative_gain)
        or derivative_gain < 0.0
    ):
        raise ValueError("kp and kv must be finite and non-negative")
    speed_limit = None
    if max_pusher_speed is not None:
        speed_limit = float(max_pusher_speed)
        if speed_limit < 0 or not np.isfinite(speed_limit):
            raise ValueError("max_pusher_speed must be finite and non-negative")

    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = damping_factor

    polygon_areas = np.asarray(
        [
            abs(
                0.5
                * np.sum(
                    polygon[:, 0] * np.roll(polygon[:, 1], -1)
                    - polygon[:, 1] * np.roll(polygon[:, 0], -1)
                )
            )
            for polygon in polygons
        ],
        dtype=np.float64,
    )
    total_area = float(np.sum(polygon_areas))
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("body polygons must have positive total area")
    moment = sum(
        pymunk.moment_for_poly(
            mass * float(polygon_area) / total_area,
            [tuple(row) for row in polygon],
        )
        for polygon, polygon_area in zip(polygons, polygon_areas)
    )
    body = pymunk.Body(mass, max(float(moment), 1e-9))
    body.position = tuple(body_pos)
    body.angle = angle
    body.velocity = tuple(body_vel)
    body.angular_velocity = angular_velocity
    shapes = [
        pymunk.Poly(body, [tuple(row) for row in polygon]) for polygon in polygons
    ]
    for shape in shapes:
        shape.friction = coefficient_friction
    space.add(body, *shapes)

    pusher = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    pusher.position = tuple(pusher_pos)
    pusher.velocity = tuple(pusher_vel)
    pusher_shape = pymunk.Circle(pusher, radius)
    pusher_shape.friction = coefficient_friction
    space.add(pusher, pusher_shape)

    if bounds is not None:
        limits = np.asarray(bounds, dtype=np.float64).reshape(-1)
        if limits.shape != (4,) or not np.all(np.isfinite(limits)):
            raise ValueError(
                "bounds must be finite [xmin, ymin, xmax, ymax]"
            )
        xmin, ymin, xmax, ymax = limits.tolist()
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("bounds must have xmin < xmax and ymin < ymax")
        walls = [
            pymunk.Segment(space.static_body, (xmin, ymin), (xmax, ymin), 0.0),
            pymunk.Segment(space.static_body, (xmax, ymin), (xmax, ymax), 0.0),
            pymunk.Segment(space.static_body, (xmax, ymax), (xmin, ymax), 0.0),
            pymunk.Segment(space.static_body, (xmin, ymax), (xmin, ymin), 0.0),
        ]
        for wall in walls:
            wall.friction = coefficient_friction
        space.add(*walls)

    for _ in range(count):
        position = np.asarray(pusher.position, dtype=np.float64)
        velocity = np.asarray(pusher.velocity, dtype=np.float64)
        acceleration = (
            proportional_gain * (target - position) - derivative_gain * velocity
        )
        velocity = velocity + acceleration * step
        if speed_limit is not None:
            speed = float(np.linalg.norm(velocity))
            if speed > speed_limit > 0:
                velocity = velocity * (speed_limit / speed)
            elif speed_limit == 0:
                velocity = np.zeros_like(velocity)
        pusher.velocity = tuple(velocity)
        space.step(step)

    return {
        "body_position": np.array([body.position.x, body.position.y], dtype=float),
        "body_angle": float(body.angle),
        "body_velocity": np.array([body.velocity.x, body.velocity.y], dtype=float),
        "body_angular_velocity": float(body.angular_velocity),
        "pusher_position": np.array(
            [pusher.position.x, pusher.position.y],
            dtype=float,
        ),
        "pusher_velocity": np.array(
            [pusher.velocity.x, pusher.velocity.y],
            dtype=float,
        ),
    }
