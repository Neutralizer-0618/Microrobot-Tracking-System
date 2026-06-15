from __future__ import annotations

from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from micromotor_tracker.models.interactive_classifier import InteractiveClassifier
from micromotor_tracker.utils.config import DetectionConfig
from micromotor_tracker.utils.geometry import box_to_mask, centroid_from_contour


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def preprocess_frame(frame: np.ndarray, config: DetectionConfig) -> np.ndarray:
    gray = _gray(frame)
    blur_kernel = _odd(config.blur_kernel)
    processed = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    if config.use_background_subtraction:
        bg_kernel = _odd(config.background_blur_kernel)
        background = cv2.GaussianBlur(processed, (bg_kernel, bg_kernel), 0)
        processed = cv2.absdiff(processed, background)
    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    return processed


def threshold_frame(processed: np.ndarray, config: DetectionConfig) -> np.ndarray:
    if config.threshold_method == "manual":
        _, binary = cv2.threshold(processed, config.manual_threshold, 255, cv2.THRESH_BINARY)
    elif config.threshold_method == "adaptive":
        block_size = _odd(config.adaptive_block_size)
        binary = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            config.adaptive_c,
        )
    else:
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if config.morph_open_iterations > 0:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=config.morph_open_iterations,
        )
    if config.morph_close_iterations > 0:
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=config.morph_close_iterations,
        )
    return binary


def contour_features(frame: np.ndarray, contour: np.ndarray) -> Optional[Dict[str, float]]:
    centroid = centroid_from_contour(contour)
    if centroid is None:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    aspect_ratio = float(w / max(h, 1))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    gray = _gray(frame)
    mean_intensity, std_intensity = cv2.meanStdDev(gray, mask=mask)
    laplacian_var = float(cv2.Laplacian(gray[y : y + h, x : x + w], cv2.CV_32F).var())
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0
    texture_score = laplacian_var
    return {
        "x": float(centroid[0]),
        "y": float(centroid[1]),
        "area": area,
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(w),
        "bbox_h": int(h),
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "mean_intensity": float(mean_intensity[0][0]),
        "std_intensity": float(std_intensity[0][0]),
        "texture_score": texture_score,
        "solidity": solidity,
        "bbox_area": float(w * h),
        "contour_area": area,
        "laplacian_var": laplacian_var,
        "edge_density": float(perimeter / max(2 * (w + h), 1)),
    }


def detect_objects_in_frame(
    frame: np.ndarray,
    frame_index: int,
    config: DetectionConfig,
    classifier: Optional[InteractiveClassifier] = None,
    roi_box: Optional[tuple[int, int, int, int]] = None,
) -> List[Dict[str, float]]:
    processed = preprocess_frame(frame, config)
    binary = threshold_frame(processed, config)
    roi_mask = box_to_mask(roi_box, frame.shape)
    binary = cv2.bitwise_and(binary, binary, mask=roi_mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Dict[str, float]] = []
    for contour in contours:
        features = contour_features(frame, contour)
        if not features:
            continue
        if features["area"] < config.min_area or features["area"] > config.max_area:
            continue
        if not (config.min_circularity <= features["circularity"] <= config.max_circularity):
            continue
        if not (config.min_aspect_ratio <= features["aspect_ratio"] <= config.max_aspect_ratio):
            continue
        if roi_box is not None:
            x0, y0, w0, h0 = roi_box
            if not (x0 <= features["x"] <= x0 + w0 and y0 <= features["y"] <= y0 + h0):
                continue
        confidence = 0.65
        if classifier is not None:
            confidence = classifier.score_features(features)
        features["confidence"] = float(confidence)
        if confidence < config.min_confidence:
            continue
        features["frame"] = int(frame_index)
        detections.append(features)
    return detections


def run_detection(
    video_source,
    config: DetectionConfig,
    classifier: Optional[InteractiveClassifier] = None,
    roi_box: Optional[tuple[int, int, int, int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    total_frames = video_source.metadata.frame_count
    start = max(0, int(frame_start))
    end = total_frames - 1 if frame_end is None else min(int(frame_end), total_frames - 1)
    if end < start:
        return pd.DataFrame()
    total = end - start + 1
    for index, frame in enumerate(video_source.iter_frames()):
        if index < start:
            continue
        if index > end:
            break
        rows.extend(detect_objects_in_frame(frame, index, config, classifier=classifier, roi_box=roi_box))
        if progress_callback:
            progress_callback(index - start + 1, total)
    if not rows:
        columns = [
            "frame",
            "x",
            "y",
            "area",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "confidence",
            "circularity",
            "aspect_ratio",
            "mean_intensity",
            "std_intensity",
            "texture_score",
            "solidity",
            "bbox_area",
            "contour_area",
            "laplacian_var",
            "edge_density",
        ]
        return pd.DataFrame(columns=columns)
    detections = pd.DataFrame(rows).sort_values(["frame", "confidence"], ascending=[True, False]).reset_index(drop=True)
    detections["detection_id"] = np.arange(1, len(detections) + 1)
    return detections
