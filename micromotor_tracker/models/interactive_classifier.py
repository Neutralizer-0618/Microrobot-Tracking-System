from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from micromotor_tracker.utils.geometry import Box, clip_box, crop_box


def _safe_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def compute_patch_features(frame: np.ndarray, box: Box) -> Dict[str, float]:
    clipped = clip_box(box, frame.shape[1], frame.shape[0])
    patch = crop_box(frame, clipped)
    gray = _safe_gray(patch)
    area = float(clipped[2] * clipped[3])
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / max(edges.size, 1))
    aspect_ratio = float(clipped[2] / max(clipped[3], 1))
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = 0.0
    circularity = 0.0
    solidity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter > 0:
            circularity = float(4.0 * np.pi * contour_area / (perimeter * perimeter))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 0:
            solidity = contour_area / hull_area
    return {
        "bbox_area": area,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "laplacian_var": laplacian_var,
        "edge_density": edge_density,
        "aspect_ratio": aspect_ratio,
        "contour_area": contour_area,
        "circularity": circularity,
        "solidity": solidity,
    }


@dataclass
class InteractiveClassifier:
    model: Optional[RandomForestClassifier] = None
    trained: bool = False
    feature_order: Sequence[str] = field(
        default_factory=lambda: [
            "bbox_area",
            "mean_intensity",
            "std_intensity",
            "laplacian_var",
            "edge_density",
            "aspect_ratio",
            "contour_area",
            "circularity",
            "solidity",
        ]
    )
    positive_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    negative_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def fit_from_labeled_frames(
        self,
        labeled_frames: Iterable[Tuple[np.ndarray, Iterable[Box], Iterable[Box], int]],
    ) -> Dict[str, Dict[str, float]]:
        pos_features: List[Dict[str, float]] = []
        neg_features: List[Dict[str, float]] = []
        positive_frame_indices = set()
        negative_frame_indices = set()

        for frame, positive_boxes, negative_boxes, frame_index in labeled_frames:
            frame_pos = [compute_patch_features(frame, box) for box in positive_boxes]
            frame_neg = [compute_patch_features(frame, box) for box in negative_boxes]
            if frame_pos:
                positive_frame_indices.add(frame_index)
            if frame_neg:
                negative_frame_indices.add(frame_index)
            pos_features.extend(frame_pos)
            neg_features.extend(frame_neg)

        if not pos_features:
            raise ValueError("At least one positive example is required.")

        self._fit_feature_sets(pos_features, neg_features)
        self.summary["positive_example_count"] = {"count": float(len(pos_features))}
        self.summary["negative_example_count"] = {"count": float(len(neg_features))}
        self.summary["positive_annotated_frames"] = {"count": float(len(positive_frame_indices))}
        self.summary["negative_annotated_frames"] = {"count": float(len(negative_frame_indices))}
        return self.summary

    def fit_from_boxes(
        self,
        frame: np.ndarray,
        positive_boxes: Iterable[Box],
        negative_boxes: Iterable[Box],
    ) -> Dict[str, Dict[str, float]]:
        pos_features = [compute_patch_features(frame, box) for box in positive_boxes]
        neg_features = [compute_patch_features(frame, box) for box in negative_boxes]
        if not pos_features:
            raise ValueError("At least one positive example is required.")

        self._fit_feature_sets(pos_features, neg_features)
        return self.summary

    def _fit_feature_sets(
        self,
        pos_features: List[Dict[str, float]],
        neg_features: List[Dict[str, float]],
    ) -> None:
        self.positive_ranges = self._estimate_ranges(pos_features)
        self.negative_ranges = self._estimate_ranges(neg_features) if neg_features else {}
        self.summary = {
            "positive_feature_ranges": {
                name: {"min": bounds[0], "max": bounds[1]} for name, bounds in self.positive_ranges.items()
            },
            "negative_feature_ranges": {
                name: {"min": bounds[0], "max": bounds[1]} for name, bounds in self.negative_ranges.items()
            },
        }

        all_features = pos_features + neg_features
        labels = np.array([1] * len(pos_features) + [0] * len(neg_features))
        if len(pos_features) >= 2 and len(neg_features) >= 2:
            x = np.array([[features[name] for name in self.feature_order] for features in all_features], dtype=float)
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                random_state=42,
                class_weight="balanced",
            )
            model.fit(x, labels)
            self.model = model
            self.trained = True
        else:
            self.model = None
            self.trained = False

    def score_features(self, features: Dict[str, float]) -> float:
        if self.trained and self.model is not None:
            x = np.array([[features[name] for name in self.feature_order]], dtype=float)
            return float(self.model.predict_proba(x)[0, 1])

        score = 0.0
        feature_count = 0
        for name, bounds in self.positive_ranges.items():
            value = features.get(name, 0.0)
            if bounds[0] <= value <= bounds[1]:
                score += 1.0
            else:
                span = max(bounds[1] - bounds[0], 1e-6)
                distance = min(abs(value - bounds[0]), abs(value - bounds[1]))
                score += max(0.0, 1.0 - distance / span)
            feature_count += 1
        for name, bounds in self.negative_ranges.items():
            value = features.get(name, 0.0)
            if bounds[0] <= value <= bounds[1]:
                score -= 0.5
        if feature_count == 0:
            return 0.5
        return float(np.clip(score / feature_count, 0.0, 1.0))

    @staticmethod
    def _estimate_ranges(feature_rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        if not feature_rows:
            return {}
        ranges: Dict[str, Tuple[float, float]] = {}
        for name in feature_rows[0].keys():
            values = np.array([row[name] for row in feature_rows], dtype=float)
            lower = float(np.percentile(values, 5))
            upper = float(np.percentile(values, 95))
            if lower == upper:
                padding = max(abs(lower) * 0.1, 1e-3)
                lower -= padding
                upper += padding
            ranges[name] = (lower, upper)
        return ranges
