from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Box = Tuple[int, int, int, int]


def clip_box(box: Box, width: int, height: int) -> Box:
    x, y, w, h = box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def line_length(point_a: Point, point_b: Point) -> float:
    return float(math.dist(point_a, point_b))


def point_in_box(point: Point, box: Box) -> bool:
    x, y = point
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def box_to_mask(box: Optional[Box], shape: Sequence[int]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if box is None:
        mask[:, :] = 255
        return mask
    x, y, w, h = clip_box(box, shape[1], shape[0])
    mask[y : y + h, x : x + w] = 255
    return mask


def rect_from_canvas_object(obj: dict, width: int, height: int) -> Optional[Box]:
    if obj.get("type") != "rect":
        return None
    x = int(round(obj.get("left", 0)))
    y = int(round(obj.get("top", 0)))
    w = int(round(obj.get("width", 0) * obj.get("scaleX", 1)))
    h = int(round(obj.get("height", 0) * obj.get("scaleY", 1)))
    if w <= 0 or h <= 0:
        return None
    return clip_box((x, y, w, h), width, height)


def line_from_canvas_object(obj: dict) -> Optional[Tuple[Point, Point]]:
    if obj.get("type") not in {"line", "path"}:
        return None
    if obj.get("type") == "line":
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        scale_x = float(obj.get("scaleX", 1))
        scale_y = float(obj.get("scaleY", 1))
        x1 = left + float(obj.get("x1", 0)) * scale_x
        y1 = top + float(obj.get("y1", 0)) * scale_y
        x2 = left + float(obj.get("x2", 0)) * scale_x
        y2 = top + float(obj.get("y2", 0)) * scale_y
        return (x1, y1), (x2, y2)
    path = obj.get("path", [])
    points: List[Point] = []
    for item in path:
        if len(item) >= 3 and item[0] in {"M", "L"}:
            points.append((float(item[1]), float(item[2])))
    if len(points) >= 2:
        return points[0], points[-1]
    return None


def crop_box(frame: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = clip_box(box, frame.shape[1], frame.shape[0])
    return frame[y : y + h, x : x + w].copy()


def centroid_from_contour(contour: np.ndarray) -> Optional[Point]:
    import cv2

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def polyline_length(points: Iterable[Point]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    return float(sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])))
