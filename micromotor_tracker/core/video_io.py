from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import tifffile


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".tif", ".tiff"}


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = frame.astype(np.float32)
    min_value = float(frame.min())
    max_value = float(frame.max())
    if max_value <= min_value:
        return np.zeros_like(frame, dtype=np.uint8)
    scaled = (frame - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    frame = _ensure_uint8(frame)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise ValueError("Unsupported frame format.")


@dataclass
class VideoMetadata:
    path: str
    source_name: str
    kind: str
    backend: str
    frame_count: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    fps_reliable: bool

    def with_fps(self, fps: float) -> "VideoMetadata":
        duration = self.frame_count / fps if fps > 0 else 0.0
        return replace(self, fps=float(fps), duration_seconds=duration, fps_reliable=True)


class VideoSource:
    def __init__(self, path: str, metadata: VideoMetadata, tiff_stack: Optional[np.ndarray] = None):
        self.path = path
        self.metadata = metadata
        self._tiff_stack = tiff_stack

    def get_frame(self, index: int) -> np.ndarray:
        index = max(0, min(index, self.metadata.frame_count - 1))
        if self.metadata.kind == "tiff":
            assert self._tiff_stack is not None
            frame = self._tiff_stack[index]
            return _ensure_bgr(frame)
        cap = cv2.VideoCapture(self.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Unable to read frame {index}.")
            return frame
        finally:
            cap.release()

    def iter_frames(self) -> Generator[np.ndarray, None, None]:
        if self.metadata.kind == "tiff":
            assert self._tiff_stack is not None
            for frame in self._tiff_stack:
                yield _ensure_bgr(frame)
            return
        cap = cv2.VideoCapture(self.path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                yield frame
        finally:
            cap.release()


def save_uploaded_file(file_name: str, data: bytes) -> str:
    suffix = Path(file_name).suffix
    temp_dir = Path(tempfile.gettempdir()) / "micromotor_tracker_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file_name
    temp_path.write_bytes(data)
    if temp_path.suffix.lower() != suffix.lower():
        temp_path = temp_path.with_suffix(suffix)
    return str(temp_path)


def load_video_source(path: str) -> VideoSource:
    file_path = Path(path)
    if file_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. Supported types: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )
    if file_path.suffix.lower() in {".tif", ".tiff"}:
        stack = tifffile.imread(path)
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        frame_count = int(stack.shape[0])
        sample = _ensure_bgr(stack[0])
        metadata = VideoMetadata(
            path=str(file_path),
            source_name=file_path.name,
            kind="tiff",
            backend="tifffile",
            frame_count=frame_count,
            fps=1.0,
            duration_seconds=float(frame_count),
            width=int(sample.shape[1]),
            height=int(sample.shape[0]),
            fps_reliable=False,
        )
        return VideoSource(str(file_path), metadata, tiff_stack=stack)

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps_reliable = fps > 0.1
        duration = frame_count / fps if fps_reliable and frame_count > 0 else 0.0
        metadata = VideoMetadata(
            path=str(file_path),
            source_name=file_path.name,
            kind="video",
            backend="opencv",
            frame_count=frame_count,
            fps=fps if fps_reliable else 0.0,
            duration_seconds=duration,
            width=width,
            height=height,
            fps_reliable=fps_reliable,
        )
        return VideoSource(str(file_path), metadata)
    finally:
        cap.release()

