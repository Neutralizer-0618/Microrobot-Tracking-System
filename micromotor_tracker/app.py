from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from micromotor_tracker.core.analysis import compute_track_statistics
from micromotor_tracker.core.calibration import compute_calibration
from micromotor_tracker.core.detection import run_detection
from micromotor_tracker.core.export import export_results
from micromotor_tracker.core.tracking import run_tracking
from micromotor_tracker.core.video_io import load_video_source, save_uploaded_file
from micromotor_tracker.core.visualization import (
    build_speed_distribution_table,
    create_motility_ratio_figure,
    create_speed_distribution_figure,
    draw_detection_overlay,
    draw_track_overlay,
)
from micromotor_tracker.models.interactive_classifier import InteractiveClassifier
from micromotor_tracker.utils.canvas_compat import st_canvas
from micromotor_tracker.utils.config import (
    AnalysisConfig,
    CalibrationConfig,
    DetectionConfig,
    TrackingConfig,
    serialize_parameters,
)
from micromotor_tracker.utils.geometry import Box, line_from_canvas_object, rect_from_canvas_object


st.set_page_config(page_title="MicroMotorTracker-AI", layout="wide")

CALIBRATION_COLOR = "#ff006e"
ROI_COLOR = "#3a86ff"
POSITIVE_COLOR = "#ff9f1c"
NEGATIVE_COLOR = "#d90429"


def init_state() -> None:
    defaults = {
        "uploaded_name": None,
        "video_source": None,
        "annotation_mode": "inspect",
        "canvas_revision": 0,
        "canvas_drawings": {},
        "last_canvas_seed_key": None,
        "roi_box": None,
        "roi_frame_index": None,
        "calibration": None,
        "calibration_frame_index": None,
        "positive_boxes_by_frame": {},
        "negative_boxes_by_frame": {},
        "classifier": None,
        "classifier_summary": None,
        "detections": pd.DataFrame(),
        "track_rows": pd.DataFrame(),
        "track_meta": pd.DataFrame(),
        "track_stats": pd.DataFrame(),
        "speed_table": pd.DataFrame(),
        "population_summary": pd.DataFrame(),
        "export_paths": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    st.session_state["detections"] = pd.DataFrame()
    st.session_state["track_rows"] = pd.DataFrame()
    st.session_state["track_meta"] = pd.DataFrame()
    st.session_state["track_stats"] = pd.DataFrame()
    st.session_state["speed_table"] = pd.DataFrame()
    st.session_state["population_summary"] = pd.DataFrame()
    st.session_state["export_paths"] = None


def bump_canvas_revision() -> None:
    st.session_state["canvas_revision"] += 1


def scale_frame_for_canvas(frame, max_width: int = 900) -> tuple[Image.Image, float, float]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    if width <= max_width:
        return Image.fromarray(rgb), 1.0, 1.0
    scale = max_width / width
    resized = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized), width / resized.shape[1], height / resized.shape[0]


def box_to_original(box: Box, scale_x: float, scale_y: float) -> Box:
    x, y, w, h = box
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        int(round(w * scale_x)),
        int(round(h * scale_y)),
    )


def point_to_original(point: Tuple[float, float], scale_x: float, scale_y: float) -> Tuple[float, float]:
    return point[0] * scale_x, point[1] * scale_y


def canvas_state_key(mode: str, frame_index: int) -> str:
    return f"{mode}:{frame_index}"


def clear_canvas_drawings(mode: str, frame_index: Optional[int] = None) -> None:
    if frame_index is None:
        st.session_state["canvas_drawings"] = {
            key: value for key, value in st.session_state["canvas_drawings"].items() if not key.startswith(f"{mode}:")
        }
        return
    st.session_state["canvas_drawings"].pop(canvas_state_key(mode, frame_index), None)


def make_rect_object(box: Box, scale_x: float, scale_y: float, stroke: str) -> dict:
    x, y, w, h = box
    return {
        "type": "rect",
        "version": "4.4.0",
        "originX": "left",
        "originY": "top",
        "left": float(x / scale_x),
        "top": float(y / scale_y),
        "width": float(w / scale_x),
        "height": float(h / scale_y),
        "scaleX": 1.0,
        "scaleY": 1.0,
        "fill": "rgba(0,0,0,0)",
        "stroke": stroke,
        "strokeWidth": 2,
    }


def make_line_object(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    scale_x: float,
    scale_y: float,
    stroke: str,
) -> dict:
    return {
        "type": "line",
        "version": "4.4.0",
        "left": 0.0,
        "top": 0.0,
        "x1": float(point_a[0] / scale_x),
        "y1": float(point_a[1] / scale_y),
        "x2": float(point_b[0] / scale_x),
        "y2": float(point_b[1] / scale_y),
        "scaleX": 1.0,
        "scaleY": 1.0,
        "stroke": stroke,
        "strokeWidth": 3,
    }


def overlay_saved_annotations(frame, frame_index: int, exclude_mode: Optional[str] = None):
    canvas = frame.copy()
    if st.session_state["roi_box"] is not None and exclude_mode != "roi":
        x, y, w, h = st.session_state["roi_box"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 170, 20), 2)
    calibration = st.session_state["calibration"]
    if (
        calibration is not None
        and st.session_state["calibration_frame_index"] == frame_index
        and exclude_mode != "calibration"
    ):
        p1 = (int(round(calibration.point_a[0])), int(round(calibration.point_a[1])))
        p2 = (int(round(calibration.point_b[0])), int(round(calibration.point_b[1])))
        cv2.line(canvas, p1, p2, (255, 0, 180), 2)
    if exclude_mode != "positive":
        for x, y, w, h in st.session_state["positive_boxes_by_frame"].get(frame_index, []):
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 160, 255), 2)
    if exclude_mode != "negative":
        for x, y, w, h in st.session_state["negative_boxes_by_frame"].get(frame_index, []):
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 20, 220), 2)
    return canvas


def build_mode_initial_drawing(mode: str, frame_index: int, scale_x: float, scale_y: float) -> dict:
    objects: List[dict] = []
    if mode == "roi" and st.session_state["roi_box"] is not None:
        objects.append(make_rect_object(st.session_state["roi_box"], scale_x, scale_y, ROI_COLOR))
    elif mode == "calibration":
        calibration = st.session_state["calibration"]
        if calibration is not None and st.session_state["calibration_frame_index"] == frame_index:
            objects.append(make_line_object(calibration.point_a, calibration.point_b, scale_x, scale_y, CALIBRATION_COLOR))
    elif mode == "positive":
        for box in st.session_state["positive_boxes_by_frame"].get(frame_index, []):
            objects.append(make_rect_object(box, scale_x, scale_y, POSITIVE_COLOR))
    elif mode == "negative":
        for box in st.session_state["negative_boxes_by_frame"].get(frame_index, []):
            objects.append(make_rect_object(box, scale_x, scale_y, NEGATIVE_COLOR))
    return {"version": "4.4.0", "objects": objects}


def parse_mode_rectangles_from_json(json_data, frame_shape, scale_x: float, scale_y: float) -> Optional[List[Box]]:
    if json_data is None:
        return None
    width = int(round(frame_shape[1] / scale_x))
    height = int(round(frame_shape[0] / scale_y))
    boxes: List[Box] = []
    for obj in json_data.get("objects", []):
        if obj.get("type") != "rect":
            continue
        box = rect_from_canvas_object(obj, width, height)
        if box is None:
            continue
        boxes.append(box_to_original(box, scale_x, scale_y))
    return boxes


def parse_mode_lines_from_json(json_data, scale_x: float, scale_y: float) -> Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    if json_data is None:
        return None
    lines = []
    for obj in json_data.get("objects", []):
        line = line_from_canvas_object(obj)
        if line is None:
            continue
        lines.append((point_to_original(line[0], scale_x, scale_y), point_to_original(line[1], scale_x, scale_y)))
    return lines


def collect_labeled_samples(video_source, positive_boxes_by_frame, negative_boxes_by_frame):
    frame_indices = sorted(set(positive_boxes_by_frame.keys()) | set(negative_boxes_by_frame.keys()))
    samples = []
    for sample_frame_index in frame_indices:
        samples.append(
            (
                video_source.get_frame(sample_frame_index),
                positive_boxes_by_frame.get(sample_frame_index, []),
                negative_boxes_by_frame.get(sample_frame_index, []),
                sample_frame_index,
            )
        )
    return samples


def set_annotation_mode(mode: str) -> None:
    st.session_state["annotation_mode"] = mode
    bump_canvas_revision()


def mode_config(mode: str) -> tuple[str, str, str]:
    if mode == "calibration":
        return "line", CALIBRATION_COLOR, "Calibration mode: draw a line directly on the current video frame."
    if mode == "roi":
        return "rect", ROI_COLOR, "ROI mode: draw a rectangle directly on the current video frame."
    if mode == "positive":
        return "rect", POSITIVE_COLOR, "Positive mode: draw orange boxes around micromotors."
    if mode == "negative":
        return "rect", NEGATIVE_COLOR, "Negative mode: draw red boxes around debris or artifacts."
    return "transform", "#ffffff", "Inspect mode: browse frames and review overlays."


init_state()

st.title("MicroMotorTracker-AI")
st.write(
    "Local Streamlit app for microscopy video loading, calibration, example-guided micromotor detection, tracking, motility analysis, and export."
)
st.caption("UI build: 2026-06-13-canvas-fix")

uploaded_file = st.file_uploader("Load video", type=["mp4", "avi", "mov", "tif", "tiff"])

if uploaded_file is not None and uploaded_file.name != st.session_state["uploaded_name"]:
    saved_path = save_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
    st.session_state["video_source"] = load_video_source(saved_path)
    st.session_state["uploaded_name"] = uploaded_file.name
    st.session_state["annotation_mode"] = "inspect"
    st.session_state["canvas_revision"] = 0
    st.session_state["canvas_drawings"] = {}
    st.session_state["last_canvas_seed_key"] = None
    st.session_state["roi_box"] = None
    st.session_state["roi_frame_index"] = None
    st.session_state["calibration"] = None
    st.session_state["calibration_frame_index"] = None
    st.session_state["positive_boxes_by_frame"] = {}
    st.session_state["negative_boxes_by_frame"] = {}
    st.session_state["classifier"] = None
    st.session_state["classifier_summary"] = None
    reset_analysis_state()

video_source = st.session_state["video_source"]
if video_source is None:
    st.info("Load a microscopy video or TIFF stack to begin.")
    st.stop()

metadata = video_source.metadata
default_fps = float(metadata.fps if metadata.fps_reliable and metadata.fps > 0 else 30.0)
fps_value = st.number_input("Frame rate (fps)", min_value=0.1, value=default_fps, step=0.1)
frame_index = st.slider("Frame navigation", 0, max(metadata.frame_count - 1, 0), 0, 1)
frame = video_source.get_frame(frame_index)

if not metadata.fps_reliable:
    st.warning("The file FPS was unavailable or unreliable. Please confirm the manual FPS value above before analysis.")

info_cols = st.columns(5)
info_cols[0].metric("Frames", metadata.frame_count)
info_cols[1].metric("FPS", f"{fps_value:.2f}")
info_cols[2].metric("Duration", f"{metadata.frame_count / fps_value:.2f} s")
info_cols[3].metric("Width", metadata.width)
info_cols[4].metric("Height", metadata.height)

toolbar_cols = st.columns(5)
if toolbar_cols[0].button("Inspect"):
    set_annotation_mode("inspect")
if toolbar_cols[1].button("Set fps and calibration"):
    set_annotation_mode("calibration")
if toolbar_cols[2].button("Select ROI"):
    set_annotation_mode("roi")
if toolbar_cols[3].button("Select positive"):
    set_annotation_mode("positive")
if toolbar_cols[4].button("Select negative"):
    set_annotation_mode("negative")

annotation_mode = st.session_state["annotation_mode"]
drawing_mode, stroke_color, mode_message = mode_config(annotation_mode)
st.caption(f"Current mode: `{annotation_mode}`")
st.caption(mode_message)

background_frame = overlay_saved_annotations(frame, frame_index, exclude_mode=annotation_mode if annotation_mode != "inspect" else None)
canvas_image, canvas_scale_x, canvas_scale_y = scale_frame_for_canvas(background_frame)
active_canvas_state_key = canvas_state_key(annotation_mode, frame_index)
component_canvas_key = f"main_canvas_{frame_index}_{annotation_mode}_{st.session_state['canvas_revision']}"
should_seed_canvas = st.session_state.get("last_canvas_seed_key") != component_canvas_key
initial_drawing = None
if should_seed_canvas:
    initial_drawing = st.session_state["canvas_drawings"].get(active_canvas_state_key)
    if initial_drawing is None:
        initial_drawing = build_mode_initial_drawing(annotation_mode, frame_index, canvas_scale_x, canvas_scale_y)
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=3 if annotation_mode == "calibration" else 2,
    stroke_color=stroke_color,
    background_image=canvas_image,
    update_streamlit=True,
    height=canvas_image.height,
    width=canvas_image.width,
    drawing_mode=drawing_mode,
    initial_drawing=initial_drawing,
    key=component_canvas_key,
)
st.session_state["last_canvas_seed_key"] = component_canvas_key
if annotation_mode != "inspect" and canvas_result.json_data is not None:
    st.session_state["canvas_drawings"][active_canvas_state_key] = canvas_result.json_data

current_canvas_json = canvas_result.json_data or st.session_state["canvas_drawings"].get(active_canvas_state_key)

if annotation_mode == "positive":
    parsed_positive = parse_mode_rectangles_from_json(current_canvas_json, frame.shape, canvas_scale_x, canvas_scale_y)
    if parsed_positive is not None:
        st.session_state["positive_boxes_by_frame"][frame_index] = parsed_positive
elif annotation_mode == "negative":
    parsed_negative = parse_mode_rectangles_from_json(current_canvas_json, frame.shape, canvas_scale_x, canvas_scale_y)
    if parsed_negative is not None:
        st.session_state["negative_boxes_by_frame"][frame_index] = parsed_negative

current_positive = st.session_state["positive_boxes_by_frame"].get(frame_index, [])
current_negative = st.session_state["negative_boxes_by_frame"].get(frame_index, [])
total_positive = sum(len(boxes) for boxes in st.session_state["positive_boxes_by_frame"].values())
total_negative = sum(len(boxes) for boxes in st.session_state["negative_boxes_by_frame"].values())
detections = st.session_state["detections"]
track_rows = st.session_state["track_rows"]
track_stats = st.session_state["track_stats"]
population_summary = st.session_state["population_summary"]

panel = st.container(border=True)
with panel:
    if annotation_mode == "calibration":
        st.subheader("Set fps and calibration")
        calibration_cols = st.columns([1, 1, 2])
        with calibration_cols[0]:
            real_length_um = st.number_input("Real-world line length (um)", min_value=0.01, value=10.0, step=0.1)
        with calibration_cols[1]:
            if st.button("Save calibration from frame"):
                parsed_lines = parse_mode_lines_from_json(current_canvas_json, canvas_scale_x, canvas_scale_y)
                if not parsed_lines:
                    st.error("Draw a calibration line on the current frame first.")
                else:
                    line = parsed_lines[-1]
                    st.session_state["calibration"] = compute_calibration(line[0], line[1], real_length_um)
                    st.session_state["calibration_frame_index"] = frame_index
                    clear_canvas_drawings("calibration", frame_index)
                    bump_canvas_revision()
                    st.success("Calibration saved for the full video.")
        with calibration_cols[2]:
            calibration = st.session_state["calibration"]
            if calibration is not None:
                st.caption(
                    f"Saved calibration: {calibration.micron_per_pixel:.6f} um/px, "
                    f"real length {calibration.real_length_um:.2f} um, frame {st.session_state['calibration_frame_index']}"
                )
            else:
                st.caption("No calibration saved yet.")

    elif annotation_mode == "roi":
        st.subheader("Select ROI")
        roi_cols = st.columns([1, 1, 2])
        with roi_cols[0]:
            if st.button("Save ROI from frame"):
                parsed_boxes = parse_mode_rectangles_from_json(current_canvas_json, frame.shape, canvas_scale_x, canvas_scale_y)
                if not parsed_boxes:
                    st.error("Draw an ROI rectangle on the current frame first.")
                else:
                    st.session_state["roi_box"] = parsed_boxes[-1]
                    st.session_state["roi_frame_index"] = frame_index
                    clear_canvas_drawings("roi", frame_index)
                    bump_canvas_revision()
                    reset_analysis_state()
                    st.success("ROI saved for the full video.")
        with roi_cols[1]:
            if st.button("Clear ROI"):
                st.session_state["roi_box"] = None
                st.session_state["roi_frame_index"] = None
                clear_canvas_drawings("roi")
                bump_canvas_revision()
                reset_analysis_state()
                st.success("ROI cleared.")
        with roi_cols[2]:
            if st.session_state["roi_box"] is not None:
                x, y, w, h = st.session_state["roi_box"]
                st.caption(f"Saved ROI: x={x}, y={y}, width={w}, height={h}, frame {st.session_state['roi_frame_index']}")
            else:
                st.caption("No ROI saved yet.")

    elif annotation_mode == "positive":
        st.subheader("Select Positive Examples")
        sample_cols = st.columns(4)
        with sample_cols[0]:
            st.caption(f"Current frame positive boxes: {len(current_positive)}")
            if st.button("Clear current positive"):
                st.session_state["positive_boxes_by_frame"][frame_index] = []
                st.session_state["classifier"] = None
                st.session_state["classifier_summary"] = None
                clear_canvas_drawings("positive", frame_index)
                bump_canvas_revision()
                reset_analysis_state()
        with sample_cols[1]:
            st.caption(f"All positive boxes: {total_positive}")
            if st.button("Clear all positive"):
                st.session_state["positive_boxes_by_frame"] = {}
                st.session_state["classifier"] = None
                st.session_state["classifier_summary"] = None
                clear_canvas_drawings("positive")
                bump_canvas_revision()
                reset_analysis_state()
        with sample_cols[2]:
            st.caption("Current positive annotations are now preserved as raw canvas state.")
        with sample_cols[3]:
            if st.button("Estimate object model"):
                try:
                    classifier = InteractiveClassifier()
                    labeled_samples = collect_labeled_samples(
                        video_source,
                        st.session_state["positive_boxes_by_frame"],
                        st.session_state["negative_boxes_by_frame"],
                    )
                    summary = classifier.fit_from_labeled_frames(labeled_samples)
                    st.session_state["classifier"] = classifier
                    st.session_state["classifier_summary"] = summary
                    reset_analysis_state()
                    st.success("Interactive classifier updated from the annotated boxes across frames.")
                except Exception as exc:
                    st.error(str(exc))
        st.caption(
            "Why the second box used to flash: the app was rebuilding the positive rectangles from parsed boxes on every rerun, "
            "which reset the canvas state. The current version keeps the raw canvas drawing state for this frame, so adding a third box should remain stable."
        )

    elif annotation_mode == "negative":
        st.subheader("Select Negative Examples")
        sample_cols = st.columns(4)
        with sample_cols[0]:
            st.caption(f"Current frame negative boxes: {len(current_negative)}")
            if st.button("Clear current negative"):
                st.session_state["negative_boxes_by_frame"][frame_index] = []
                st.session_state["classifier"] = None
                st.session_state["classifier_summary"] = None
                clear_canvas_drawings("negative", frame_index)
                bump_canvas_revision()
                reset_analysis_state()
        with sample_cols[1]:
            st.caption(f"All negative boxes: {total_negative}")
            if st.button("Clear all negative"):
                st.session_state["negative_boxes_by_frame"] = {}
                st.session_state["classifier"] = None
                st.session_state["classifier_summary"] = None
                clear_canvas_drawings("negative")
                bump_canvas_revision()
                reset_analysis_state()
        with sample_cols[2]:
            st.caption("Negative annotations share the same preserved raw canvas behavior.")
        with sample_cols[3]:
            if st.button("Estimate object model "):
                try:
                    classifier = InteractiveClassifier()
                    labeled_samples = collect_labeled_samples(
                        video_source,
                        st.session_state["positive_boxes_by_frame"],
                        st.session_state["negative_boxes_by_frame"],
                    )
                    summary = classifier.fit_from_labeled_frames(labeled_samples)
                    st.session_state["classifier"] = classifier
                    st.session_state["classifier_summary"] = summary
                    reset_analysis_state()
                    st.success("Interactive classifier updated from the annotated boxes across frames.")
                except Exception as exc:
                    st.error(str(exc))

    else:
        st.subheader("Inspect")
        status_cols = st.columns(4)
        status_cols[0].metric("Calibration", "Saved" if st.session_state["calibration"] is not None else "Missing")
        status_cols[1].metric("ROI", "Saved" if st.session_state["roi_box"] is not None else "Off")
        status_cols[2].metric("Positive Boxes", total_positive)
        status_cols[3].metric("Negative Boxes", total_negative)

        if st.session_state["classifier_summary"] is not None:
            with st.expander("Classifier summary", expanded=False):
                st.json(st.session_state["classifier_summary"])

        st.subheader("Analysis frame range")
        range_cols = st.columns(3)
        analysis_frame_start = int(
            range_cols[0].number_input("Start frame", min_value=0, max_value=max(metadata.frame_count - 1, 0), value=0, step=1)
        )
        analysis_frame_end = int(
            range_cols[1].number_input(
                "End frame",
                min_value=analysis_frame_start,
                max_value=max(metadata.frame_count - 1, analysis_frame_start),
                value=max(metadata.frame_count - 1, analysis_frame_start),
                step=1,
            )
        )
        range_cols[2].caption(
            f"Only frames {analysis_frame_start} to {analysis_frame_end} will be used for detection, tracking, speed analysis, and export."
        )

        st.subheader("Configure detection parameters")
        det_cols = st.columns(4)
        threshold_method = det_cols[0].selectbox("Threshold method", ["otsu", "adaptive", "manual"])
        manual_threshold = det_cols[1].slider("Manual threshold", 0, 255, 128)
        min_area = det_cols[2].number_input("Min area", min_value=1.0, value=20.0, step=1.0)
        max_area = det_cols[3].number_input("Max area", min_value=2.0, value=5000.0, step=10.0)

        det_cols2 = st.columns(4)
        blur_kernel = det_cols2[0].slider("Blur kernel", 1, 31, 5, step=2)
        bg_sub = det_cols2[1].checkbox("Background subtraction", value=True)
        min_circularity = det_cols2[2].slider("Min circularity", 0.0, 1.5, 0.05, 0.01)
        max_circularity = det_cols2[3].slider("Max circularity", 0.1, 2.0, 1.5, 0.01)

        det_cols3 = st.columns(4)
        min_aspect_ratio = det_cols3[0].number_input("Min aspect ratio", min_value=0.05, value=0.2, step=0.05)
        max_aspect_ratio = det_cols3[1].number_input("Max aspect ratio", min_value=0.1, value=5.0, step=0.1)
        min_confidence = det_cols3[2].slider("Min confidence", 0.0, 1.0, 0.2, 0.01)
        adaptive_block_size = det_cols3[3].slider("Adaptive block size", 3, 101, 31, step=2)

        detection_config = DetectionConfig(
            threshold_method=threshold_method,
            manual_threshold=manual_threshold,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=5,
            blur_kernel=blur_kernel,
            use_background_subtraction=bg_sub,
            background_blur_kernel=31,
            min_area=min_area,
            max_area=max_area,
            min_circularity=min_circularity,
            max_circularity=max_circularity,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            min_confidence=min_confidence,
        )

        if st.button("Run detection", type="primary"):
            progress_bar = st.progress(0.0, text="Running detection...")

            def update_progress(current: int, total: int) -> None:
                progress_bar.progress(min(current / max(total, 1), 1.0), text=f"Running detection... {current}/{total} frames")

            try:
                detections = run_detection(
                    video_source,
                    detection_config,
                    classifier=st.session_state["classifier"],
                    roi_box=st.session_state["roi_box"],
                    progress_callback=update_progress,
                    frame_start=analysis_frame_start,
                    frame_end=analysis_frame_end,
                )
                st.session_state["detections"] = detections
                st.session_state["track_rows"] = pd.DataFrame()
                st.session_state["track_meta"] = pd.DataFrame()
                st.session_state["track_stats"] = pd.DataFrame()
                st.session_state["speed_table"] = pd.DataFrame()
                st.session_state["population_summary"] = pd.DataFrame()
                progress_bar.empty()
                st.success(f"Detection complete: {len(detections)} detections.")
            except Exception as exc:
                progress_bar.empty()
                st.error(f"Detection failed: {exc}")

        detections = st.session_state["detections"]
        if not detections.empty:
            frame_detections = detections[detections["frame"] == frame_index]
            with st.expander("Detection review", expanded=False):
                st.image(
                    cv2.cvtColor(draw_detection_overlay(frame, frame_detections, roi_box=st.session_state["roi_box"]), cv2.COLOR_BGR2RGB),
                    caption="Detection overlay",
                    use_container_width=True,
                )
                st.dataframe(frame_detections.head(50), use_container_width=True)

        st.subheader("Run tracking")
        track_cols = st.columns(4)
        max_link_distance = track_cols[0].number_input("Max linking distance (px)", min_value=1.0, value=40.0, step=1.0)
        max_frame_gap = track_cols[1].number_input("Max frame gap", min_value=0, value=3, step=1)
        min_track_length = track_cols[2].number_input("Min track length", min_value=2, value=5, step=1)
        low_conf_threshold = track_cols[3].slider("Low-confidence track threshold", 0.0, 1.0, 0.45, 0.01)

        tracking_config = TrackingConfig(
            max_link_distance=max_link_distance,
            max_frame_gap=int(max_frame_gap),
            min_track_length=int(min_track_length),
            min_object_area=min_area,
            max_object_area=max_area,
            low_confidence_threshold=low_conf_threshold,
        )

        analysis_cols = st.columns(4)
        speed_window_frames = int(analysis_cols[0].number_input("Speed window (frames)", min_value=2, value=20, step=1))
        motility_speed_threshold = analysis_cols[1].number_input("Motility threshold (um/s)", min_value=0.0, value=10.0, step=0.1)
        speed_bin_min = float(analysis_cols[2].number_input("Speed bin lower (um/s)", value=0.0, step=1.0))
        speed_bin_width = float(analysis_cols[3].number_input("Speed bin width (um/s)", min_value=0.1, value=10.0, step=1.0))
        analysis_cols2 = st.columns(2)
        speed_bin_max = float(analysis_cols2[0].number_input("Speed bin upper (um/s)", min_value=speed_bin_min + speed_bin_width, value=50.0, step=1.0))
        analysis_cols2[1].caption(
            "Speed is now measured in non-overlapping frame windows. By default, each 20-frame segment contributes one speed value; leftover segments shorter than the window are ignored."
        )

        analysis_config = AnalysisConfig(
            active_speed_threshold_um_s=motility_speed_threshold,
            min_active_duration_s=0.0,
            active_mode="mean_speed",
            active_net_displacement_threshold_um_s=motility_speed_threshold,
            speed_window_frames=speed_window_frames,
            speed_bin_min_um_s=speed_bin_min,
            speed_bin_max_um_s=speed_bin_max,
            speed_bin_width_um_s=speed_bin_width,
            motility_speed_threshold_um_s=motility_speed_threshold,
        )

        if st.button("Run tracking and motility analysis"):
            if detections.empty:
                st.error("Run detection first.")
            else:
                try:
                    track_rows, track_meta = run_tracking(detections, tracking_config)
                    calibration_obj = st.session_state["calibration"]
                    micron_per_pixel = calibration_obj.micron_per_pixel if calibration_obj else None
                    track_stats, speed_table, population_summary = compute_track_statistics(
                        track_rows,
                        fps=fps_value,
                        micron_per_pixel=micron_per_pixel,
                        analysis_config=analysis_config,
                        track_meta=track_meta,
                    )
                    st.session_state["track_rows"] = track_rows
                    st.session_state["track_meta"] = track_meta
                    st.session_state["track_stats"] = track_stats
                    st.session_state["speed_table"] = speed_table
                    st.session_state["population_summary"] = population_summary
                    st.success(
                        f"Tracking complete: {track_stats['track_id'].nunique() if not track_stats.empty else 0} valid tracks, "
                        f"{len(speed_table)} valid speed measurements."
                    )
                except Exception as exc:
                    st.error(f"Tracking failed: {exc}")

        track_rows = st.session_state["track_rows"]
        track_stats = st.session_state["track_stats"]
        population_summary = st.session_state["population_summary"]

        st.subheader("Results")
        if not track_rows.empty:
            speed_distribution = build_speed_distribution_table(
                st.session_state["speed_table"],
                speed_min=analysis_config.speed_bin_min_um_s,
                speed_max=analysis_config.speed_bin_max_um_s,
                bin_width=analysis_config.speed_bin_width_um_s,
            )
            motile_measurements = int(population_summary.loc[0, "motile_measurements"]) if not population_summary.empty else 0
            total_measurements = int(population_summary.loc[0, "total_speed_measurements"]) if not population_summary.empty else 0
            mean_speed = float(population_summary.loc[0, "mean_speed_um_s"]) if not population_summary.empty else float("nan")
            speed_variance = float(population_summary.loc[0, "speed_variance_um_s"]) if not population_summary.empty else float("nan")

            summary_cols = st.columns(3)
            summary_cols[0].metric("Speed measurements", total_measurements)
            summary_cols[1].metric("Motility ratio (%)", f"{population_summary.loc[0, 'motility_ratio_pct']:.2f}" if not population_summary.empty and pd.notna(population_summary.loc[0, "motility_ratio_pct"]) else "N/A")
            summary_cols[2].metric("Mean speed ± variance (um/s)", f"{mean_speed:.2f} ± {speed_variance:.2f}" if pd.notna(mean_speed) and pd.notna(speed_variance) else "N/A")

            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.pyplot(create_speed_distribution_figure(speed_distribution), use_container_width=True)
            with chart_cols[1]:
                st.pyplot(create_motility_ratio_figure(motile_measurements, total_measurements), use_container_width=True)

            with st.expander("Track overlay and detailed tables", expanded=False):
                st.image(
                    cv2.cvtColor(draw_track_overlay(frame, track_rows, frame_index, roi_box=st.session_state["roi_box"]), cv2.COLOR_BGR2RGB),
                    caption="Track overlay",
                    use_container_width=True,
                )
                st.dataframe(track_stats, use_container_width=True)
                st.dataframe(st.session_state["speed_table"], use_container_width=True)
                st.dataframe(population_summary, use_container_width=True)
        else:
            st.info("No tracks available yet.")

        st.subheader("Export results")
        default_output_dir = str((Path(__file__).resolve().parents[1] / "outputs").resolve())
        output_dir = st.text_input("Output base folder", value=default_output_dir)
        if st.button("Export annotated video, CSV, and plots"):
            if track_rows.empty or track_stats.empty:
                st.error("Run tracking first so there is something to export.")
            else:
                calibration_obj = st.session_state["calibration"]
                calibration_config = (
                    CalibrationConfig(
                        line_length_pixels=calibration_obj.line_length_pixels,
                        real_length_um=calibration_obj.real_length_um,
                        micron_per_pixel=calibration_obj.micron_per_pixel,
                    )
                    if calibration_obj
                    else CalibrationConfig()
                )
                parameters = serialize_parameters(
                    detection=detection_config,
                    tracking=tracking_config,
                    analysis=analysis_config,
                    calibration=calibration_config,
                    extra={
                        "fps_used": fps_value,
                        "roi_box": st.session_state["roi_box"],
                        "roi_frame_index": st.session_state["roi_frame_index"],
                        "calibration_frame_index": st.session_state["calibration_frame_index"],
                        "positive_boxes_by_frame": st.session_state["positive_boxes_by_frame"],
                        "negative_boxes_by_frame": st.session_state["negative_boxes_by_frame"],
                        "classifier_summary": st.session_state["classifier_summary"],
                    },
                )
                try:
                    export_paths = export_results(
                        video_source=video_source,
                        fps=fps_value,
                        micron_per_pixel=calibration_obj.micron_per_pixel if calibration_obj else None,
                        calibration_metadata=calibration_obj.to_dict() if calibration_obj else {},
                        analysis_parameters=parameters,
                        detections=detections,
                        track_rows=track_rows,
                        track_stats=track_stats,
                        population_summary=population_summary,
                        speed_table=st.session_state["speed_table"],
                        roi_box=st.session_state["roi_box"],
                        output_dir=output_dir,
                        frame_start=analysis_frame_start,
                        frame_end=analysis_frame_end,
                        speed_bin_min_um_s=analysis_config.speed_bin_min_um_s,
                        speed_bin_max_um_s=analysis_config.speed_bin_max_um_s,
                        speed_bin_width_um_s=analysis_config.speed_bin_width_um_s,
                    )
                    st.session_state["export_paths"] = export_paths
                    st.success(f"Results exported to {export_paths['output_dir']}")
                except Exception as exc:
                    st.error(f"Export failed: {exc}")

if st.session_state["export_paths"] is not None:
    with st.expander("Exported file paths", expanded=False):
        st.json(st.session_state["export_paths"])
