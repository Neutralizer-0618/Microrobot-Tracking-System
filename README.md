# Microrobot Tracking System

Microrobot Tracking System is a local Streamlit application for microscopy-video analysis of micromotors and biohybrid microrobots. It provides a full MVP pipeline from video loading and scale calibration to object detection, multi-object tracking, motility analysis, and export of annotated results.

The project is designed for Windows-first local use, with an emphasis on reproducibility, interpretability, and lightweight classical computer vision rather than heavy cloud or deep-learning dependencies.

## What it does

- Loads microscopy videos in `mp4`, `avi`, `mov`, `tif`, and `tiff`
- Reads frame count, FPS, duration, width, and height
- Allows manual FPS entry when metadata is unreliable
- Supports pixel-to-micron calibration from a user-drawn line
- Supports an ROI that applies to the full video
- Lets the user annotate positive and negative examples on selected frames
- Detects candidate micromotors using classical CV
- Tracks multiple objects over time
- Computes per-track and population-level motility statistics
- Exports CSV, JSON, annotated video, and plot outputs

## Current UI workflow

The current UI uses a single main frame canvas plus a compact top toolbar:

- `Inspect`
- `Set fps and calibration`
- `Select ROI`
- `Select positive`
- `Select negative`

The selected mode controls what you draw on the current frame:

- Calibration mode: draw a reference line
- ROI mode: draw a whole-video analysis box
- Positive mode: draw orange sample boxes around valid micromotors
- Negative mode: draw red sample boxes around debris, bubbles, dust, or other distractors

The same main frame area is reused throughout the workflow so the user stays anchored to the video frame they are currently reviewing.

## Analysis pipeline

1. Load a microscopy video or TIFF stack
2. Confirm or enter FPS
3. Draw a calibration line and enter the real-world length
4. Optionally draw an ROI
5. Annotate positive examples on informative frames
6. Optionally annotate negative examples on one or more frames
7. Estimate an object model from the labeled boxes
8. Tune detection parameters
9. Run detection
10. Run tracking and motility analysis
11. Review overlays and summary tables
12. Export results

## Technical approach

This MVP intentionally prioritizes reliability and interpretability over deep-learning complexity.

### Detection

- Thresholding
- Background subtraction
- Contour-based filtering
- Morphology and size constraints

### Sample-guided learning

- Handcrafted features from labeled boxes
- Intensity, variance, circularity, aspect ratio, solidity, texture, and edge features
- Lightweight classifier: `RandomForestClassifier`

### Tracking

- Centroid-based association
- Hungarian matching
- Short-gap interpolation
- Simple confidence and track-quality flags

## Metrics

For each valid track, the app computes:

- instantaneous speed
- mean speed
- median speed
- maximum speed
- total path length
- net displacement
- track duration
- straightness
- directionality ratio
- tracked frame count

Population-level summaries include:

- total detections
- valid tracks
- mean speed
- median speed
- speed standard deviation
- moving fraction
- inactive fraction

## Export outputs

Each run creates a timestamped export folder containing files such as:

- `per_frame_detections.csv`
- `per_track_statistics.csv`
- `population_summary.csv`
- `calibration_metadata.json`
- `analysis_parameters.json`
- `annotated_tracking_video.mp4`
- `trajectory_overlay.png`
- `speed_histogram.png`
- `per_track_speed_vs_time.png`
- `msd_plot.png` when enough trajectories are available

## Project structure

```text
micromotor_tracker/
  app.py
  core/
    analysis.py
    calibration.py
    detection.py
    export.py
    tracking.py
    video_io.py
    visualization.py
  models/
    interactive_classifier.py
  utils/
    canvas_compat.py
    config.py
    geometry.py
tests/
example_data/
requirements.txt
run_app.bat
README.md
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run micromotor_tracker/app.py
```

Or on Windows:

```powershell
.\run_app.bat
```

## Run tests

```powershell
python -m unittest discover -s tests
```

## Limitations

- ROI is currently rectangular rather than polygonal
- The classifier is small-sample and feature-based, not deep-learning based
- Dense overlaps and frequent occlusions can still reduce tracking quality
- The review workflow does not yet include advanced annotation management or manual track editing
- TIFF stacks generally still require manual FPS confirmation

## Future improvements

- Polygon ROI support
- Better annotation management and editing tools
- Stronger background modeling
- Batch processing for multiple videos
- More review and QA tools for trajectories
- Optional deep-learning detector support once labeled data is available
