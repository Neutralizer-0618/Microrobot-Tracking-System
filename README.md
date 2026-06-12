# Microrobot Tracking System

Microrobot Tracking System is a fully local Python + Streamlit application for microscopy-video analysis of micromotors and biohybrid microrobots. The current MVP focuses on a practical, interpretable workflow: load a microscopy video, calibrate pixel scale, optionally define an ROI, annotate positive and negative examples, detect candidate objects with classical computer vision, track them over time, compute motility statistics, and export analysis-ready outputs.

## Highlights

- Local-first workflow. No cloud upload is required for microscopy data.
- Supports `mp4`, `avi`, `mov`, `tif`, and `tiff` inputs.
- Manual or file-derived FPS handling.
- Pixel-to-micron calibration from a user-drawn reference line.
- ROI restriction for whole-video analysis.
- Positive/negative sample annotation on microscopy frames.
- Classical CV + lightweight ML pipeline for reliable first-pass analysis.
- Multi-object tracking with trajectory statistics and population summaries.
- Export of CSV, JSON, annotated video, and publication-ready plots.

## MVP capabilities

- Video loading and frame navigation.
- Calibration with real-world length input.
- Optional rectangular ROI.
- Positive and negative sample collection across frames.
- Candidate detection using thresholding, contour filtering, and background subtraction.
- Lightweight interactive classification using handcrafted features plus a `RandomForestClassifier`.
- Multi-object linking using centroid prediction with Hungarian matching and short-gap interpolation.
- Per-track motility analysis:
  - instantaneous speed
  - mean / median / max speed
  - total path length
  - net displacement
  - track duration
  - straightness / directionality
- Population analysis:
  - valid track count
  - mean / median / standard deviation of speed
  - moving fraction
  - inactive fraction
- Export bundle generation for reproducible downstream analysis.

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

1. Install Python 3.10 or newer on Windows.
2. Open a terminal in the project root.
3. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to run

```powershell
streamlit run micromotor_tracker/app.py
```

Or on Windows:

```powershell
.\run_app.bat
```

The app runs fully locally. No video data is uploaded to any cloud service by this project.

## Expected input videos

- Brightfield or fluorescence microscopy videos where micromotors are visible as compact moving objects.
- Formats: `mp4`, `avi`, `mov`, `tif`, `tiff`.
- TIFF stacks are supported as frame sequences, but FPS usually needs manual entry.
- Inputs with stable illumination and reasonable contrast will work best in this MVP.

## Calibration workflow

1. Load a video or TIFF stack.
2. Navigate to a frame with a visible scale bar or known reference feature.
3. Draw a line over the known reference.
4. Enter its real-world length in micrometers.
5. Save calibration to compute `micron_per_pixel`.

Calibration metadata is included in export outputs.

## Analysis workflow

1. Load the input video.
2. Confirm or enter FPS.
3. Draw a calibration line if real-world units are needed.
4. Optionally draw an ROI.
5. Use frame navigation to move to informative frames and draw positive examples around representative micromotors.
6. Optionally annotate negative examples around debris or artifacts on one or more frames.
7. Estimate the object model from all annotated frames.
8. Tune detection parameters such as threshold method, object area, circularity, and confidence threshold.
9. Run detection.
10. Tune linking distance, frame gap, and minimum track length.
11. Run tracking and motility analysis.
12. Review overlays and tables.
13. Export results.

## Machine-learning approach

The first version intentionally avoids heavy deep learning in favor of a more interpretable local workflow:

- Candidate generation uses classical computer vision:
  - thresholding
  - contour-based segmentation
  - background subtraction
- Sample-guided classification uses handcrafted morphology, intensity, and texture features.
- The current lightweight classifier is a `RandomForestClassifier`, chosen because it works well with small labeled datasets, trains quickly, runs locally, and is easy to reason about during scientific QA.

## Output files

Each export creates a timestamped output folder containing:

- `per_frame_detections.csv`: accepted detections for each frame with centroid, bbox, area, and confidence.
- `per_track_statistics.csv`: one row per trajectory with speed, displacement, duration, and quality fields.
- `population_summary.csv`: aggregate summary statistics for the analyzed population.
- `calibration_metadata.json`: calibration points, line length, and `micron_per_pixel`.
- `analysis_parameters.json`: saved detection, tracking, analysis, ROI, and sampling parameters.
- `annotated_tracking_video.mp4`: overlay video with IDs and trajectory trails.
- `trajectory_overlay.png`: single-image trajectory summary.
- `speed_histogram.png`: distribution of per-track mean speeds.
- `per_track_speed_vs_time.png`: speed time-series plot for tracked objects.
- `msd_plot.png`: exported when enough trajectory data is available.

## Testing

```powershell
python -m unittest discover -s tests
```

## Limitations

- The MVP uses rectangular ROI selection rather than arbitrary polygons.
- Example-guided learning is based on lightweight handcrafted features and a random forest, not deep learning.
- Detection is strongest when objects are reasonably separated from the background.
- Manual sample annotation supports accumulating positive and negative boxes across multiple frames, but it does not yet include dedicated annotation-management tools such as per-frame delete/history views.
- The tracker uses a simple constant-velocity centroid model, so dense overlaps and frequent occlusions can reduce quality.
- TIFF stacks rely on manual FPS input unless metadata is available elsewhere.

## Future features

- Polygon ROI support.
- Better background modeling and adaptive motion segmentation.
- Dedicated review tools for track editing, pruning, and relabeling.
- Batch processing across multiple videos.
- Additional plots such as turning-angle statistics and directional rose plots.
- Optional deep-learning detector integration once a robust labeled dataset is available.
