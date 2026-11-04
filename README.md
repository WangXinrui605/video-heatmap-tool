# Video Heatmap Tool for Gaze Data

A Python tool for visualising eye-tracking gaze data over scene-camera video.
It provides a command-line workflow and a Windows desktop interface for producing
screen-space, face-relative, marker-relative, and optical-flow heatmaps.

## What is included

- Reusable Python package in `src/video_heatmap/`.
- Command-line scripts for generating overlays, static heatmaps, and videos.
- A PySide6 desktop application for the baseline, face-relative, and optical-flow
  workflows.
- Automated tests for the processing pipeline and desktop-app components.
- One small, non-sensitive example image in `outputs/sample_images/`.

Research data, model weights, generated videos, personal working notes, and
dissertation materials are deliberately excluded from this repository.

## Methods

| Method | Purpose | Main output |
| --- | --- | --- |
| Baseline | Cumulative gaze heatmap in the original video coordinate system. | Static image and development video |
| Face-relative | Maps gaze inside a selected face bounding box to a canonical face canvas. | Static/dynamic overlays and JSON summary |
| Optical flow | Propagates a recent window of gaze points using dense optical flow. | Windowed overlay video and JSON summary |
| Marker-relative | Maps gaze to a planar marker coordinate system. This is available through the CLI. | Marker heatmap video and JSON summary |

The face-relative workflow uses either OpenCV YuNet or OpenCV DNN SSD. Detection
loss and multiple-face frames are recorded explicitly; they are not treated as
successful face-relative observations. Optical flow is an experimental visual
comparison method, not object recognition or long-term identity tracking.

## Requirements

- Python 3.10 or later
- A local gaze-data text file and the matching scene-camera video

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data layout

The source recordings are not included. Keep them locally, for example:

```text
data/Face-to-Face/P2/
├── participant_gaze.txt
└── scene_camera.mp4
```

## Quick start

Generate a baseline static heatmap:

```bash
python scripts/generate_final_heatmap.py \
  --gaze "data/Face-to-Face/P2/participant_gaze.txt" \
  --video "data/Face-to-Face/P2/scene_camera.mp4" \
  --out-dir "outputs/baseline" \
  --start-frame 0 --end-frame 300
```

Launch the desktop application on Windows:

```bash
python scripts/launch_heatmap_tool.py
```

To use the face-relative method, download a compatible face-detector model
locally and supply its path in the application or script. Model files are not
bundled here because their distribution terms must be checked separately.

## Testing

```bash
pytest -q
```

## Repository layout

```text
src/video_heatmap/       Core processing and desktop-application code
scripts/                 Command-line entry points and evaluation utilities
tests/                   Automated tests and lightweight fixtures
outputs/sample_images/   Example image safe to publish
data/                    Local research data (ignored)
```

## Scope and limitations

This repository is a software demonstration, not a dataset release. Results
depend on the local video, gaze calibration, selected time range, detector model,
and processing parameters. The tool does not establish gaze accuracy, detector
accuracy, identity recognition, clinical validity, or general usability.
