# Video Heatmap Tool for Gaze Data

This repository contains an early Python prototype for my dissertation project. The project aims to generate heatmap visualisations from eye-tracking gaze data and scene camera videos.

The current prototype focuses on the `Face-to-Face` part of the provided demo data. It uses the Dikablis `.txt` gaze data files and the matching scene camera videos. The JSON files in the `Augmented Reality` folder are outside the first version of the project.

## Current Status

Implemented so far:

- dataset inspection for the Face-to-Face demo data;
- participant-level gaze and video inspection;
- loading and filtering gaze coordinates from `.txt` files;
- matching gaze samples to scene camera video time using `rec_time`;
- red-point gaze overlay video for checking the mapping;
- frame extraction for manual checking;
- static heatmap image generation for a selected frame range.

Still to do:

- heatmap development video output;
- more testing with P3, P4, and P5;
- clearer configuration options;
- later, a simple interface aligned with the existing Unity tool.

## Project Structure

```text
video-heatmap-tool/
|-- src/video_heatmap/        reusable project code
|-- scripts/                  command-line scripts for running each step
|-- outputs/sample_images/    small example images for GitHub
|-- data/                     local data only, ignored by Git
|-- docs/                     local notes only, ignored by Git
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Setup

Create or activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The main packages used are:

- `pandas` for reading the gaze data table;
- `numpy` for numerical processing;
- `opencv-python-headless` for video reading, video writing, and image overlays.

## Data

The demo data is not included in this repository because it is large and may contain research data.

Expected local folder structure:

```text
data/Face-to-Face/
|-- P2/
|-- P3/
|-- P4/
`-- P5/
```

Each participant folder should contain a gaze `.txt` file and the matching scene camera video files.

## Main Scripts

Inspect the dataset:

```bash
python scripts/inspect_dataset.py --data-root "data/Face-to-Face"
```

Inspect one participant:

```bash
python scripts/inspect_participant.py --participant-dir "data/Face-to-Face/P2" --out-dir "outputs/P2_inspection"
```

Generate a short red-point overlay video:

```bash
python scripts/generate_gaze_overlay.py --gaze "data/Face-to-Face/P2/P2_2. Recording 13.09.2024 115435_CsvData.txt" --video "data/Face-to-Face/P2/P2_2. Recording 13.09.2024 115435_Dikablis Glasses 3_Scene Cam.mp4" --out-dir "outputs/P2_overlay" --start-frame 0 --end-frame 300
```

Extract still frames from a video:

```bash
python scripts/extract_video_frames.py --video "outputs/P2_overlay/gaze_overlay_sample.mp4" --out-dir "outputs/P2_overlay_frames" --frames 1 50 100 150 200 250
```

Generate a static heatmap image:

```bash
python scripts/generate_final_heatmap.py --gaze "data/Face-to-Face/P2/P2_2. Recording 13.09.2024 115435_CsvData.txt" --video "data/Face-to-Face/P2/P2_2. Recording 13.09.2024 115435_Dikablis Glasses 3_Scene Cam.mp4" --out-dir "outputs/P2_final_heatmap_1000" --start-frame 0 --end-frame 1000 --brush-size 40 --opacity 0.55
```

## Current Notes

The `.txt` gaze data files contain more rows than the scene camera videos have frames. For example, P2 has 30,568 gaze rows, while the scene camera video has 6,981 frames. The prototype therefore does not assume that row number equals frame number. It currently uses `rec_time` to select gaze samples for a video time range.

The static heatmap is currently an image-coordinate heatmap. Since the scene camera is head-mounted, objects can move within the frame over time. The static heatmap is useful for showing the overall gaze distribution in the video image, but it should be interpreted together with the planned heatmap development video.
