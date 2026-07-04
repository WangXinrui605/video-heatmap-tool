# Video Heatmap Tool for Gaze Data

This project develops a video heatmap tool for TXT-based eye-tracking recordings in the provided Face-to-Face dataset.

The current implementation focuses on:

- reading Dikablis `.txt` gaze data;
- extracting gaze coordinates for the scene camera video;
- validating gaze coordinates by drawing gaze points over video frames;
- preparing the pipeline for final heatmap image and heatmap development video outputs.

The `Augmented Reality` JSON files are out of scope for now, following the supervisor's instruction.

## Project Structure

```text
video-heatmap-tool/
├── src/video_heatmap/        # reusable project code
├── scripts/                  # command-line scripts for running tasks
├── docs/                     # project notes and progress summaries
├── outputs/sample_images/    # small result images suitable for GitHub
├── data/                     # local data folder, ignored by Git
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

The full Demo Data folder is not included in the repository because it is large and may contain research data.

Place the Face-to-Face data locally under:

```text
data/Face-to-Face/
```

Expected structure:

```text
data/Face-to-Face/
├── P2/
├── P3/
├── P4/
└── P5/
```

Each participant folder should contain:

- one `*_CsvData.txt` gaze data file;
- one `*_Scene Cam.mp4` video;
- one `*_Scene Cam_Original_Eye Tracking Video.mp4` reference video.

## Week 1: Inspect Dataset

```bash
python scripts/inspect_dataset.py --data-root "data/Face-to-Face"
```

Inspect one participant:

```bash
python scripts/inspect_participant.py --participant-dir "data/Face-to-Face/P2" --out-dir "outputs/P2_inspection"
```

## Week 2: Generate Gaze Overlay

```bash
python scripts/generate_gaze_overlay.py ^
  --gaze "data/Face-to-Face/P2/P2_2. Recording 13.09.2024 115435_CsvData.txt" ^
  --video "data/Face-to-Face/P2/P2_2. Recording 13.09.2024 115435_Dikablis Glasses 3_Scene Cam.mp4" ^
  --out-dir "outputs/P2_overlay" ^
  --start-frame 0 ^
  --end-frame 300
```

This creates a short video with red gaze points drawn on the scene camera video.

## Current Key Finding

The TXT data has more rows than the video has frames. Therefore, the tool should not map row `N` directly to frame `N`.

The current prototype uses `rec_time` to match each video frame to the nearest valid gaze sample.

## Next Steps

1. Compare the generated gaze overlay with the original eye-tracking reference video.
2. Implement a final heatmap image containing all selected gaze data.
3. Implement a video showing heatmap development over time.
4. Add configuration options similar to the existing Unity tool.
5. Later, expose configuration options through UI controls such as sliders, checkboxes, dropdowns, file pickers, and buttons.
