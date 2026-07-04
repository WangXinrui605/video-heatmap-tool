# Video Heatmap Tool for Gaze Data

This is my dissertation project prototype for generating heatmap visualisations from eye-tracking gaze data and scene camera videos.

At the moment I am focusing on the `Face-to-Face` part of the provided demo data. The current version works with the `.txt` gaze data files and the corresponding scene camera videos. The JSON files in the `Augmented Reality` folder are not included in the first version.

## Current Aim

The aim is to build a tool that can:

- read gaze coordinates from a Dikablis `.txt` file;
- match the gaze data to the scene camera video;
- create a red-point overlay video for checking the gaze mapping;
- generate a final static heatmap image from a selected time range;
- later generate a heatmap video that shows how the heatmap develops over time.

## Project Structure

```text
video-heatmap-tool/
|-- src/video_heatmap/        reusable project code
|-- scripts/                  scripts used for testing each step
|-- docs/                     short progress notes
|-- outputs/sample_images/    small example images for GitHub
|-- data/                     local data only, not uploaded to GitHub
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Setup

I am using Python with an Anaconda environment.

Install the required packages:

```bash
pip install -r requirements.txt
```

The main packages used so far are:

- `pandas` for reading the gaze data table;
- `numpy` for numerical processing;
- `opencv-python-headless` for reading videos, writing videos, and drawing overlays.

## Data

The full demo data is not included in this repository because it is large and contains research data.

Expected local folder:

```text
data/Face-to-Face/
|-- P2/
|-- P3/
|-- P4/
`-- P5/
```

Each participant folder contains one gaze `.txt` file and scene camera video files.

## Commands Used So Far

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

## Notes

One important finding so far is that the `.txt` files have many more rows than the videos have frames. For example, P2 has 30,568 rows in the gaze file, while the scene camera video has 6,981 frames. Because of this, I am currently matching gaze samples to video frames using `rec_time` instead of assuming row number equals frame number.

The static heatmap should be interpreted as a heatmap over video image coordinates. Since the video is recorded with a head-mounted scene camera, objects can move in the frame over time.

## Next Steps

- Add the heatmap development video output.
- Compare different time ranges and heatmap settings.
- Add more configurable parameters.
- Later, consider a simple interface aligned with the existing Unity tool.
