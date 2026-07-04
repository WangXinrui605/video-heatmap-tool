# Week 1-2 Progress Summary

## Current scope

The project currently focuses on the `Face-to-Face` part of `Demo Data`.

The `Augmented Reality` JSON files are ignored for now, following the supervisor's instruction.

## Week 1: Dataset inspection

The dataset has been placed in:

```text
Demo Data/Sui Li Demo Data/Face-to-Face
```

The folder contains four participants:

| Participant | TXT file | Scene video | Eye-tracking reference video |
| --- | --- | --- | --- |
| P2 | yes | yes | yes |
| P3 | yes | yes | yes |
| P4 | yes | yes | yes |
| P5 | yes | yes | yes |

All participants have the same main gaze coordinate columns:

```text
Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze X
Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze Y
```

The scene camera videos are 1920x1080.

## Gaze data inspection results

After filtering invalid gaze points and coordinates outside the 1920x1080 video frame:

| Participant | Total TXT rows | Valid gaze rows | Invalid rows | Valid X range | Valid Y range |
| --- | ---: | ---: | ---: | --- | --- |
| P2 | 30,568 | 6,149 | 24,419 | 601.207-1121.189 | 61.049-1079.990 |
| P3 | 27,585 | 11,819 | 15,766 | 6.187-1919.588 | 59.718-1075.364 |
| P4 | 22,455 | 13,144 | 9,311 | 158.101-1897.887 | 37.978-1079.399 |
| P5 | 15,898 | 10,727 | 5,171 | 17.620-1880.760 | 15.064-1078.237 |

Important finding:

The `.txt` files contain more rows than the video has frames. For example, P2 has 30,568 TXT rows but the scene video has 6,981 frames. Therefore, one TXT row should not be treated as one video frame. The current prototype uses `rec_time` to match each video frame to the nearest valid gaze sample.

## Week 2: Gaze overlay prototype

A first gaze overlay prototype has been implemented.

For P2:

- scene video resolution: 1920x1080
- scene video FPS: approximately 30.00 FPS
- scene video frame count: 6,981
- tested range: first 300 frames
- frames written: 300
- frames with matched valid gaze point: 172
- matching method: nearest valid gaze sample by `rec_time`

Output files:

```text
outputs/P2_overlay/gaze_overlay_sample.mp4
outputs/P2_overlay/gaze_overlay_summary.json
outputs/P2_overlay/first_frame.png
outputs/P2_overlay/first_matched_frame_1.png
```

The first matched frame shows the red gaze point drawn on the scene video at:

```text
frame: 1
frame time: 0.033 seconds
gaze time: 0.054 seconds
x: 927.144
y: 636.524
```

## Implementation notes

The first version of the project scaffold is in:

```text
work/video_heatmap_tool
```

Implemented so far:

- dataset inspection script;
- participant inspection script;
- gaze TXT loader;
- valid gaze filtering;
- MP4 video metadata reading;
- OpenCV-based video reading/writing;
- gaze point overlay prototype.

## Next steps

The next practical tasks are:

1. Compare the generated gaze overlay video with the original eye-tracking reference video.
2. If the gaze position is correct, implement the final heatmap image output.
3. Implement the heatmap development video output.
4. Add configuration options such as start/end frame, brush size, opacity, normalization, frame skip, and heatmap gradient.
5. Later, convert the configuration options into UI controls aligned with the existing Unity tool, such as sliders, checkboxes, dropdowns, file pickers, and buttons.
