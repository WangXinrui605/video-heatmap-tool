# Week 1-2 Progress Notes

## Project Scope

The project is currently focused on the `Face-to-Face` demo data. I am using the `.txt` gaze data files and the matching scene camera videos. The JSON files from the `Augmented Reality` folder are being left out for now, based on the supervisor's feedback.

Python is being used for the prototype because it is practical for reading table data, processing video frames, and generating image overlays.

## What I Have Done

- Set up a local project folder and Git repository.
- Created a basic Python project structure with `src`, `scripts`, `docs`, `outputs`, and `data` folders.
- Set up a Conda environment and installed the required packages.
- Copied the Face-to-Face demo data into the local `data` folder.
- Wrote a dataset inspection script to check which participants, `.txt` files, and `.mp4` files are available.
- Wrote a participant inspection script to check gaze data quality and video metadata.
- Generated a short red-point overlay video for P2 to check whether gaze coordinates can be placed back onto the scene camera video.
- Extracted still frames from the generated overlay video to make the visual checking easier.
- Generated static heatmap images for P2 using the first 300 frames and the first 1000 frames.

## Current Understanding

The `.txt` files are not normal text notes. They are table-like eye-tracking data files. The gaze coordinates I am currently using are:

```text
Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze X
Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze Y
```

These columns appear to describe gaze position in the scene camera video frame.

The `.txt` rows should not be treated as video frames directly. For example, P2 has 30,568 rows in the `.txt` file, but the scene camera video has 6,981 frames. The current prototype therefore matches video frames to gaze samples using `rec_time`.

## Results So Far

For P2:

- Scene camera video resolution: 1920 x 1080
- Scene camera FPS: about 30 FPS
- Scene camera frame count: 6,981
- Gaze file rows: 30,568
- Valid gaze rows after filtering: 6,149

The red-point overlay appears to place gaze points in plausible positions. I also compared the generated overlay with frames extracted from the same scene camera video to confirm that the background frames match.

I also tried comparing with the exported original eye-tracking video, but the videos do not seem to stay frame-synchronised when using the same frame numbers. This may be because the eye-tracking reference video was exported with a different timing or start point. I plan to discuss this with my supervisor.

The static heatmap for the first 1000 frames shows gaze concentration around the participant's upper body / marker area, which matches my rough visual inspection of the red-point overlay video.

## Questions For The Next Meeting

1. Should the final static heatmap usually be generated over the whole video, or should the user choose a time range?
2. Should the heatmap development video be cumulative over time, or should it show only a recent time window?
3. Is it acceptable to interpret the static heatmap as image-coordinate based, given that the scene camera is head-mounted and moves over time?
4. Should I focus first on reproducing the existing Unity tool's main features before adding extra options?
5. What kind of evaluation would be suitable for this project: technical validation, expert walkthrough, small user test, or a combination?

## Next Steps

- Implement the heatmap development video output.
- Test the same pipeline with P3, P4, and P5.
- Add clearer configuration options such as start frame, end frame, brush size, opacity, normalisation, and frame skip.
- Start documenting the implementation and decisions in a way that can support the final dissertation report.
