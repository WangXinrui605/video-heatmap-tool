from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt
from video_heatmap.marker import MarkerColumns, discover_marker_columns
from video_heatmap.video_io import get_video_info, require_cv2


MARKER_COLORS = {
    "Brasilia": (0, 255, 0),
    "Seoul": (0, 255, 255),
    "Zurich": (255, 0, 255),
    "Monaco": (255, 255, 0),
}


def nearest_time_index(times: np.ndarray, target: float) -> int:
    idx = int(times.searchsorted(target, side="left"))
    if idx >= len(times):
        idx = len(times) - 1
    if idx > 0 and abs(times[idx - 1] - target) < abs(times[idx] - target):
        idx -= 1
    return idx


def prepare_marker_corners(
    dataframe: pd.DataFrame, marker: MarkerColumns
) -> np.ndarray:
    columns = [column for pair in marker.corner_cols for column in pair]
    numeric = dataframe[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.to_numpy(dtype=float).reshape(-1, 4, 2)


def marker_corner_valid_mask(corners: np.ndarray, min_area: float) -> np.ndarray:
    finite = np.isfinite(corners).all(axis=(1, 2))
    nonzero = ~np.all(np.isclose(corners, 0.0), axis=(1, 2))
    x = corners[:, :, 0]
    y = corners[:, :, 1]
    area = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
    )
    return finite & nonzero & (area >= min_area)


def draw_marker(cv2, frame, name: str, corners: np.ndarray, color) -> None:
    points = np.rint(corners).astype(np.int32)
    cv2.polylines(frame, [points], True, color, 3, cv2.LINE_AA)

    for corner_number, point in enumerate(points, start=1):
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(corner_number),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    label_x = int(points[:, 0].min())
    label_y = max(28, int(points[:, 1].min()) - 14)
    cv2.putText(
        frame,
        name,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw detected D-Lab marker names and corners on a Scene Cam video."
    )
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=300)
    parser.add_argument("--max-time-delta", type=float, default=0.05)
    parser.add_argument("--min-marker-area", type=float, default=100.0)
    args = parser.parse_args()

    video_info = get_video_info(args.video)
    end_frame = min(args.end_frame, video_info.frame_count)
    if args.start_frame < 0 or args.start_frame >= end_frame:
        raise SystemExit("Invalid frame range.")
    if args.max_time_delta < 0:
        raise SystemExit("--max-time-delta must be non-negative.")

    gaze = load_gaze_txt(
        args.gaze,
        video_width=video_info.width,
        video_height=video_info.height,
    )
    timed = gaze.dataframe.dropna(subset=["time_seconds"]).copy()
    timed = timed.sort_values("time_seconds").reset_index(drop=True)
    if timed.empty:
        raise SystemExit("No gaze rows with valid rec_time were found.")

    markers = [
        marker
        for marker in discover_marker_columns(timed.columns)
        if marker.has_all_corners
    ]
    if not markers:
        raise SystemExit("No complete D-Lab marker corner columns were found.")

    times = timed["time_seconds"].to_numpy(dtype=float)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "marker_overlay.mp4"

    cv2 = require_cv2()
    marker_samples = {}
    for marker in markers:
        corners = prepare_marker_corners(timed, marker)
        valid = marker_corner_valid_mask(corners, min_area=args.min_marker_area)
        marker_samples[marker.name] = {
            "times": times[valid],
            "corners": corners[valid],
        }

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise SystemExit("Could not open the input video or output video writer.")

    detection_counts = {marker.name: 0 for marker in markers}
    frames_with_any_marker = 0
    written_frames = 0
    frame_index = args.start_frame

    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame_time = frame_index / video_info.fps if video_info.fps else 0.0
        detected_this_frame = False
        for marker in markers:
            sample_times = marker_samples[marker.name]["times"]
            if sample_times.size == 0:
                continue
            sample_index = nearest_time_index(sample_times, frame_time)
            time_delta = abs(sample_times[sample_index] - frame_time)
            if time_delta > args.max_time_delta:
                continue
            corners = marker_samples[marker.name]["corners"][sample_index]
            color = MARKER_COLORS.get(marker.name, (255, 255, 255))
            draw_marker(cv2, frame, marker.name, corners, color)
            detection_counts[marker.name] += 1
            detected_this_frame = True
        if detected_this_frame:
            frames_with_any_marker += 1

        cv2.putText(
            frame,
            f"frame {frame_index}",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        written_frames += 1
        frame_index += 1

    cap.release()
    writer.release()

    summary = {
        "output_video": str(output_path),
        "frame_range": {
            "start_frame_inclusive": args.start_frame,
            "end_frame_exclusive": end_frame,
        },
        "written_frames": written_frames,
        "frames_with_any_marker": frames_with_any_marker,
        "matching_method": "nearest valid sample per marker by rec_time",
        "max_time_delta_seconds": args.max_time_delta,
        "min_marker_area_pixels": args.min_marker_area,
        "markers": [
            {
                "name": marker.name,
                "has_marker_relative_gaze": marker.has_gaze_coordinates,
                "valid_input_samples": int(
                    marker_samples[marker.name]["times"].size
                ),
                "detected_output_frames": detection_counts[marker.name],
                "color_bgr": list(
                    MARKER_COLORS.get(marker.name, (255, 255, 255))
                ),
            }
            for marker in markers
        ],
        "video_info": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "frame_count": video_info.frame_count,
        },
    }
    summary_path = out_dir / "marker_overlay_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
