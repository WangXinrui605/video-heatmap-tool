from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt
from video_heatmap.marker import MarkerColumns, discover_marker_columns
from video_heatmap.video_io import get_video_info, require_cv2


# D-Lab marker coordinates use corner 1 as the origin. Corners 2 and 4
# define the positive X and Y axes respectively.
MARKER_LOCAL_CORNERS = np.array(
    [
        [0.0, 0.0],  # P1
        [1.0, 0.0],  # P2
        [1.0, 1.0],  # P3
        [0.0, 1.0],  # P4
    ],
    dtype=np.float32,
)

MARKER_COLORS = {
    "Brasilia": (0, 255, 0),
    "Seoul": (0, 255, 255),
    "Zurich": (255, 0, 255),
    "Monaco": (255, 255, 0),
}


def marker_corner_valid_mask(corners: np.ndarray, min_area: float) -> np.ndarray:
    finite = np.isfinite(corners).all(axis=(1, 2))
    nonzero = ~np.all(np.isclose(corners, 0.0), axis=(1, 2))
    x = corners[:, :, 0]
    y = corners[:, :, 1]
    area = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
    )
    return finite & nonzero & (area >= min_area)


def numeric_marker_arrays(dataframe: pd.DataFrame, marker: MarkerColumns):
    corner_columns = [column for pair in marker.corner_cols for column in pair]
    corners = (
        dataframe[corner_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .reshape(-1, 4, 2)
    )
    marker_x = pd.to_numeric(dataframe[marker.gaze_x_col], errors="coerce").to_numpy(
        dtype=float
    )
    marker_y = pd.to_numeric(dataframe[marker.gaze_y_col], errors="coerce").to_numpy(
        dtype=float
    )
    return corners, marker_x, marker_y


def project_marker_gaze(cv2, local_point: np.ndarray, corners: np.ndarray):
    transform = cv2.getPerspectiveTransform(
        MARKER_LOCAL_CORNERS, corners.astype(np.float32)
    )
    projected = cv2.perspectiveTransform(
        local_point.astype(np.float32).reshape(1, 1, 2), transform
    )
    return projected.reshape(2).astype(float)


def nearest_time_index(times: np.ndarray, target: float) -> int:
    idx = int(times.searchsorted(target, side="left"))
    if idx >= len(times):
        idx = len(times) - 1
    if idx > 0 and abs(times[idx - 1] - target) < abs(times[idx] - target):
        idx -= 1
    return idx


def draw_cross(cv2, frame, point, color, size: int = 12, thickness: int = 3):
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def summarize_errors(errors: np.ndarray) -> dict:
    if errors.size == 0:
        return {
            "count": 0,
            "mean_pixels": None,
            "median_pixels": None,
            "p95_pixels": None,
            "max_pixels": None,
        }
    return {
        "count": int(errors.size),
        "mean_pixels": float(np.mean(errors)),
        "median_pixels": float(np.median(errors)),
        "p95_pixels": float(np.quantile(errors, 0.95)),
        "max_pixels": float(np.max(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Project D-Lab marker-relative gaze back to Scene Cam pixels and "
            "compare it with the exported Scene Cam gaze."
        )
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

    start_time = args.start_frame / video_info.fps
    end_time = (end_frame - 1) / video_info.fps
    gaze = load_gaze_txt(
        args.gaze,
        video_width=video_info.width,
        video_height=video_info.height,
    )
    dataframe = gaze.dataframe
    markers = [
        marker
        for marker in discover_marker_columns(dataframe.columns)
        if marker.has_all_corners and marker.has_gaze_coordinates
    ]
    if not markers:
        raise SystemExit("No complete marker gaze/corner columns were found.")

    cv2 = require_cv2()
    screen_x = dataframe["gaze_x"].to_numpy(dtype=float)
    screen_y = dataframe["gaze_y"].to_numpy(dtype=float)
    times = dataframe["time_seconds"].to_numpy(dtype=float)
    screen_valid = dataframe["valid"].to_numpy(dtype=bool)

    records: list[dict] = []
    samples_by_marker = {}
    marker_summaries = {}

    for marker in markers:
        corners, marker_x, marker_y = numeric_marker_arrays(dataframe, marker)
        valid = marker_corner_valid_mask(corners, args.min_marker_area)
        valid &= screen_valid
        valid &= np.isfinite(times)
        valid &= np.isfinite(marker_x) & np.isfinite(marker_y)
        valid &= (times >= start_time) & (times <= end_time)

        marker_records = []
        for row_index in np.flatnonzero(valid):
            local_point = np.array([marker_x[row_index], marker_y[row_index]])
            projected = project_marker_gaze(
                cv2, local_point=local_point, corners=corners[row_index]
            )
            original = np.array([screen_x[row_index], screen_y[row_index]])
            error = float(np.linalg.norm(projected - original))
            record = {
                "marker": marker.name,
                "row_index": int(row_index),
                "time_seconds": float(times[row_index]),
                "screen_x": float(original[0]),
                "screen_y": float(original[1]),
                "marker_x": float(local_point[0]),
                "marker_y": float(local_point[1]),
                "projected_x": float(projected[0]),
                "projected_y": float(projected[1]),
                "error_pixels": error,
            }
            records.append(record)
            marker_records.append(record)

        marker_records.sort(key=lambda item: item["time_seconds"])
        sample_times = np.array(
            [item["time_seconds"] for item in marker_records], dtype=float
        )
        samples_by_marker[marker.name] = {
            "times": sample_times,
            "records": marker_records,
        }
        errors = np.array(
            [item["error_pixels"] for item in marker_records], dtype=float
        )
        marker_summaries[marker.name] = summarize_errors(errors)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "marker_reprojection_samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "marker",
            "row_index",
            "time_seconds",
            "screen_x",
            "screen_y",
            "marker_x",
            "marker_y",
            "projected_x",
            "projected_y",
            "error_pixels",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    video_path = out_dir / "marker_reprojection_overlay.mp4"
    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_info.fps,
        (video_info.width, video_info.height),
    )
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise SystemExit("Could not open the input video or output video writer.")

    frame_index = args.start_frame
    written_frames = 0
    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frame_time = frame_index / video_info.fps

        label_y = 78
        for marker in markers:
            samples = samples_by_marker[marker.name]
            if samples["times"].size == 0:
                continue
            sample_index = nearest_time_index(samples["times"], frame_time)
            record = samples["records"][sample_index]
            if abs(record["time_seconds"] - frame_time) > args.max_time_delta:
                continue

            original = (record["screen_x"], record["screen_y"])
            projected = (record["projected_x"], record["projected_y"])
            color = MARKER_COLORS.get(marker.name, (255, 255, 255))
            cv2.circle(
                frame,
                (int(round(original[0])), int(round(original[1]))),
                9,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )
            draw_cross(cv2, frame, projected, color)
            cv2.line(
                frame,
                (int(round(original[0])), int(round(original[1]))),
                (int(round(projected[0])), int(round(projected[1]))),
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{marker.name}: {record['error_pixels']:.2f} px",
                (24, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )
            label_y += 34

        cv2.putText(
            frame,
            "red dot = exported screen gaze",
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        written_frames += 1
        frame_index += 1

    cap.release()
    writer.release()

    all_errors = np.array([item["error_pixels"] for item in records], dtype=float)
    summary = {
        "frame_range": {
            "start_frame_inclusive": args.start_frame,
            "end_frame_exclusive": end_frame,
        },
        "coordinate_convention": {
            "corner_1": [0.0, 0.0],
            "corner_2": [1.0, 0.0],
            "corner_3": [1.0, 1.0],
            "corner_4": [0.0, 1.0],
        },
        "overall_error": summarize_errors(all_errors),
        "markers": marker_summaries,
        "output_video": str(video_path),
        "output_samples_csv": str(csv_path),
        "written_frames": written_frames,
    }
    summary_path = out_dir / "marker_reprojection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
