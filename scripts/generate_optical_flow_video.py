from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt
from video_heatmap.heatmap import (
    build_final_heatmap,
    normalize_with_max,
    overlay_heatmap,
)
from video_heatmap.optical_flow import (
    FarnebackConfig,
    calculate_flow_pair,
    gaze_frame_indices,
    is_shot_boundary,
    load_manual_boundaries,
    propagate_points,
    resolve_frame_range,
    summarize_numbers,
)
from video_heatmap.video_io import get_video_info, require_cv2


TRAJECTORY_FIELDS = [
    "frame",
    "time_seconds",
    "gaze_id",
    "sample_time",
    "x",
    "y",
    "active",
    "termination_reason",
    "forward_backward_error",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Farneback motion-compensated dynamic gaze heatmap "
            "with a sliding window or shot-local cumulative history."
        )
    )
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--history-mode", choices=["window", "cumulative"], default="window")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--flow-scale", type=float, default=0.125)
    parser.add_argument(
        "--fb-threshold",
        type=float,
        default=12.0,
        help="Full-resolution pixel threshold; use 0 to disable the project extension.",
    )
    parser.add_argument("--shot-threshold", type=float, default=0.03)
    parser.add_argument("--shot-hist-threshold", type=float, default=0.15)
    parser.add_argument("--manual-boundaries")
    parser.add_argument("--brush-size", type=int, default=40)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--heatmap-max", type=float, default=20.0)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()

    if args.history_mode == "window" and args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive.")
    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be at least 1.")
    if args.brush_size < 1 or args.heatmap_max <= 0:
        raise SystemExit("--brush-size and --heatmap-max must be positive.")
    if not 0.0 <= args.opacity <= 1.0:
        raise SystemExit("--opacity must be between 0 and 1.")

    video_info = get_video_info(args.video)
    try:
        frame_range = resolve_frame_range(
            video_info,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    gaze = load_gaze_txt(
        args.gaze, video_width=video_info.width, video_height=video_info.height
    )
    selected = gaze.dataframe[
        gaze.dataframe["valid"] & gaze.dataframe["time_seconds"].notna()
    ].copy()
    selected = selected[
        (selected["time_seconds"] >= frame_range.start_time)
        & (selected["time_seconds"] <= frame_range.end_time)
    ].sort_values("time_seconds")
    if selected.empty:
        raise SystemExit("No valid gaze samples in the selected range.")
    selected["assigned_frame"] = gaze_frame_indices(
        selected["time_seconds"].to_numpy(dtype=float), video_info.fps
    )
    selected = selected[
        (selected["assigned_frame"] >= frame_range.start_frame)
        & (selected["assigned_frame"] < frame_range.end_frame)
    ]
    rows_by_frame = {
        int(frame): group
        for frame, group in selected.groupby("assigned_frame", sort=False)
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "optical_flow_heatmap.mp4"
    summary_path = out_dir / "optical_flow_video_summary.json"
    boundary_csv_path = out_dir / "shot_boundaries.csv"
    trajectory_path = out_dir / "optical_flow_gaze_trajectories.csv"
    config = FarnebackConfig(scale=args.flow_scale)
    fb_threshold = None if args.fb_threshold <= 0 else args.fb_threshold
    manual = load_manual_boundaries(args.manual_boundaries)
    cv2 = require_cv2()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range.start_frame)
    output_fps = video_info.fps / args.frame_skip
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (video_info.width, video_info.height),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create output video: {output_path}")

    trajectory_stream = None
    trajectory_writer = None
    if args.save_trajectories:
        trajectory_stream = trajectory_path.open("w", newline="", encoding="utf-8")
        trajectory_writer = csv.DictWriter(
            trajectory_stream, fieldnames=TRAJECTORY_FIELDS
        )
        trajectory_writer.writeheader()

    active_points = np.empty((0, 2), dtype=np.float32)
    active_ids = np.empty(0, dtype=np.int64)
    active_sample_times = np.empty(0, dtype=np.float64)
    active_errors = np.empty(0, dtype=np.float32)
    next_gaze_id = 0
    boundaries: list[dict] = []
    fb_errors: list[float] = []
    termination_counts = {
        "flow_or_bounds": 0,
        "shot_boundary": 0,
        "window_expired": 0,
    }
    written_frames = 0
    processed_frames = 0
    maximum_active = 0
    total_added = 0

    def write_trajectory_rows(
        frame_index: int,
        points: np.ndarray,
        ids: np.ndarray,
        sample_times: np.ndarray,
        errors: np.ndarray,
        active: bool,
        reason: str,
    ) -> None:
        if trajectory_writer is None:
            return
        frame_time = frame_index / video_info.fps
        for point, gaze_id, sample_time, error in zip(
            points, ids, sample_times, errors
        ):
            trajectory_writer.writerow(
                {
                    "frame": frame_index,
                    "time_seconds": frame_time,
                    "gaze_id": int(gaze_id),
                    "sample_time": float(sample_time),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "active": active,
                    "termination_reason": reason,
                    "forward_backward_error": (
                        "" if not np.isfinite(error) else float(error)
                    ),
                }
            )

    def add_new_gaze(frame_index: int) -> None:
        nonlocal active_points, active_ids, active_sample_times
        nonlocal active_errors, next_gaze_id, total_added
        group = rows_by_frame.get(frame_index)
        if group is None:
            return
        points = group[["gaze_x", "gaze_y"]].to_numpy(dtype=np.float32)
        times = group["time_seconds"].to_numpy(dtype=np.float64)
        ids = np.arange(next_gaze_id, next_gaze_id + len(points), dtype=np.int64)
        next_gaze_id += len(points)
        active_points = np.vstack([active_points, points])
        active_ids = np.concatenate([active_ids, ids])
        active_sample_times = np.concatenate([active_sample_times, times])
        active_errors = np.concatenate(
            [active_errors, np.full(len(points), np.nan, dtype=np.float32)]
        )
        total_added += len(points)

    ok, previous = cap.read()
    if not ok:
        writer.release()
        cap.release()
        if trajectory_stream:
            trajectory_stream.close()
        raise SystemExit("Could not read first selected frame.")
    add_new_gaze(frame_range.start_frame)

    for frame_index in range(frame_range.start_frame, frame_range.end_frame):
        if frame_index == frame_range.start_frame:
            current = previous
            boundary = False
        else:
            ok, current = cap.read()
            if not ok:
                break
            pair = calculate_flow_pair(previous, current, config)
            is_manual = frame_index in manual
            boundary = is_manual or is_shot_boundary(
                pair, args.shot_threshold, args.shot_hist_threshold
            )
            if boundary:
                if len(active_points):
                    write_trajectory_rows(
                        frame_index,
                        active_points,
                        active_ids,
                        active_sample_times,
                        active_errors,
                        False,
                        "shot_boundary",
                    )
                termination_counts["shot_boundary"] += int(len(active_points))
                active_points = np.empty((0, 2), dtype=np.float32)
                active_ids = np.empty(0, dtype=np.int64)
                active_sample_times = np.empty(0, dtype=np.float64)
                active_errors = np.empty(0, dtype=np.float32)
                boundaries.append(
                    {
                        "frame": frame_index,
                        "time_seconds": frame_index / video_info.fps,
                        "disturbance_score": pair.disturbance_score,
                        "appearance_change_score": pair.appearance_change_score,
                        "threshold": args.shot_threshold,
                        "appearance_threshold": args.shot_hist_threshold,
                        "source": "manual" if is_manual else "farneback_auto",
                    }
                )
            elif len(active_points):
                result = propagate_points(
                    active_points,
                    pair,
                    video_info.width,
                    video_info.height,
                    fb_threshold,
                )
                invalid = ~result.valid
                if np.any(invalid):
                    write_trajectory_rows(
                        frame_index,
                        result.points[invalid],
                        active_ids[invalid],
                        active_sample_times[invalid],
                        result.forward_backward_error[invalid],
                        False,
                        "flow_or_bounds",
                    )
                termination_counts["flow_or_bounds"] += int(np.count_nonzero(invalid))
                fb_errors.extend(
                    result.forward_backward_error[result.valid].tolist()
                )
                active_points = result.points[result.valid]
                active_ids = active_ids[result.valid]
                active_sample_times = active_sample_times[result.valid]
                active_errors = result.forward_backward_error[result.valid]

            add_new_gaze(frame_index)

        frame_time = frame_index / video_info.fps
        if args.history_mode == "window" and len(active_points):
            expired = active_sample_times < frame_time - args.window_seconds
            if np.any(expired):
                write_trajectory_rows(
                    frame_index,
                    active_points[expired],
                    active_ids[expired],
                    active_sample_times[expired],
                    active_errors[expired],
                    False,
                    "window_expired",
                )
            termination_counts["window_expired"] += int(np.count_nonzero(expired))
            keep = ~expired
            active_points = active_points[keep]
            active_ids = active_ids[keep]
            active_sample_times = active_sample_times[keep]
            active_errors = active_errors[keep]

        maximum_active = max(maximum_active, len(active_points))
        should_write = (
            frame_index - frame_range.start_frame
        ) % args.frame_skip == 0
        if should_write:
            heatmap = build_final_heatmap(
                active_points,
                video_info.width,
                video_info.height,
                args.brush_size,
            )
            heatmap_uint8 = normalize_with_max(heatmap, args.heatmap_max)
            overlay, _ = overlay_heatmap(current, heatmap_uint8, args.opacity)
            if args.debug_overlay:
                cv2.putText(
                    overlay,
                    (
                        f"frame={frame_index} active={len(active_points)} "
                        f"cuts={len(boundaries)}"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                for point in active_points[:: max(1, len(active_points) // 100)]:
                    cv2.circle(
                        overlay,
                        (int(round(point[0])), int(round(point[1]))),
                        2,
                        (255, 255, 0),
                        -1,
                        cv2.LINE_AA,
                    )
            writer.write(overlay)
            written_frames += 1

        write_trajectory_rows(
            frame_index,
            active_points,
            active_ids,
            active_sample_times,
            active_errors,
            True,
            "",
        )
        processed_frames += 1
        previous = current

    cap.release()
    writer.release()
    if trajectory_stream:
        trajectory_stream.close()

    with boundary_csv_path.open("w", newline="", encoding="utf-8") as stream:
        boundary_writer = csv.DictWriter(
            stream,
            fieldnames=[
                "frame",
                "time_seconds",
                "disturbance_score",
                "appearance_change_score",
                "threshold",
                "appearance_threshold",
                "source",
            ],
        )
        boundary_writer.writeheader()
        boundary_writer.writerows(boundaries)

    summary = {
        "mode": "farneback_motion_compensated_dynamic",
        "output_video": str(output_path),
        "shot_boundaries_csv": str(boundary_csv_path),
        "trajectory_csv": str(trajectory_path) if args.save_trajectories else None,
        "selected_range": frame_range.__dict__,
        "history_mode": args.history_mode,
        "processed_frames": processed_frames,
        "written_frames": written_frames,
        "selected_gaze_count": int(len(selected)),
        "added_gaze_count": int(total_added),
        "final_active_count": int(len(active_points)),
        "maximum_active_count": int(maximum_active),
        "boundary_count": len(boundaries),
        "boundaries": boundaries,
        "termination_counts": termination_counts,
        "forward_backward_errors": summarize_numbers(fb_errors),
        "parameters": {
            "window_seconds": args.window_seconds,
            "flow_scale": args.flow_scale,
            "fb_threshold": fb_threshold,
            "shot_threshold": args.shot_threshold,
            "shot_hist_threshold": args.shot_hist_threshold,
            "brush_size": args.brush_size,
            "opacity": args.opacity,
            "heatmap_max": args.heatmap_max,
            "frame_skip": args.frame_skip,
            "output_fps": output_fps,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
