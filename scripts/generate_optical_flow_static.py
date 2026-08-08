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
    group_gaze_by_frame,
    is_shot_boundary,
    load_manual_boundaries,
    propagate_points,
    resolve_frame_range,
    summarize_numbers,
)
from video_heatmap.video_io import get_video_info, require_cv2


def write_shot_outputs(
    cv2,
    shot_dir: Path,
    shot_index: int,
    start_frame: int,
    end_frame: int,
    key_frame: np.ndarray,
    original_points: np.ndarray,
    compensated_points: np.ndarray,
    brush_size: int,
    opacity: float,
) -> tuple[dict, np.ndarray]:
    prefix = f"shot_{shot_index:03d}_{start_frame:06d}_{end_frame:06d}"
    screen_heatmap = build_final_heatmap(
        original_points, key_frame.shape[1], key_frame.shape[0], brush_size
    )
    compensated_heatmap = build_final_heatmap(
        compensated_points, key_frame.shape[1], key_frame.shape[0], brush_size
    )
    common_max = max(
        float(np.max(screen_heatmap)) if screen_heatmap.size else 0.0,
        float(np.max(compensated_heatmap)) if compensated_heatmap.size else 0.0,
    )
    screen_uint8 = normalize_with_max(screen_heatmap, common_max)
    compensated_uint8 = normalize_with_max(compensated_heatmap, common_max)
    screen_overlay, _ = overlay_heatmap(key_frame, screen_uint8, opacity)
    compensated_overlay, _ = overlay_heatmap(
        key_frame, compensated_uint8, opacity
    )
    comparison = cv2.hconcat([screen_overlay, compensated_overlay])
    cv2.putText(
        comparison,
        "screen-space",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        comparison,
        "motion-compensated",
        (key_frame.shape[1] + 24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    key_path = shot_dir / f"{prefix}_key_frame.jpg"
    screen_path = shot_dir / f"{prefix}_screen_space.png"
    compensated_path = shot_dir / f"{prefix}_motion_compensated.png"
    comparison_path = shot_dir / f"{prefix}_comparison.jpg"
    cv2.imwrite(str(key_path), key_frame)
    cv2.imwrite(str(screen_path), screen_overlay)
    cv2.imwrite(str(compensated_path), compensated_overlay)
    cv2.imwrite(str(comparison_path), comparison)
    return (
        {
            "shot_index": shot_index,
            "start_frame": start_frame,
            "end_frame_inclusive": end_frame,
            "key_frame": end_frame,
            "original_gaze_count": int(len(original_points)),
            "compensated_gaze_count": int(len(compensated_points)),
            "key_frame_path": str(key_path),
            "screen_space_path": str(screen_path),
            "motion_compensated_path": str(compensated_path),
            "comparison_path": str(comparison_path),
        },
        comparison,
    )


def make_contact_sheet(cv2, comparisons: list[np.ndarray], path: Path) -> None:
    if not comparisons:
        return
    thumbs = []
    for image in comparisons[:100]:
        width = 960
        height = max(1, int(round(image.shape[0] * width / image.shape[1])))
        thumbs.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    if len(thumbs) % 2:
        thumbs.append(np.zeros_like(thumbs[-1]))
    rows = [cv2.hconcat(thumbs[index : index + 2]) for index in range(0, len(thumbs), 2)]
    cv2.imwrite(str(path), cv2.vconcat(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-style Farneback motion-compensated static heatmaps. "
            "Ranges crossing cuts are split into one output per shot."
        )
    )
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
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
    args = parser.parse_args()

    if args.brush_size < 1:
        raise SystemExit("--brush-size must be positive.")
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
    ]
    if selected.empty:
        raise SystemExit("No valid gaze samples in the selected range.")
    grouped = group_gaze_by_frame(
        selected["time_seconds"].to_numpy(dtype=float),
        selected[["gaze_x", "gaze_y"]].to_numpy(dtype=np.float32),
        video_info.fps,
        frame_range,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shot_dir = out_dir / "static_heatmaps"
    shot_dir.mkdir(parents=True, exist_ok=True)
    config = FarnebackConfig(scale=args.flow_scale)
    fb_threshold = None if args.fb_threshold <= 0 else args.fb_threshold
    manual = load_manual_boundaries(args.manual_boundaries)
    cv2 = require_cv2()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range.start_frame)
    ok, previous = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Could not read first selected frame.")

    first_background = previous.copy()
    active = grouped.get(
        frame_range.start_frame, np.empty((0, 2), dtype=np.float32)
    ).copy()
    originals = active.copy()
    shot_start = frame_range.start_frame
    shot_index = 0
    shot_records: list[dict] = []
    comparisons: list[np.ndarray] = []
    boundaries: list[dict] = []
    fb_errors: list[float] = []
    terminated = 0

    for current_index in range(frame_range.start_frame + 1, frame_range.end_frame):
        ok, current = cap.read()
        if not ok:
            break
        pair = calculate_flow_pair(previous, current, config)
        is_manual = current_index in manual
        is_boundary = is_manual or is_shot_boundary(
            pair, args.shot_threshold, args.shot_hist_threshold
        )
        if is_boundary:
            record, comparison = write_shot_outputs(
                cv2,
                shot_dir,
                shot_index,
                shot_start,
                current_index - 1,
                previous,
                originals,
                active,
                args.brush_size,
                args.opacity,
            )
            shot_records.append(record)
            comparisons.append(comparison)
            boundaries.append(
                {
                    "frame": current_index,
                    "time_seconds": current_index / video_info.fps,
                    "disturbance_score": pair.disturbance_score,
                    "appearance_change_score": pair.appearance_change_score,
                    "threshold": args.shot_threshold,
                    "appearance_threshold": args.shot_hist_threshold,
                    "source": "manual" if is_manual else "farneback_auto",
                }
            )
            shot_index += 1
            shot_start = current_index
            active = np.empty((0, 2), dtype=np.float32)
            originals = np.empty((0, 2), dtype=np.float32)
        elif len(active):
            result = propagate_points(
                active,
                pair,
                video_info.width,
                video_info.height,
                fb_threshold,
            )
            fb_errors.extend(result.forward_backward_error[result.valid].tolist())
            terminated += int(np.count_nonzero(~result.valid))
            active = result.points[result.valid]

        new_points = grouped.get(current_index)
        if new_points is not None:
            active = np.vstack([active, new_points])
            originals = np.vstack([originals, new_points])
        previous = current
    cap.release()

    last_frame = frame_range.end_frame - 1
    record, comparison = write_shot_outputs(
        cv2,
        shot_dir,
        shot_index,
        shot_start,
        last_frame,
        previous,
        originals,
        active,
        args.brush_size,
        args.opacity,
    )
    shot_records.append(record)
    comparisons.append(comparison)

    global_points = selected[["gaze_x", "gaze_y"]].to_numpy(dtype=np.float32)
    global_heatmap = build_final_heatmap(
        global_points, video_info.width, video_info.height, args.brush_size
    )
    global_uint8 = normalize_with_max(global_heatmap, float(np.max(global_heatmap)))
    global_overlay, _ = overlay_heatmap(first_background, global_uint8, args.opacity)
    global_path = out_dir / "global_screen_space_overlay.png"
    cv2.imwrite(str(global_path), global_overlay)
    contact_sheet_path = out_dir / "contact_sheet.jpg"
    make_contact_sheet(cv2, comparisons, contact_sheet_path)

    boundary_csv = out_dir / "shot_boundaries.csv"
    with boundary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
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
        writer.writeheader()
        writer.writerows(boundaries)

    summary = {
        "mode": "farneback_motion_compensated_static",
        "global_screen_space_overlay": str(global_path),
        "contact_sheet": str(contact_sheet_path),
        "shot_boundaries_csv": str(boundary_csv),
        "shots": shot_records,
        "boundaries": boundaries,
        "selected_range": frame_range.__dict__,
        "selected_gaze_count": int(len(selected)),
        "terminated_trajectories": terminated,
        "forward_backward_errors": summarize_numbers(fb_errors),
        "parameters": {
            "flow_scale": args.flow_scale,
            "fb_threshold": fb_threshold,
            "shot_threshold": args.shot_threshold,
            "shot_hist_threshold": args.shot_hist_threshold,
            "brush_size": args.brush_size,
            "opacity": args.opacity,
        },
    }
    summary_path = out_dir / "optical_flow_static_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
