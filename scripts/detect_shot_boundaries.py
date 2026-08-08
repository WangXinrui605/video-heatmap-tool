from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.optical_flow import (
    FarnebackConfig,
    calculate_flow_pair,
    is_shot_boundary,
    load_manual_boundaries,
    resolve_frame_range,
    summarize_numbers,
)
from video_heatmap.video_io import get_video_info, require_cv2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect shot boundaries from Farneback flow-compensation disturbance."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--flow-scale", type=float, default=0.125)
    parser.add_argument("--shot-threshold", type=float, default=0.03)
    parser.add_argument(
        "--shot-hist-threshold",
        type=float,
        default=0.15,
        help="Appearance-change guard; use 0 for the paper-style disturbance-only rule.",
    )
    parser.add_argument("--manual-boundaries")
    parser.add_argument("--save-boundary-frames", action="store_true")
    args = parser.parse_args()

    if args.shot_threshold < 0:
        raise SystemExit("--shot-threshold must be non-negative.")
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

    config = FarnebackConfig(scale=args.flow_scale)
    manual = load_manual_boundaries(args.manual_boundaries)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "boundary_frames"
    if args.save_boundary_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    cv2 = require_cv2()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range.start_frame)
    ok, previous = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Could not read the first selected frame.")

    scores: list[float] = []
    boundaries: list[dict] = []
    for current_index in range(frame_range.start_frame + 1, frame_range.end_frame):
        ok, current = cap.read()
        if not ok:
            break
        pair = calculate_flow_pair(previous, current, config)
        score = pair.disturbance_score
        scores.append(score)
        is_manual = current_index in manual
        is_auto = is_shot_boundary(
            pair, args.shot_threshold, args.shot_hist_threshold
        )
        if is_manual or is_auto:
            source = "manual" if is_manual else "farneback_auto"
            record = {
                "frame": current_index,
                "time_seconds": current_index / video_info.fps,
                "disturbance_score": score,
                "appearance_change_score": pair.appearance_change_score,
                "threshold": args.shot_threshold,
                "appearance_threshold": args.shot_hist_threshold,
                "source": source,
            }
            boundaries.append(record)
            if args.save_boundary_frames:
                comparison = cv2.hconcat([previous, current])
                cv2.putText(
                    comparison,
                    (
                        f"cut {current_index} flow={score:.4f} "
                        f"hist={pair.appearance_change_score:.4f}"
                    ),
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(
                    str(frame_dir / f"boundary_{current_index:06d}.jpg"),
                    comparison,
                )
        previous = current
    cap.release()

    csv_path = out_dir / "shot_boundaries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
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
        "video": str(Path(args.video)),
        "shot_boundaries_csv": str(csv_path),
        "boundary_count": len(boundaries),
        "boundaries": boundaries,
        "disturbance_scores": summarize_numbers(scores),
        "selected_range": frame_range.__dict__,
        "parameters": {
            "flow_scale": args.flow_scale,
            "shot_threshold": args.shot_threshold,
            "shot_hist_threshold": args.shot_hist_threshold,
            "manual_boundaries": args.manual_boundaries,
        },
    }
    summary_path = out_dir / "shot_boundary_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
