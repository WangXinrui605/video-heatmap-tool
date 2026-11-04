from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.optical_flow import (
    AUTOMATIC_EVENT_LABEL,
    O2_SCHEMA_VERSION,
    FarnebackConfig,
    calculate_flow_pair,
    is_visual_discontinuity_candidate,
    load_manual_boundaries,
    resolve_frame_range,
    summarize_numbers,
)
from video_heatmap.video_io import get_video_info, require_cv2


ANNOTATION_FIELDS = [
    "frame",
    "time_seconds",
    "automatic_label",
    "reviewed_label",
    "review_confidence",
    "reviewer",
    "review_window",
    "action",
    "confirmed_hard_cut",
    "evidence_path",
    "notes",
]


def write_review_clip(cv2, video: str, output: Path, frame: int, fps: float,
                      frame_count: int) -> tuple[int, int]:
    radius = int(round(0.5 * fps))
    start = max(0, frame - radius)
    end = min(frame_count - 1, frame + radius)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read review window for frame {frame}.")
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / 2.0),
        (first.shape[1], first.shape[0]),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create review clip: {output}")
    current = first
    for index in range(start, end + 1):
        if index > start:
            ok, current = cap.read()
            if not ok:
                break
        annotated = current.copy()
        colour = (0, 255, 255) if index != frame else (0, 165, 255)
        cv2.putText(
            annotated,
            f"visual candidate={frame} review frame={index}",
            (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA,
        )
        writer.write(annotated)
    writer.release()
    cap.release()
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record neutral visual-discontinuity candidates and review clips."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--flow-scale", type=float, default=0.125)
    parser.add_argument("--shot-threshold", type=float, default=0.03)
    parser.add_argument("--shot-hist-threshold", type=float, default=0.15)
    parser.add_argument("--manual-boundaries")
    parser.add_argument("--no-auto-event", action="store_true")
    parser.add_argument("--reviewer", default="O2_executor")
    parser.add_argument("--save-boundary-frames", action="store_true",
                        help="Legacy alias; review clips are always generated for candidates.")
    args = parser.parse_args()

    if args.shot_threshold < 0:
        raise SystemExit("--shot-threshold must be non-negative.")
    video_info = get_video_info(args.video)
    try:
        frame_range = resolve_frame_range(
            video_info, start_frame=args.start_frame, end_frame=args.end_frame,
            start_time=args.start_time, end_time=args.end_time,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    config = FarnebackConfig(scale=args.flow_scale)
    manual = load_manual_boundaries(args.manual_boundaries)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir = out_dir / "events"
    review_dir.mkdir(parents=True, exist_ok=True)
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
    candidates: list[dict] = []
    for current_index in range(frame_range.start_frame + 1, frame_range.end_frame):
        ok, current = cap.read()
        if not ok:
            break
        pair = calculate_flow_pair(previous, current, config)
        scores.append(pair.disturbance_score)
        automatic = (not args.no_auto_event) and is_visual_discontinuity_candidate(
            pair, args.shot_threshold, args.shot_hist_threshold
        )
        if automatic:
            candidates.append({
                "frame": current_index,
                "time_seconds": current_index / video_info.fps,
                "disturbance_score": pair.disturbance_score,
                "appearance_change_score": pair.appearance_change_score,
                "threshold": args.shot_threshold,
                "appearance_threshold": args.shot_hist_threshold,
                "automatic_label": AUTOMATIC_EVENT_LABEL,
            })
        previous = current
    cap.release()

    annotations: list[dict] = []
    for event in candidates:
        frame = int(event["frame"])
        clip_path = review_dir / f"frame_{frame:06d}_review.mp4"
        review_start, review_end = write_review_clip(
            cv2, args.video, clip_path, frame, video_info.fps, video_info.frame_count
        )
        confirmed = frame in manual
        annotations.append({
            "frame": frame,
            "time_seconds": event["time_seconds"],
            "automatic_label": AUTOMATIC_EVENT_LABEL,
            "reviewed_label": "confirmed_hard_cut" if confirmed else "unreviewed",
            "review_confidence": "high" if confirmed else "",
            "reviewer": args.reviewer if confirmed else "",
            "review_window": f"[{review_start},{review_end}]",
            "action": "confirmed_hard_cut_reset" if confirmed else "no_reset",
            "confirmed_hard_cut": confirmed,
            "evidence_path": str(clip_path),
            "notes": "manual boundary" if confirmed else "pending temporal review",
        })

    csv_path = out_dir / "event_annotations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(annotations)
    json_path = out_dir / "event_annotations.json"
    json_path.write_text(json.dumps({
        "schema_version": O2_SCHEMA_VERSION,
        "fields": ANNOTATION_FIELDS,
        "events": annotations,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    candidate_path = out_dir / "visual_discontinuity_candidates.csv"
    with candidate_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(candidates[0].keys()) if candidates else [
            "frame", "time_seconds", "disturbance_score", "appearance_change_score",
            "threshold", "appearance_threshold", "automatic_label",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)

    summary = {
        "schema_version": O2_SCHEMA_VERSION,
        "video": str(Path(args.video)),
        "event_annotations_csv": str(csv_path),
        "event_annotations_json": str(json_path),
        "visual_candidates_csv": str(candidate_path),
        "event_counts": {
            AUTOMATIC_EVENT_LABEL: len(candidates),
            "reviewed_continuous_non_cut": 0,
            "confirmed_hard_cut": sum(int(row["confirmed_hard_cut"]) for row in annotations),
            "uncertain_event": 0,
        },
        "events": candidates,
        "disturbance_scores": summarize_numbers(scores),
        "selected_range": frame_range.__dict__,
        "parameters": {
            "flow_scale": args.flow_scale,
            "shot_threshold": args.shot_threshold,
            "shot_hist_threshold": args.shot_hist_threshold,
            "manual_boundaries": args.manual_boundaries,
            "auto_event_enabled": not args.no_auto_event,
            "review_clip_radius_seconds": 0.5,
            "review_clip_speed": "0.5x",
        },
    }
    summary_path = out_dir / "visual_event_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
