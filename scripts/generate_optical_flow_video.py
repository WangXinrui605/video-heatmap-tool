from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt
from video_heatmap.heatmap import build_final_heatmap, overlay_heatmap
from video_heatmap.optical_flow import (
    AUTOMATIC_EVENT_LABEL,
    O2_SCHEMA_VERSION,
    EventPolicyState,
    FarnebackConfig,
    advance_event_policy,
    calculate_flow_pair,
    gaze_frame_indices,
    is_visual_discontinuity_candidate,
    load_event_annotations,
    load_manual_boundaries,
    next_technical_segment_id,
    propagate_points,
    resolve_frame_range,
    summarize_numbers,
)
from video_heatmap.rendering import HeatmapGradient, rendering_summary, resolve_heat_scale, scale_heatmap
from video_heatmap.video_io import get_video_info, require_cv2
from video_heatmap.tool_app.progress import ProgressEmitter


ANNOTATION_FIELDS = [
    "frame", "time_seconds", "automatic_label", "reviewed_label",
    "review_confidence", "reviewer", "review_window", "action",
    "confirmed_hard_cut", "evidence_path", "notes",
]
TRAJECTORY_FIELDS = [
    "frame", "time_seconds", "gaze_id", "sample_time", "x", "y", "active",
    "termination_reason", "forward_backward_error", "technical_segment_id",
    "flow_interval_id", "loss_episode_id",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a recent-window optical-flow video with explicit event semantics."
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
    parser.add_argument("--fb-threshold", type=float, default=12.0)
    parser.add_argument("--shot-threshold", type=float, default=0.03)
    parser.add_argument("--shot-hist-threshold", type=float, default=0.15)
    parser.add_argument("--manual-boundaries")
    parser.add_argument("--event-annotations")
    parser.add_argument("--event-policy", choices=["candidate-only", "flow-loss"],
                        default="candidate-only")
    parser.add_argument("--recovery-stable-frames", type=int, default=3)
    parser.add_argument("--no-auto-event", action="store_true")
    parser.add_argument("--brush-size", type=int, default=40)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--heatmap-max", type=float, default=20.0)
    parser.add_argument("--gradient", choices=[item.value for item in HeatmapGradient], default="jet")
    parser.add_argument("--heat-scale-mode", choices=("auto", "fixed"), default="fixed")
    parser.add_argument("--heat-scale-maximum", type=float)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()

    if args.history_mode == "window" and args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive.")
    if args.recovery_stable_frames < 1 or args.frame_skip < 1:
        raise SystemExit("Recovery stable frames and frame skip must be at least 1.")
    if args.brush_size < 1 or args.heatmap_max <= 0 or not 0 <= args.opacity <= 1:
        raise SystemExit("Invalid heatmap rendering parameters.")
    if args.heat_scale_mode == "fixed" and args.heat_scale_maximum is None:
        args.heat_scale_maximum = args.heatmap_max
    try:
        heat_scale = resolve_heat_scale(args.heat_scale_mode, args.heat_scale_maximum)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    video_info = get_video_info(args.video)
    frame_range = resolve_frame_range(
        video_info, start_frame=args.start_frame, end_frame=args.end_frame,
        start_time=args.start_time, end_time=args.end_time,
    )
    gaze = load_gaze_txt(args.gaze, video_width=video_info.width, video_height=video_info.height)
    selected = gaze.dataframe[gaze.dataframe["valid"] & gaze.dataframe["time_seconds"].notna()].copy()
    selected = selected[
        (selected["time_seconds"] >= frame_range.start_time)
        & (selected["time_seconds"] <= frame_range.end_time)
    ].sort_values("time_seconds")
    if selected.empty:
        raise SystemExit("No valid gaze samples in the selected range.")
    selected["assigned_frame"] = gaze_frame_indices(selected["time_seconds"].to_numpy(float), video_info.fps)
    selected = selected[
        (selected["assigned_frame"] >= frame_range.start_frame)
        & (selected["assigned_frame"] < frame_range.end_frame)
    ]
    rows_by_frame = {int(frame): group for frame, group in selected.groupby("assigned_frame", sort=False)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "optical_flow_window_video.mp4"
    summary_path = out_dir / "optical_flow_video_summary.json"
    event_csv_path = out_dir / "runtime_event_annotations.csv"
    event_json_path = out_dir / "runtime_event_annotations.json"
    trajectory_path = out_dir / "optical_flow_gaze_trajectories.csv"
    config = FarnebackConfig(scale=args.flow_scale)
    fb_threshold = None if args.fb_threshold <= 0 else args.fb_threshold
    manual = load_manual_boundaries(args.manual_boundaries)
    reviewed = load_event_annotations(args.event_annotations)
    cv2 = require_cv2()
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range.start_frame)
    output_fps = video_info.fps / args.frame_skip
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps,
                             (video_info.width, video_info.height))
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        raise SystemExit("Could not open input or output video.")

    trajectory_stream = None
    trajectory_writer = None
    if args.save_trajectories:
        trajectory_stream = trajectory_path.open("w", newline="", encoding="utf-8")
        trajectory_writer = csv.DictWriter(trajectory_stream, fieldnames=TRAJECTORY_FIELDS)
        trajectory_writer.writeheader()

    active_points = np.empty((0, 2), np.float32)
    active_ids = np.empty(0, np.int64)
    active_times = np.empty(0, np.float64)
    active_errors = np.empty(0, np.float32)
    next_gaze_id = 0
    policy_state = EventPolicyState()
    technical_segment_id = 0
    events: list[dict] = []
    loss_episodes: list[dict] = []
    fb_errors: list[float] = []
    termination_counts = {key: 0 for key in (
        "window_expired", "flow_or_bounds", "flow_tracking_lost", "confirmed_hard_cut"
    )}
    processed_frames = written_frames = total_added = maximum_active = 0

    def write_rows(frame: int, points: np.ndarray, ids: np.ndarray, times: np.ndarray,
                   errors: np.ndarray, active: bool, reason: str) -> None:
        if trajectory_writer is None:
            return
        for point, gaze_id, sample_time, error in zip(points, ids, times, errors):
            trajectory_writer.writerow({
                "frame": frame, "time_seconds": frame / video_info.fps,
                "gaze_id": int(gaze_id), "sample_time": float(sample_time),
                "x": float(point[0]), "y": float(point[1]), "active": active,
                "termination_reason": reason,
                "forward_backward_error": "" if not np.isfinite(error) else float(error),
                "technical_segment_id": technical_segment_id,
                "flow_interval_id": policy_state.flow_interval_id,
                "loss_episode_id": "" if policy_state.loss_episode_id is None else policy_state.loss_episode_id,
            })

    def clear_active(frame: int, reason: str) -> int:
        nonlocal active_points, active_ids, active_times, active_errors
        count = len(active_points)
        if count:
            write_rows(frame, active_points, active_ids, active_times, active_errors, False, reason)
        termination_counts[reason] += count
        active_points = np.empty((0, 2), np.float32)
        active_ids = np.empty(0, np.int64)
        active_times = np.empty(0, np.float64)
        active_errors = np.empty(0, np.float32)
        return count

    def add_gaze(frame: int) -> None:
        nonlocal active_points, active_ids, active_times, active_errors, next_gaze_id, total_added
        group = rows_by_frame.get(frame)
        if group is None:
            return
        points = group[["gaze_x", "gaze_y"]].to_numpy(np.float32)
        times = group["time_seconds"].to_numpy(np.float64)
        ids = np.arange(next_gaze_id, next_gaze_id + len(points), dtype=np.int64)
        next_gaze_id += len(points)
        active_points = np.vstack([active_points, points])
        active_ids = np.concatenate([active_ids, ids])
        active_times = np.concatenate([active_times, times])
        active_errors = np.concatenate([active_errors, np.full(len(points), np.nan, np.float32)])
        total_added += len(points)

    ok, previous = cap.read()
    if not ok:
        raise SystemExit("Could not read first selected frame.")
    add_gaze(frame_range.start_frame)
    current_status = "normal"
    progress = ProgressEmitter.from_environment(frame_range.end_frame - frame_range.start_frame)
    if progress:
        progress.emit(0, "preparing")
    for frame in range(frame_range.start_frame, frame_range.end_frame):
        candidate = confirmed = False
        transition = None
        if frame == frame_range.start_frame:
            current = previous
        else:
            ok, current = cap.read()
            if not ok:
                break
            pair = calculate_flow_pair(previous, current, config)
            candidate = (not args.no_auto_event) and is_visual_discontinuity_candidate(
                pair, args.shot_threshold, args.shot_hist_threshold
            )
            annotation = reviewed.get(frame, {})
            confirmed = frame in manual or bool(annotation.get("confirmed_hard_cut", False))
            transition = advance_event_policy(
                policy_state, candidate=candidate, confirmed_hard_cut=confirmed,
                event_policy=args.event_policy,
                recovery_stable_frames=args.recovery_stable_frames,
            )
            old_state = policy_state
            policy_state = transition.state
            if confirmed:
                cleared = clear_active(frame, "confirmed_hard_cut")
                technical_segment_id = next_technical_segment_id(technical_segment_id, True)
                current_status = "confirmed hard cut"
            elif args.event_policy == "flow-loss" and transition.entered_loss:
                cleared = clear_active(frame, "flow_tracking_lost")
                loss_episodes.append({
                    "loss_episode_id": policy_state.loss_episode_id,
                    "start_frame": frame, "last_candidate_frame": frame,
                    "recovery_frame": None, "end_frame": None,
                    "duration_seconds": None, "terminated_trajectories": cleared,
                })
                current_status = "flow tracking lost"
            elif old_state.loss_active and candidate:
                loss_episodes[-1]["last_candidate_frame"] = frame
                current_status = "flow tracking lost"
            elif transition.recovered:
                episode = loss_episodes[-1]
                episode["recovery_frame"] = frame
                episode["end_frame"] = frame
                episode["duration_seconds"] = (frame - episode["start_frame"]) / video_info.fps
                current_status = "normal"
            elif policy_state.loss_active:
                current_status = "flow tracking lost"
            else:
                current_status = "visual candidate" if candidate else "normal"

            if candidate or confirmed:
                reviewed_label = (
                    "confirmed_hard_cut"
                    if confirmed
                    else str(annotation.get("reviewed_label", "unreviewed"))
                )
                action = transition.action
                events.append({
                    "frame": frame, "time_seconds": frame / video_info.fps,
                    "automatic_label": AUTOMATIC_EVENT_LABEL if candidate else "",
                    "reviewed_label": reviewed_label,
                    "review_confidence": annotation.get("review_confidence", ""),
                    "reviewer": annotation.get("reviewer", ""),
                    "review_window": annotation.get("review_window", ""),
                    "action": action,
                    "confirmed_hard_cut": confirmed,
                    "evidence_path": annotation.get("evidence_path", ""),
                    "notes": annotation.get("notes", ""),
                    "disturbance_score": pair.disturbance_score,
                    "appearance_change_score": pair.appearance_change_score,
                    "technical_segment_id": technical_segment_id,
                    "flow_interval_id": policy_state.flow_interval_id,
                    "loss_episode_id": policy_state.loss_episode_id,
                })

            if not confirmed and not policy_state.loss_active and not transition.recovered and len(active_points):
                result = propagate_points(active_points, pair, video_info.width, video_info.height, fb_threshold)
                invalid = ~result.valid
                if np.any(invalid):
                    write_rows(frame, result.points[invalid], active_ids[invalid], active_times[invalid],
                               result.forward_backward_error[invalid], False, "flow_or_bounds")
                termination_counts["flow_or_bounds"] += int(np.count_nonzero(invalid))
                fb_errors.extend(result.forward_backward_error[result.valid].tolist())
                active_points = result.points[result.valid]
                active_ids = active_ids[result.valid]
                active_times = active_times[result.valid]
                active_errors = result.forward_backward_error[result.valid]

            if not policy_state.loss_active:
                add_gaze(frame)

        frame_time = frame / video_info.fps
        if args.history_mode == "window" and len(active_points):
            expired = active_times < frame_time - args.window_seconds
            if np.any(expired):
                write_rows(frame, active_points[expired], active_ids[expired], active_times[expired],
                           active_errors[expired], False, "window_expired")
            termination_counts["window_expired"] += int(np.count_nonzero(expired))
            keep = ~expired
            active_points, active_ids = active_points[keep], active_ids[keep]
            active_times, active_errors = active_times[keep], active_errors[keep]

        maximum_active = max(maximum_active, len(active_points))
        if (frame - frame_range.start_frame) % args.frame_skip == 0:
            heatmap = build_final_heatmap(active_points, video_info.width, video_info.height, args.brush_size)
            heatmap_uint8, _ = scale_heatmap(heatmap, heat_scale)
            overlay, _ = overlay_heatmap(current, heatmap_uint8, args.opacity, args.gradient)
            if args.debug_overlay or current_status != "normal":
                colour = {
                    "visual candidate": (0, 255, 255),
                    "flow tracking lost": (0, 0, 255),
                    "confirmed hard cut": (255, 0, 255),
                }.get(current_status, (255, 255, 255))
                cv2.putText(overlay, f"{current_status} frame={frame} active={len(active_points)}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
            writer.write(overlay)
            written_frames += 1
        write_rows(frame, active_points, active_ids, active_times, active_errors, True, "")
        processed_frames += 1
        if progress:
            progress.emit(processed_frames, "processing")
        previous = current

    if policy_state.loss_active and loss_episodes:
        episode = loss_episodes[-1]
        episode["end_frame"] = frame_range.start_frame + processed_frames - 1
        episode["duration_seconds"] = (episode["end_frame"] - episode["start_frame"] + 1) / video_info.fps
    cap.release()
    writer.release()
    if progress and processed_frames == progress.total:
        progress.emit(processed_frames, "finalising")
    if trajectory_stream:
        trajectory_stream.close()

    with event_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS, extrasaction="ignore")
        writer_csv.writeheader()
        writer_csv.writerows(events)
    event_json_path.write_text(json.dumps({
        "schema_version": O2_SCHEMA_VERSION, "fields": ANNOTATION_FIELDS, "events": events,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    event_counts = {
        AUTOMATIC_EVENT_LABEL: sum(e["automatic_label"] == AUTOMATIC_EVENT_LABEL for e in events),
        "reviewed_continuous_non_cut": sum(e["reviewed_label"] == "continuous_non_cut" for e in events),
        "confirmed_hard_cut": sum(bool(e["confirmed_hard_cut"]) for e in events),
        "uncertain_event": sum(e["reviewed_label"] == "uncertain" for e in events),
    }
    summary = {
        "schema_version": O2_SCHEMA_VERSION,
        "mode": "farneback_recent_window_dynamic" if args.history_mode == "window" else "farneback_short_clip_cumulative_diagnostic",
        "output_video": str(output_path),
        "runtime_event_annotations_csv": str(event_csv_path),
        "runtime_event_annotations_json": str(event_json_path),
        "trajectory_csv": str(trajectory_path) if args.save_trajectories else None,
        "selected_range": frame_range.__dict__, "history_mode": args.history_mode,
        "processed_frames": processed_frames, "written_frames": written_frames,
        "selected_gaze_count": int(len(selected)), "added_gaze_count": int(total_added),
        "final_active_count": int(len(active_points)), "maximum_active_count": int(maximum_active),
        "event_counts": event_counts, "events": events,
        "termination_counts": termination_counts,
        "termination_count_unit": "trajectories",
        "event_count_unit": "frame-level observations/reviewed events",
        "technical_segment_count": technical_segment_id + 1,
        "final_technical_segment_id": technical_segment_id,
        "flow_interval_count": policy_state.flow_interval_id + 1,
        "final_flow_interval_id": policy_state.flow_interval_id,
        "loss_episode_count": len(loss_episodes), "loss_episodes": loss_episodes,
        "forward_backward_errors": summarize_numbers(fb_errors),
        "parameters": {
            "event_policy": args.event_policy,
            "recovery_stable_frames": args.recovery_stable_frames,
            "auto_event_enabled": not args.no_auto_event,
            "window_seconds": args.window_seconds, "flow_scale": args.flow_scale,
            "fb_threshold": fb_threshold, "shot_threshold": args.shot_threshold,
            "shot_hist_threshold": args.shot_hist_threshold, "manual_boundaries": args.manual_boundaries,
            "event_annotations": args.event_annotations, "brush_size": args.brush_size,
            "opacity": args.opacity, "heatmap_max": args.heatmap_max,
            "frame_skip": args.frame_skip, "output_fps": output_fps,
            "output_update_interval_semantics": "write_every_nth_output_frame_only; all frames drive optical flow and event policy",
            "rendering": rendering_summary(args.gradient, heat_scale, resolved_maximum=heat_scale.maximum),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
