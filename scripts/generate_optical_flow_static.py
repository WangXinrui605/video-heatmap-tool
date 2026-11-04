from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt
from video_heatmap.heatmap import build_final_heatmap, normalize_with_max, overlay_heatmap
from video_heatmap.optical_flow import (
    AUTOMATIC_EVENT_LABEL, O2_SCHEMA_VERSION, EventPolicyState, FarnebackConfig,
    advance_event_policy, calculate_flow_pair, group_gaze_by_frame,
    is_visual_discontinuity_candidate, load_event_annotations, load_manual_boundaries,
    next_technical_segment_id, propagate_points, resolve_frame_range, summarize_numbers,
)
from video_heatmap.video_io import get_video_info, require_cv2


def write_interval(cv2, out_dir: Path, technical_id: int, flow_id: int,
                   start: int, end: int, key_frame: np.ndarray,
                   originals: np.ndarray, compensated: np.ndarray,
                   brush: int, opacity: float) -> dict:
    prefix = f"technical_segment_{technical_id:03d}_flow_interval_{flow_id:03d}"
    screen = build_final_heatmap(originals, key_frame.shape[1], key_frame.shape[0], brush)
    motion = build_final_heatmap(compensated, key_frame.shape[1], key_frame.shape[0], brush)
    common_max = max(float(np.max(screen)) if screen.size else 0.0,
                     float(np.max(motion)) if motion.size else 0.0)
    screen_overlay, _ = overlay_heatmap(key_frame, normalize_with_max(screen, common_max), opacity)
    motion_overlay, _ = overlay_heatmap(key_frame, normalize_with_max(motion, common_max), opacity)
    comparison = cv2.hconcat([screen_overlay, motion_overlay])
    cv2.putText(comparison, "screen-space", (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(comparison, "flow interval diagnostic", (key_frame.shape[1] + 20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    path = out_dir / f"{prefix}_comparison.jpg"
    cv2.imwrite(str(path), comparison)
    return {
        "technical_segment_id": technical_id, "flow_interval_id": flow_id,
        "start_frame": start, "end_frame_inclusive": end, "key_frame": end,
        "original_gaze_count": int(len(originals)),
        "compensated_gaze_count": int(len(compensated)), "comparison_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate O2 static diagnostics split only by confirmed cuts/flow intervals."
    )
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--flow-scale", type=float, default=0.125)
    parser.add_argument("--fb-threshold", type=float, default=12.0)
    parser.add_argument("--shot-threshold", type=float, default=0.03)
    parser.add_argument("--shot-hist-threshold", type=float, default=0.15)
    parser.add_argument("--manual-boundaries")
    parser.add_argument("--event-annotations")
    parser.add_argument("--event-policy", choices=["candidate-only", "flow-loss"], default="candidate-only")
    parser.add_argument("--recovery-stable-frames", type=int, default=3)
    parser.add_argument("--no-auto-event", action="store_true")
    parser.add_argument("--brush-size", type=int, default=40)
    parser.add_argument("--opacity", type=float, default=0.55)
    args = parser.parse_args()
    if args.recovery_stable_frames < 1:
        raise SystemExit("--recovery-stable-frames must be at least 1.")

    info = get_video_info(args.video)
    frame_range = resolve_frame_range(
        info, start_frame=args.start_frame, end_frame=args.end_frame,
        start_time=args.start_time, end_time=args.end_time,
    )
    gaze = load_gaze_txt(args.gaze, video_width=info.width, video_height=info.height)
    selected = gaze.dataframe[gaze.dataframe["valid"] & gaze.dataframe["time_seconds"].notna()].copy()
    selected = selected[(selected["time_seconds"] >= frame_range.start_time)
                        & (selected["time_seconds"] <= frame_range.end_time)]
    if selected.empty:
        raise SystemExit("No valid gaze samples in the selected range.")
    grouped = group_gaze_by_frame(selected["time_seconds"].to_numpy(float),
                                  selected[["gaze_x", "gaze_y"]].to_numpy(np.float32),
                                  info.fps, frame_range)
    out_dir = Path(args.out_dir)
    interval_dir = out_dir / "static_intervals"
    interval_dir.mkdir(parents=True, exist_ok=True)
    manual = load_manual_boundaries(args.manual_boundaries)
    reviewed = load_event_annotations(args.event_annotations)
    config = FarnebackConfig(scale=args.flow_scale)
    fb_threshold = None if args.fb_threshold <= 0 else args.fb_threshold
    cv2 = require_cv2()
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range.start_frame)
    ok, previous = cap.read()
    if not ok:
        raise SystemExit("Could not read first selected frame.")

    active = grouped.get(frame_range.start_frame, np.empty((0, 2), np.float32)).copy()
    originals = active.copy()
    policy_state = EventPolicyState()
    technical_id = 0
    interval_start = frame_range.start_frame
    records: list[dict] = []
    events: list[dict] = []
    fb_errors: list[float] = []
    termination_counts = {"window_expired": 0, "flow_or_bounds": 0,
                          "flow_tracking_lost": 0, "confirmed_hard_cut": 0}

    for frame in range(frame_range.start_frame + 1, frame_range.end_frame):
        ok, current = cap.read()
        if not ok:
            break
        pair = calculate_flow_pair(previous, current, config)
        candidate = (not args.no_auto_event) and is_visual_discontinuity_candidate(
            pair, args.shot_threshold, args.shot_hist_threshold
        )
        annotation = reviewed.get(frame, {})
        confirmed = frame in manual or bool(annotation.get("confirmed_hard_cut", False))
        old_state = policy_state
        transition = advance_event_policy(
            policy_state, candidate=candidate, confirmed_hard_cut=confirmed,
            event_policy=args.event_policy, recovery_stable_frames=args.recovery_stable_frames,
        )
        policy_state = transition.state
        if candidate or confirmed:
            events.append({
                "frame": frame, "time_seconds": frame / info.fps,
                "automatic_label": AUTOMATIC_EVENT_LABEL if candidate else "",
                "reviewed_label": "confirmed_hard_cut" if confirmed else annotation.get("reviewed_label", "unreviewed"),
                "action": transition.action, "confirmed_hard_cut": confirmed,
                "technical_segment_id": technical_id,
                "flow_interval_id": policy_state.flow_interval_id,
                "loss_episode_id": policy_state.loss_episode_id,
            })

        if confirmed or transition.entered_loss:
            records.append(write_interval(
                cv2, interval_dir, technical_id, old_state.flow_interval_id,
                interval_start, frame - 1, previous, originals, active,
                args.brush_size, args.opacity,
            ))
            reason = "confirmed_hard_cut" if confirmed else "flow_tracking_lost"
            termination_counts[reason] += len(active)
            active = np.empty((0, 2), np.float32)
            originals = np.empty((0, 2), np.float32)
            if confirmed:
                technical_id = next_technical_segment_id(technical_id, True)
                interval_start = frame
            else:
                interval_start = -1
        elif transition.recovered:
            interval_start = frame
        elif not policy_state.loss_active and len(active):
            result = propagate_points(active, pair, info.width, info.height, fb_threshold)
            termination_counts["flow_or_bounds"] += int(np.count_nonzero(~result.valid))
            fb_errors.extend(result.forward_backward_error[result.valid].tolist())
            active = result.points[result.valid]

        if not policy_state.loss_active:
            new = grouped.get(frame)
            if new is not None:
                active = np.vstack([active, new])
                originals = np.vstack([originals, new])
        previous = current
    cap.release()
    if interval_start >= 0:
        records.append(write_interval(
            cv2, interval_dir, technical_id, policy_state.flow_interval_id,
            interval_start, frame_range.end_frame - 1, previous, originals, active,
            args.brush_size, args.opacity,
        ))

    summary = {
        "schema_version": O2_SCHEMA_VERSION,
        "mode": "farneback_short_clip_static_diagnostic",
        "selected_range": frame_range.__dict__, "selected_gaze_count": int(len(selected)),
        "static_intervals": records,
        "event_counts": {
            AUTOMATIC_EVENT_LABEL: sum(e["automatic_label"] == AUTOMATIC_EVENT_LABEL for e in events),
            "reviewed_continuous_non_cut": sum(e["reviewed_label"] == "continuous_non_cut" for e in events),
            "confirmed_hard_cut": sum(bool(e["confirmed_hard_cut"]) for e in events),
            "uncertain_event": sum(e["reviewed_label"] == "uncertain" for e in events),
        },
        "events": events, "termination_counts": termination_counts,
        "event_count_unit": "frame-level observations/reviewed events",
        "termination_count_unit": "trajectories",
        "technical_segment_count": technical_id + 1,
        "flow_interval_count": policy_state.flow_interval_id + 1,
        "loss_episode_count": 0 if policy_state.loss_episode_id is None else policy_state.loss_episode_id + 1,
        "forward_backward_errors": summarize_numbers(fb_errors),
        "parameters": {
            "event_policy": args.event_policy,
            "recovery_stable_frames": args.recovery_stable_frames,
            "auto_event_enabled": not args.no_auto_event,
            "flow_scale": args.flow_scale, "fb_threshold": fb_threshold,
            "shot_threshold": args.shot_threshold,
            "shot_hist_threshold": args.shot_hist_threshold,
            "manual_boundaries": args.manual_boundaries,
            "event_annotations": args.event_annotations,
        },
    }
    summary_path = out_dir / "optical_flow_static_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
