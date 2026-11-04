from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.detectors import create_face_detector
from video_heatmap.detector_registry import FACE_DETECTOR_IDS, validate_face_detector_config
from video_heatmap.gaze_assignment import (
    ASSIGNMENT_SCHEMA_VERSION,
    SourceGazeSample,
    assign_source_rows_to_frames,
    audit_consistency_summary,
    duplicate_consumption_summary,
)
from video_heatmap.gaze_loader import load_gaze_txt, summarize_gaze
from video_heatmap.region_heatmap import (
    RegionHeatmapAccumulator,
    overlay_region_heatmap,
    region_to_screen,
    screen_to_region,
)
from video_heatmap.regions import RegionDetection, select_target_detection
from video_heatmap.rendering import HeatmapGradient, colorize, rendering_summary, resolve_heat_scale, scale_heatmap
from video_heatmap.time_range import resolve_selected_range
from video_heatmap.tool_app.progress import ProgressEmitter
from video_heatmap.tracking_state import TrackingLifecycle
from video_heatmap.video_io import get_video_info, require_cv2


def _nearest_timed_row(dataframe, frame_time: float, maximum_distance: float):
    """Return the nearest TXT row by rec_time, including invalid-gaze rows."""

    timed = dataframe[dataframe["time_valid"]]
    if timed.empty:
        return None
    times = timed["time_seconds"].to_numpy(dtype=float)
    position = int(np.searchsorted(times, frame_time, side="left"))
    candidates = []
    if position < len(times):
        candidates.append(position)
    if position > 0:
        candidates.append(position - 1)
    best = min(candidates, key=lambda index: (abs(times[index] - frame_time), index))
    if abs(times[best] - frame_time) > maximum_distance:
        return None
    return timed.iloc[best]


def _box_points(box: RegionDetection) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (int(round(box.x)), int(round(box.y))),
        (int(round(box.x + box.width)), int(round(box.y + box.height))),
    )


def _write_status(cv2, frame, lines: list[str]) -> None:
    panel_height = 14 + len(lines) * 25
    panel_width = min(frame.shape[1], 720)
    panel = frame[:panel_height, :panel_width]
    dark = np.zeros_like(panel)
    frame[:panel_height, :panel_width] = cv2.addWeighted(panel, 0.35, dark, 0.65, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 28 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a region-generic canonical heatmap using face detections as "
            "the first provider. Outside-face gaze is reported but never accumulated."
        )
    )
    parser.add_argument("--gaze", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector-provider", choices=FACE_DETECTOR_IDS, default="opencv-yunet")
    parser.add_argument("--model", required=True, help="Provider-specific detector model path")
    parser.add_argument("--detector-config", help="Provider config required by opencv-dnn-ssd")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--max-time-distance", type=float, default=0.05)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Write every Nth frame; all frames still drive assignment and lifecycle.",
    )
    parser.add_argument(
        "--evidence-purpose",
        choices=("coverage", "lifecycle_stress", "alignment_review"),
        default="coverage",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--min-size", type=int, default=30)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--canvas-width", type=int, default=256)
    parser.add_argument("--canvas-height", type=int, default=256)
    parser.add_argument("--brush-radius", type=int, default=12)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--gradient", choices=[item.value for item in HeatmapGradient], default="jet")
    parser.add_argument("--heat-scale-mode", choices=("auto", "fixed"), default="auto")
    parser.add_argument("--heat-scale-maximum", type=float)
    parser.add_argument(
        "--history-mode", choices=("cumulative", "window"), default="cumulative"
    )
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-freeze-seconds",
        type=float,
        default=0.5,
        help=(
            "Maximum detector-loss grace period showing the last valid rendered "
            "frame. After this, the overlay is hidden but canonical memory is retained."
        ),
    )
    parser.add_argument(
        "--hard-cut-frame",
        type=int,
        action="append",
        default=[],
        help=(
            "Optional manually confirmed hard-cut frame. Clears only transient freeze "
            "display state; canonical face memory is retained. May be repeated."
        ),
    )
    args = parser.parse_args()

    if args.max_time_distance < 0:
        raise SystemExit("--max-time-distance must not be negative")
    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be at least one")
    if args.canvas_width <= 1 or args.canvas_height <= 1:
        raise SystemExit("canonical canvas dimensions must be greater than one")
    if args.brush_radius <= 0:
        raise SystemExit("--brush-radius must be positive")
    if not 0.0 <= args.opacity <= 1.0:
        raise SystemExit("--opacity must be between 0 and 1")
    if args.history_mode == "window" and args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive for window history")
    if args.max_freeze_seconds < 0:
        raise SystemExit("--max-freeze-seconds must not be negative")
    if any(frame < 0 for frame in args.hard_cut_frame):
        raise SystemExit("--hard-cut-frame must not be negative")
    try:
        heat_scale = resolve_heat_scale(args.heat_scale_mode, args.heat_scale_maximum)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cv2 = require_cv2()
    video_info = get_video_info(args.video)
    default_end = None
    if all(
        value is None
        for value in (args.start_frame, args.end_frame, args.start_time, args.end_time)
    ):
        default_end = min(300, video_info.frame_count)
    try:
        selected_range = resolve_selected_range(
            video_info,
            start_frame=args.start_frame,
            end_frame=args.end_frame if args.end_frame is not None else default_end,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    gaze = load_gaze_txt(
        args.gaze, video_width=video_info.width, video_height=video_info.height
    )
    timed_rows = gaze.dataframe[gaze.dataframe["time_valid"]].sort_values(
        "time_seconds", kind="stable"
    )
    if timed_rows.empty:
        raise SystemExit("No gaze rows with valid rec_time were found.")
    display_valid_rows = timed_rows[timed_rows["valid"]]
    selected_timed_rows = timed_rows[
        (timed_rows["time_seconds"] >= selected_range.start_time_seconds)
        & (timed_rows["time_seconds"] <= selected_range.end_time_seconds)
    ]
    selected_valid_rows = selected_timed_rows[selected_timed_rows["valid"]]
    selected_invalid_rows = selected_timed_rows[~selected_timed_rows["valid"]]
    if selected_valid_rows.empty:
        raise SystemExit("No valid screen-gaze samples were found in the selected range.")
    source_samples = [
        SourceGazeSample(
            source_row_id=int(row["frame_index"]),
            time_seconds=float(row["time_seconds"]),
            gaze_x=float(row["gaze_x"]),
            gaze_y=float(row["gaze_y"]),
        )
        for _, row in selected_valid_rows.iterrows()
    ]
    assignment_plan = assign_source_rows_to_frames(
        source_samples,
        fps=video_info.fps,
        start_frame=selected_range.start_frame,
        end_frame_exclusive=selected_range.end_frame_exclusive,
        frame_skip=args.frame_skip,
    )

    detector_config: dict[str, object] = {
        "model_path": args.model,
        "confidence_threshold": args.confidence_threshold,
        "min_size": args.min_size,
    }
    if args.detector_provider == "opencv-yunet":
        detector_config["nms_threshold"] = args.nms_threshold
    if args.detector_config:
        detector_config["config_path"] = args.detector_config
    try:
        detector_config = validate_face_detector_config(args.detector_provider, detector_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    detector = create_face_detector(args.detector_provider, **detector_config)
    window_frames = (
        max(1, int(round(args.window_seconds * video_info.fps)))
        if args.history_mode == "window"
        else None
    )
    accumulator = RegionHeatmapAccumulator(
        args.canvas_width,
        args.canvas_height,
        radius=args.brush_radius,
        history_mode=args.history_mode,
        window_frames=window_frames,
    )
    max_freeze_frames = int(round(args.max_freeze_seconds * video_info.fps))
    lifecycle = TrackingLifecycle(max_freeze_frames=max_freeze_frames)
    hard_cut_frames = set(args.hard_cut_frame)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dynamic_path = out_dir / "face_relative_dynamic_overlay.mp4"
    debug_path = out_dir / "face_relative_debug_overlay.mp4"
    static_path = out_dir / "face_relative_static_heatmap.png"
    summary_path = out_dir / "face_relative_summary.json"
    audit_csv_path = out_dir / "gaze_assignment_audit.csv"
    audit_json_path = out_dir / "gaze_assignment_audit.json"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dynamic_writer = cv2.VideoWriter(
        str(dynamic_path), fourcc, video_info.fps / args.frame_skip, (video_info.width, video_info.height)
    )
    debug_writer = cv2.VideoWriter(
        str(debug_path), fourcc, video_info.fps / args.frame_skip, (video_info.width, video_info.height)
    )
    if not dynamic_writer.isOpened() or not debug_writer.isOpened():
        dynamic_writer.release()
        debug_writer.release()
        raise SystemExit(f"Could not create output videos in {out_dir}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        dynamic_writer.release()
        debug_writer.release()
        raise SystemExit(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_range.start_frame)

    counts = {
        "processed_frames": 0,
        "frames_with_detection": 0,
        "frames_without_detection": 0,
        "multiple_face_frames": 0,
        "selected_valid_source_rows": len(selected_valid_rows),
        "selected_invalid_source_rows": len(selected_invalid_rows),
        "assigned_unique_source_rows": 0,
        "unassigned_selected_valid_source_rows": 0,
        "accepted_inside_face_unique_samples": 0,
        "rejected_outside_face_unique_samples": 0,
        "no_detection_unique_samples": 0,
        "identity_uncertain_unique_samples": 0,
        "display_frame_matches": 0,
        "display_valid_frame_matches": 0,
        "display_invalid_frame_matches": 0,
        "display_unmatched_frames": 0,
    }
    audit_records: list[dict[str, object]] = []
    for _, row in selected_invalid_rows.iterrows():
        audit_records.append(
            {
                "source_row_id": int(row["frame_index"]),
                "sample_time_seconds": float(row["time_seconds"]),
                "assigned_frame": None,
                "assigned_frame_time_seconds": None,
                "classification": "invalid_source_row",
                "invalid_reason": str(row["invalid_reason"]),
                "screen_x": (
                    None if not np.isfinite(row["gaze_x"]) else float(row["gaze_x"])
                ),
                "screen_y": (
                    None if not np.isfinite(row["gaze_y"]) else float(row["gaze_y"])
                ),
                "target_box": None,
                "local_u": None,
                "local_v": None,
                "draw_back_x": None,
                "draw_back_y": None,
                "lifecycle_phase": None,
                "lifecycle_reason": None,
                "consumption_count": 0,
            }
        )
    consumption_counts = {sample.source_row_id: 0 for sample in source_samples}
    selected_confidences: list[float] = []
    started_at = perf_counter()
    frame_index = selected_range.start_frame
    progress = ProgressEmitter.from_environment(
        selected_range.end_frame_exclusive - selected_range.start_frame
    )
    if progress:
        progress.emit(0, "preparing")
    last_valid_dynamic = None
    last_valid_frame_index: int | None = None
    previous_raw_frame = None
    review_dir = out_dir / "alignment_review_frames"
    pending_review_after: set[int] = set()
    if args.evidence_purpose == "alignment_review":
        review_dir.mkdir(parents=True, exist_ok=True)
    try:
        while frame_index < selected_range.end_frame_exclusive:
            ok, frame = cap.read()
            if not ok:
                break
            counts["processed_frames"] += 1
            if progress:
                progress.emit(counts["processed_frames"], "processing")
            accumulator.advance(frame_index)
            detections = detector.detect(frame)
            selected_target = select_target_detection(
                detections, video_info.width, video_info.height
            )
            if selected_target is None:
                counts["frames_without_detection"] += 1
            else:
                counts["frames_with_detection"] += 1
                selected_confidences.append(selected_target.confidence)
            if len(detections) > 1:
                counts["multiple_face_frames"] += 1

            decision = lifecycle.step(
                frame_index=frame_index,
                detection_count=len(detections),
                canonical_sample_count=accumulator.total_accepted,
                hard_cut=frame_index in hard_cut_frames,
            )
            target = selected_target if decision.accept_region_gaze else None

            frame_time = frame_index / video_info.fps

            # Nearest matching is display-only. It never reaches the accumulator.
            display_row = _nearest_timed_row(
                display_valid_rows, frame_time, args.max_time_distance
            )
            display_point = None
            display_source_text = "display source=n/a"
            if display_row is None:
                nearest_raw = _nearest_timed_row(
                    timed_rows, frame_time, args.max_time_distance
                )
                if nearest_raw is None:
                    counts["display_unmatched_frames"] += 1
                    display_status = "no_display_sample"
                else:
                    counts["display_frame_matches"] += 1
                    counts["display_invalid_frame_matches"] += 1
                    display_status = f"display_invalid:{nearest_raw['invalid_reason']}"
                    display_source_text = (
                        f"display row={int(nearest_raw['frame_index'])} "
                        f"t={float(nearest_raw['time_seconds']):.3f}s"
                    )
            else:
                counts["display_frame_matches"] += 1
                counts["display_valid_frame_matches"] += 1
                display_status = "display_valid"
                display_point = (
                    float(display_row["gaze_x"]),
                    float(display_row["gaze_y"]),
                )
                display_source_text = (
                    f"display row={int(display_row['frame_index'])} "
                    f"t={float(display_row['time_seconds']):.3f}s"
                )

            assigned_here = assignment_plan.by_frame.get(frame_index, ())
            assigned_statuses: list[str] = []
            accepted_debug: list[tuple[SourceGazeSample, tuple[float, float], tuple[float, float]]] = []
            for assignment in assigned_here:
                sample = assignment.sample
                consumption_counts[sample.source_row_id] += 1
                counts["assigned_unique_source_rows"] += 1
                screen_point = (sample.gaze_x, sample.gaze_y)
                relative = None
                draw_back = None
                if target is None:
                    if len(detections) > 1:
                        classification = "identity_uncertain"
                        counts["identity_uncertain_unique_samples"] += 1
                    else:
                        classification = "no_detection"
                        counts["no_detection_unique_samples"] += 1
                else:
                    relative = screen_to_region(screen_point, target)
                    if relative is None:
                        classification = "outside_face"
                        counts["rejected_outside_face_unique_samples"] += 1
                    else:
                        accumulator.add(relative, frame_index=frame_index)
                        draw_back = region_to_screen(relative, target)
                        classification = "accepted_inside_face"
                        counts["accepted_inside_face_unique_samples"] += 1
                        accepted_debug.append((sample, relative, draw_back))
                assigned_statuses.append(classification)
                audit_records.append(
                    {
                        "source_row_id": sample.source_row_id,
                        "sample_time_seconds": sample.time_seconds,
                        "assigned_frame": assignment.frame_index,
                        "assigned_frame_time_seconds": assignment.frame_time_seconds,
                        "classification": classification,
                        "invalid_reason": None,
                        "screen_x": sample.gaze_x,
                        "screen_y": sample.gaze_y,
                        "target_box": (
                            None
                            if target is None
                            else [target.x, target.y, target.width, target.height]
                        ),
                        "local_u": None if relative is None else relative[0],
                        "local_v": None if relative is None else relative[1],
                        "draw_back_x": None if draw_back is None else draw_back[0],
                        "draw_back_y": None if draw_back is None else draw_back[1],
                        "lifecycle_phase": decision.phase.value,
                        "lifecycle_reason": decision.reason,
                        "consumption_count": consumption_counts[sample.source_row_id],
                    }
                )

            lifecycle.record_post_frame_sample_count(accumulator.total_accepted)

            if decision.render_current_overlay:
                assert target is not None
                dynamic = overlay_region_heatmap(
                    frame, accumulator.canvas, target, opacity=args.opacity,
                    gradient=args.gradient, heat_scale=heat_scale,
                )
                last_valid_dynamic = dynamic.copy()
                last_valid_frame_index = frame_index
            elif decision.render_frozen_frame and last_valid_dynamic is not None:
                dynamic = last_valid_dynamic.copy()
            else:
                dynamic = frame.copy()
            debug = dynamic.copy()
            if target is not None:
                top_left, bottom_right = _box_points(target)
                cv2.rectangle(debug, top_left, bottom_right, (0, 220, 0), 2)
            if display_point is not None:
                cv2.circle(
                    debug,
                    (int(round(display_point[0])), int(round(display_point[1]))),
                    7,
                    (0, 0, 255),
                    -1,
                )
            for _, _, draw_back in accepted_debug:
                cv2.circle(
                    debug,
                    (int(round(draw_back[0])), int(round(draw_back[1]))),
                    6,
                    (0, 255, 0),
                    2,
                )
            last_accepted = accepted_debug[-1] if accepted_debug else None
            local_text = (
                f"assigned row={last_accepted[0].source_row_id} "
                f"u,v=({last_accepted[1][0]:.3f},{last_accepted[1][1]:.3f})"
                if last_accepted is not None
                else "assigned local=n/a"
            )
            freeze_text = (
                f"frozen source frame={last_valid_frame_index}"
                if decision.render_frozen_frame
                else "frozen source frame=n/a"
            )
            _write_status(
                cv2,
                debug,
                [
                    (
                        f"frame={frame_index} tracking={decision.phase.value} "
                        f"display={display_status} assigned={len(assigned_here)}"
                    ),
                    f"reason={decision.reason}",
                    display_source_text,
                    local_text,
                    f"assigned classes={','.join(assigned_statuses) or 'none'}",
                    f"unique accepted={accumulator.total_accepted} active={accumulator.active_sample_count}",
                    f"heat scale=current_max_auto; purpose={args.evidence_purpose}",
                    freeze_text,
                ],
            )
            if args.evidence_purpose == "alignment_review":
                if frame_index in pending_review_after:
                    cv2.imwrite(str(review_dir / f"frame_{frame_index:06d}_after_raw.png"), frame)
                    pending_review_after.remove(frame_index)
                if accepted_debug:
                    if previous_raw_frame is not None:
                        cv2.imwrite(
                            str(review_dir / f"frame_{frame_index - 1:06d}_before_raw.png"),
                            previous_raw_frame,
                        )
                    cv2.imwrite(str(review_dir / f"frame_{frame_index:06d}_accepted_debug.png"), debug)
                    pending_review_after.add(frame_index + 1)
            if (frame_index - selected_range.start_frame) % args.frame_skip == 0:
                dynamic_writer.write(dynamic)
                debug_writer.write(debug)
            previous_raw_frame = frame.copy()
            frame_index += 1
    finally:
        cap.release()
        dynamic_writer.release()
        debug_writer.release()

    if progress and counts["processed_frames"] == progress.total:
        progress.emit(counts["processed_frames"], "finalising")

    if counts["processed_frames"] == 0:
        raise SystemExit("No video frames were processed.")
    actual_end_frame_exclusive = selected_range.start_frame + counts["processed_frames"]
    assigned_ids = {
        int(record["source_row_id"])
        for record in audit_records
        if record["classification"] not in {"invalid_source_row", "unassigned_video_decode_ended"}
    }
    for assignment in assignment_plan.assignments:
        sample = assignment.sample
        if sample.source_row_id in assigned_ids:
            continue
        counts["unassigned_selected_valid_source_rows"] += 1
        audit_records.append(
            {
                "source_row_id": sample.source_row_id,
                "sample_time_seconds": sample.time_seconds,
                "assigned_frame": assignment.frame_index,
                "assigned_frame_time_seconds": assignment.frame_time_seconds,
                "classification": "unassigned_video_decode_ended",
                "invalid_reason": None,
                "screen_x": sample.gaze_x,
                "screen_y": sample.gaze_y,
                "target_box": None,
                "local_u": None,
                "local_v": None,
                "draw_back_x": None,
                "draw_back_y": None,
                "lifecycle_phase": None,
                "lifecycle_reason": None,
                "consumption_count": 0,
            }
        )
    duplicate_check = duplicate_consumption_summary(consumption_counts)
    audit_records.sort(key=lambda record: int(record["source_row_id"]))
    audit_json_path.write_text(
        json.dumps(
            {
                "schema_version": ASSIGNMENT_SCHEMA_VERSION,
                "records": audit_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    audit_fields = [
        "source_row_id",
        "sample_time_seconds",
        "assigned_frame",
        "assigned_frame_time_seconds",
        "classification",
        "invalid_reason",
        "screen_x",
        "screen_y",
        "target_box",
        "local_u",
        "local_v",
        "draw_back_x",
        "draw_back_y",
        "lifecycle_phase",
        "lifecycle_reason",
        "consumption_count",
    ]
    with audit_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        for record in audit_records:
            csv_record = dict(record)
            if csv_record["target_box"] is not None:
                csv_record["target_box"] = json.dumps(csv_record["target_box"])
            writer.writerow(csv_record)
    audit_consistency = audit_consistency_summary(
        audit_records, canonical_accepted_count=accumulator.total_accepted
    )
    if not audit_consistency["static_dynamic_canonical_audit_source_set_passed"]:
        raise RuntimeError("Gaze assignment audit/canonical consistency check failed")
    manual_review_path = None
    if args.evidence_purpose == "alignment_review":
        manual_review_path = out_dir / "manual_alignment_review.csv"
        review_fields = [
            "source_row_id",
            "sample_time_seconds",
            "assigned_frame",
            "assigned_frame_time_seconds",
            "screen_point",
            "local_point",
            "draw_back_point",
            "implementation_round_trip_observation",
            "approximate_reference_observation",
            "reference_marker_present",
            "reviewer_notes",
            "reference_sync_status",
        ]
        with manual_review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=review_fields)
            writer.writeheader()
            for record in audit_records:
                if record["classification"] != "accepted_inside_face":
                    continue
                writer.writerow(
                    {
                        "source_row_id": record["source_row_id"],
                        "sample_time_seconds": record["sample_time_seconds"],
                        "assigned_frame": record["assigned_frame"],
                        "assigned_frame_time_seconds": record["assigned_frame_time_seconds"],
                        "screen_point": json.dumps([record["screen_x"], record["screen_y"]]),
                        "local_point": json.dumps([record["local_u"], record["local_v"]]),
                        "draw_back_point": json.dumps([record["draw_back_x"], record["draw_back_y"]]),
                        "implementation_round_trip_observation": "pending_manual_review",
                        "approximate_reference_observation": "pending_manual_review",
                        "reference_marker_present": "pending_manual_review",
                        "reviewer_notes": "",
                        "reference_sync_status": "approximate_only_not_ground_truth",
                    }
                )
    static_uint8, resolved_maximum = scale_heatmap(accumulator.canvas, heat_scale)
    static_colour = colorize(static_uint8, args.gradient)
    static_colour[static_uint8 == 0] = 0
    if not cv2.imwrite(str(static_path), static_colour):
        raise SystemExit(f"Could not write static heatmap: {static_path}")

    valid_denominator = counts["assigned_unique_source_rows"]
    detection_denominator = counts["processed_frames"]
    runtime_seconds = perf_counter() - started_at
    classified_valid = (
        counts["accepted_inside_face_unique_samples"]
        + counts["rejected_outside_face_unique_samples"]
        + counts["no_detection_unique_samples"]
        + counts["identity_uncertain_unique_samples"]
    )
    summary = {
        "method": "region-relative heatmap; face first provider",
        "region_core": "generic normalized rectangular RegionDetection coordinates",
        "input": {"gaze": str(Path(args.gaze)), "video": str(Path(args.video))},
        "outputs": {
            "static_canonical_heatmap": str(static_path),
            "dynamic_overlay": str(dynamic_path),
            "debug_overlay": str(debug_path),
            "assignment_audit_csv": str(audit_csv_path),
            "assignment_audit_json": str(audit_json_path),
            "manual_alignment_review": (
                None if manual_review_path is None else str(manual_review_path)
            ),
        },
        "selected_range": {
            **selected_range.as_dict(),
            "actual_end_frame_exclusive": actual_end_frame_exclusive,
        },
        "video_info": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "frame_count": video_info.frame_count,
        },
        "gaze_summary": summarize_gaze(gaze),
        "accumulation_sample_assignment": {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "method": (
                "each selected valid TXT source row is assigned once to the first video "
                "frame whose timestamp is not earlier than rec_time"
            ),
            "source_identity": "TXT source row position; equal time/coordinates are not merged",
            "maximum_samples_assigned_to_one_frame": assignment_plan.maximum_samples_per_frame,
            "duplicate_consumption_check": duplicate_check,
            "audit_consistency_check": audit_consistency,
        },
        "display_frame_matching": {
            "method": "nearest timed TXT row within tolerance; display only",
            "maximum_time_distance_seconds": args.max_time_distance,
            "drives_canonical_accumulation": False,
        },
        "target_selection_policy": (
            "The identity gate requires exactly one detected face. The largest-area "
            "policy is used only for detection statistics; multiple faces are treated "
            "as identity-uncertain and never merged into canonical memory."
        ),
        "identity_policy": TrackingLifecycle.IDENTITY_POLICY,
        "parameters": {
            "detector": args.detector_provider,
            "model": str(Path(args.model)),
            "config_path": None if args.detector_config is None else str(Path(args.detector_config)),
            "confidence_threshold": args.confidence_threshold,
            "min_size": args.min_size,
            "nms_threshold": args.nms_threshold if args.detector_provider == "opencv-yunet" else None,
            "canvas_width": args.canvas_width,
            "canvas_height": args.canvas_height,
            "brush_radius": args.brush_radius,
            "opacity": args.opacity,
            "frame_skip": args.frame_skip,
            "output_update_interval_semantics": "write_every_nth_output_frame_only; all frames drive source-row assignment and the tracking lifecycle",
            "rendering": rendering_summary(args.gradient, heat_scale, resolved_maximum=resolved_maximum),
            "history_mode": args.history_mode,
            "window_seconds": args.window_seconds if args.history_mode == "window" else None,
            "window_frames": window_frames,
            "max_freeze_seconds_requested": args.max_freeze_seconds,
            "max_freeze_frames": max_freeze_frames,
            "max_freeze_seconds_effective": max_freeze_frames / video_info.fps,
            "manual_confirmed_hard_cut_frames": sorted(hard_cut_frames),
            "evidence_purpose": args.evidence_purpose,
        },
        "counts": counts,
        "unassigned_unique_source_rows_by_reason": {
            "selected_range_video_decode_ended": counts["unassigned_selected_valid_source_rows"],
            "valid_rows_without_rec_time_not_selectable": int(
                (gaze.dataframe["valid"] & ~gaze.dataframe["time_valid"]).sum()
            ),
        },
        "tracking_lifecycle": lifecycle.summary(fps=video_info.fps),
        "coverage": {
            "face_coverage_ratio": (
                counts["accepted_inside_face_unique_samples"] / valid_denominator
                if valid_denominator
                else None
            ),
            "face_coverage_numerator": counts["accepted_inside_face_unique_samples"],
            "face_coverage_denominator": valid_denominator,
            "denominator_definition": (
                "unique valid TXT source rows assigned to the selected video interval; "
                "includes inside-face, outside-face, no-detection and identity-uncertain"
            ),
            "valid_gaze_partition_check": classified_valid == valid_denominator,
            "detection_availability_ratio": (
                counts["frames_with_detection"] / detection_denominator
                if detection_denominator
                else None
            ),
            "detection_availability_denominator": detection_denominator,
        },
        "selected_face_confidence": {
            "mean": statistics.fmean(selected_confidences)
            if selected_confidences
            else None,
            "median": statistics.median(selected_confidences)
            if selected_confidences
            else None,
        },
        "runtime_seconds": runtime_seconds,
        "processing_fps": counts["processed_frames"] / runtime_seconds,
        "heat_normalization": {
            "mode": "current_max_auto",
            "final_canvas_max": float(np.max(accumulator.canvas)),
            "sample_weight": 1.0,
            "warning": (
                "Auto-normalization can make very small sample counts look visually strong; "
                "use exact unique accepted count for interpretation."
            ),
        },
        "limitations": [
            "Outside-face gaze remains valid baseline gaze but is not assigned to another region.",
            "The face box is not a facial-landmark coordinate system.",
            (
                "Reacquisition reuses canonical memory under the declared single-"
                "conversation-partner assumption; it is not biometric identity tracking."
            ),
            (
                "Multiple detected faces are treated as identity-uncertain loss and are "
                "not silently merged."
            ),
            (
                "The default 0.5-second freeze is an initial configurable engineering "
                "choice, not a scientifically validated threshold."
            ),
            "The lifecycle remains detector-only and does not use optical flow to bridge loss.",
            (
                "Original Eye Tracking reference timing/offset is not strictly synchronized; "
                "reference observations are approximate manual evidence, not accuracy ground truth."
            ),
            (
                "Lifecycle-stress heatmaps are not attention-distribution or accuracy evidence."
                if args.evidence_purpose == "lifecycle_stress"
                else "Implementation round-trip consistency does not prove input gaze accuracy."
            ),
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
