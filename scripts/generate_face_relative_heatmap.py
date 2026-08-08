from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.detectors import create_face_detector
from video_heatmap.gaze_loader import load_gaze_txt, summarize_gaze
from video_heatmap.region_heatmap import (
    RegionHeatmapAccumulator,
    normalized_heatmap,
    overlay_region_heatmap,
    screen_to_region,
)
from video_heatmap.regions import RegionDetection, select_target_detection
from video_heatmap.time_range import resolve_selected_range
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
    parser.add_argument("--model", required=True, help="Path to a YuNet ONNX model")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--max-time-distance", type=float, default=0.05)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--min-size", type=int, default=30)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--canvas-width", type=int, default=256)
    parser.add_argument("--canvas-height", type=int, default=256)
    parser.add_argument("--brush-radius", type=int, default=12)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument(
        "--history-mode", choices=("cumulative", "window"), default="cumulative"
    )
    parser.add_argument("--window-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.max_time_distance < 0:
        raise SystemExit("--max-time-distance must not be negative")
    if args.canvas_width <= 1 or args.canvas_height <= 1:
        raise SystemExit("canonical canvas dimensions must be greater than one")
    if args.brush_radius <= 0:
        raise SystemExit("--brush-radius must be positive")
    if not 0.0 <= args.opacity <= 1.0:
        raise SystemExit("--opacity must be between 0 and 1")
    if args.history_mode == "window" and args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive for window history")

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
    valid_timed_rows = timed_rows[timed_rows["valid"]]
    if valid_timed_rows.empty:
        raise SystemExit("No valid screen-gaze samples with rec_time were found.")

    detector = create_face_detector(
        model_path=args.model,
        confidence_threshold=args.confidence_threshold,
        min_size=args.min_size,
        nms_threshold=args.nms_threshold,
    )
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dynamic_path = out_dir / "face_relative_dynamic_overlay.mp4"
    debug_path = out_dir / "face_relative_debug_overlay.mp4"
    static_path = out_dir / "face_relative_static_heatmap.png"
    summary_path = out_dir / "face_relative_summary.json"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dynamic_writer = cv2.VideoWriter(
        str(dynamic_path), fourcc, video_info.fps, (video_info.width, video_info.height)
    )
    debug_writer = cv2.VideoWriter(
        str(debug_path), fourcc, video_info.fps, (video_info.width, video_info.height)
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
        "matched_gaze_frames": 0,
        "unmatched_gaze_frames": 0,
        "valid_screen_gaze": 0,
        "invalid_screen_gaze": 0,
        "accepted_inside_face_gaze": 0,
        "rejected_outside_face_gaze": 0,
        "no_detection_gaze": 0,
    }
    selected_confidences: list[float] = []
    started_at = perf_counter()
    frame_index = selected_range.start_frame
    try:
        while frame_index < selected_range.end_frame_exclusive:
            ok, frame = cap.read()
            if not ok:
                break
            counts["processed_frames"] += 1
            accumulator.advance(frame_index)
            detections = detector.detect(frame)
            target = select_target_detection(
                detections, video_info.width, video_info.height
            )
            if target is None:
                counts["frames_without_detection"] += 1
            else:
                counts["frames_with_detection"] += 1
                selected_confidences.append(target.confidence)
            if len(detections) > 1:
                counts["multiple_face_frames"] += 1

            frame_time = frame_index / video_info.fps
            row = _nearest_timed_row(
                valid_timed_rows, frame_time, args.max_time_distance
            )
            gaze_point = None
            relative = None
            if row is None:
                nearest_raw = _nearest_timed_row(
                    timed_rows, frame_time, args.max_time_distance
                )
                if nearest_raw is None:
                    counts["unmatched_gaze_frames"] += 1
                    status = "no_gaze_sample"
                else:
                    counts["matched_gaze_frames"] += 1
                    counts["invalid_screen_gaze"] += 1
                    status = f"invalid:{nearest_raw['invalid_reason']}"
            else:
                counts["matched_gaze_frames"] += 1
                counts["valid_screen_gaze"] += 1
                gaze_point = (float(row["gaze_x"]), float(row["gaze_y"]))
                if target is None:
                    counts["no_detection_gaze"] += 1
                    status = "no_detection"
                else:
                    relative = screen_to_region(gaze_point, target)
                    if relative is None:
                        counts["rejected_outside_face_gaze"] += 1
                        status = "outside_face"
                    else:
                        accumulator.add(relative, frame_index=frame_index)
                        counts["accepted_inside_face_gaze"] += 1
                        status = "accepted"

            dynamic = frame.copy()
            if target is not None:
                dynamic = overlay_region_heatmap(
                    dynamic, accumulator.canvas, target, opacity=args.opacity
                )
            dynamic_writer.write(dynamic)

            debug = dynamic.copy()
            if target is not None:
                top_left, bottom_right = _box_points(target)
                cv2.rectangle(debug, top_left, bottom_right, (0, 220, 0), 2)
            if gaze_point is not None:
                colour = (0, 255, 0) if relative is not None else (0, 0, 255)
                cv2.circle(
                    debug,
                    (int(round(gaze_point[0])), int(round(gaze_point[1]))),
                    7,
                    colour,
                    -1,
                )
            relative_text = (
                f"u,v=({relative[0]:.3f},{relative[1]:.3f})"
                if relative is not None
                else "u,v=n/a"
            )
            _write_status(
                cv2,
                debug,
                [
                    f"frame={frame_index} status={status}",
                    relative_text,
                    f"accepted total={accumulator.total_accepted} active={accumulator.active_sample_count}",
                    "target=F1 largest-area face policy",
                ],
            )
            debug_writer.write(debug)
            frame_index += 1
    finally:
        cap.release()
        dynamic_writer.release()
        debug_writer.release()

    if counts["processed_frames"] == 0:
        raise SystemExit("No video frames were processed.")
    static_uint8 = normalized_heatmap(accumulator.canvas)
    static_colour = cv2.applyColorMap(static_uint8, cv2.COLORMAP_JET)
    static_colour[static_uint8 == 0] = 0
    if not cv2.imwrite(str(static_path), static_colour):
        raise SystemExit(f"Could not write static heatmap: {static_path}")

    valid_denominator = counts["valid_screen_gaze"]
    detection_denominator = counts["processed_frames"]
    runtime_seconds = perf_counter() - started_at
    classified_valid = (
        counts["accepted_inside_face_gaze"]
        + counts["rejected_outside_face_gaze"]
        + counts["no_detection_gaze"]
    )
    summary = {
        "method": "region-relative heatmap; face first provider",
        "region_core": "generic normalized rectangular RegionDetection coordinates",
        "input": {"gaze": str(Path(args.gaze)), "video": str(Path(args.video))},
        "outputs": {
            "static_canonical_heatmap": str(static_path),
            "dynamic_overlay": str(dynamic_path),
            "debug_overlay": str(debug_path),
        },
        "selected_range": {
            **selected_range.as_dict(),
            "actual_end_frame_exclusive": selected_range.start_frame
            + counts["processed_frames"],
        },
        "video_info": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "frame_count": video_info.frame_count,
        },
        "gaze_summary": summarize_gaze(gaze),
        "gaze_frame_assignment": {
            "method": (
                "nearest valid TXT gaze by rec_time, matching the baseline; when none "
                "is within tolerance, the nearest raw row distinguishes invalid from unmatched"
            ),
            "maximum_time_distance_seconds": args.max_time_distance,
            "unit": "at most one classified gaze result per processed video frame",
        },
        "target_selection_policy": (
            "F1 public policy: largest face area, then frame-centre distance, "
            "confidence, x and y"
        ),
        "parameters": {
            "detector": "opencv-yunet",
            "model": str(Path(args.model)),
            "confidence_threshold": args.confidence_threshold,
            "min_size": args.min_size,
            "nms_threshold": args.nms_threshold,
            "canvas_width": args.canvas_width,
            "canvas_height": args.canvas_height,
            "brush_radius": args.brush_radius,
            "opacity": args.opacity,
            "history_mode": args.history_mode,
            "window_seconds": args.window_seconds if args.history_mode == "window" else None,
            "window_frames": window_frames,
        },
        "counts": counts,
        "coverage": {
            "face_coverage_ratio": (
                counts["accepted_inside_face_gaze"] / valid_denominator
                if valid_denominator
                else None
            ),
            "face_coverage_numerator": counts["accepted_inside_face_gaze"],
            "face_coverage_denominator": valid_denominator,
            "denominator_definition": (
                "valid screen-gaze samples assigned to processed video frames by rec_time; "
                "includes inside-face, outside-face and no-detection gaze"
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
        "limitations": [
            "Outside-face gaze remains valid baseline gaze but is not assigned to another region in H1.",
            "The face box is not a facial-landmark coordinate system.",
            "Target selection is frame-local and does not establish persistent identity.",
            "Tracking-loss last-valid-frame behaviour belongs to R1 and is not implemented here.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
