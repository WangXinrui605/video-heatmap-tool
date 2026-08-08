from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.detectors import create_face_detector
from video_heatmap.regions import RegionDetection, select_target_detection


def _box_points(detection: RegionDetection) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (int(round(detection.x)), int(round(detection.y))),
        (
            int(round(detection.x + detection.width)),
            int(round(detection.y + detection.height)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a face-detection overlay and JSON detection statistics."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True, help="Path to a YuNet ONNX model")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--min-size", type=int, default=30)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    args = parser.parse_args()

    if args.start_frame < 0:
        raise SystemExit("--start-frame must not be negative")
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV is required for overlay generation") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = create_face_detector(
        model_path=args.model,
        confidence_threshold=args.confidence_threshold,
        min_size=args.min_size,
        nms_threshold=args.nms_threshold,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    output_video = out_dir / "face_detection_overlay.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create output video: {output_video}")

    started_at = perf_counter()
    processed_frames = 0
    frames_with_face = 0
    multiple_face_frames = 0
    all_confidences: list[float] = []
    selected_confidences: list[float] = []
    try:
        while processed_frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame)
            target = select_target_detection(detections, width, height)
            if detections:
                frames_with_face += 1
                all_confidences.extend(detection.confidence for detection in detections)
            if len(detections) > 1:
                multiple_face_frames += 1
            if target is not None:
                selected_confidences.append(target.confidence)

            for detection in detections:
                colour = (0, 220, 0) if detection is target else (0, 165, 255)
                top_left, bottom_right = _box_points(detection)
                cv2.rectangle(frame, top_left, bottom_right, colour, 2)
                cv2.putText(
                    frame,
                    f"face {detection.confidence:.2f}",
                    (top_left[0], max(18, top_left[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
            status = "target: largest face" if target is not None else "no face"
            cv2.putText(
                frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            processed_frames += 1
    finally:
        cap.release()
        writer.release()

    runtime_seconds = perf_counter() - started_at
    frames_without_face = processed_frames - frames_with_face
    summary = {
        "detector": "opencv-yunet",
        "detector_parameters": {
            "model": str(Path(args.model)),
            "confidence_threshold": args.confidence_threshold,
            "min_size": args.min_size,
            "nms_threshold": args.nms_threshold,
        },
        "coordinate_system": "full-resolution frame pixels; origin at top-left",
        "target_selection_policy": (
            "largest face area; ties resolved by distance to frame centre, confidence, x, y"
        ),
        "input_video": str(Path(args.video)),
        "output_video": str(output_video),
        "source_video": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": source_frame_count,
        },
        "selected_range": {
            "start_frame": args.start_frame,
            "end_frame_exclusive": args.start_frame + processed_frames,
        },
        "processed_frames": processed_frames,
        "frames_with_face": frames_with_face,
        "frames_without_face": frames_without_face,
        "multiple_face_frames": multiple_face_frames,
        "face_frame_ratio": (
            frames_with_face / processed_frames if processed_frames else 0.0
        ),
        "mean_confidence": statistics.fmean(all_confidences) if all_confidences else None,
        "median_confidence": statistics.median(all_confidences) if all_confidences else None,
        "mean_selected_confidence": (
            statistics.fmean(selected_confidences) if selected_confidences else None
        ),
        "runtime_seconds": runtime_seconds,
        "processing_fps": processed_frames / runtime_seconds if runtime_seconds else None,
    }
    summary_path = out_dir / "face_detection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
