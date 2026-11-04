from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.detector_registry import validate_face_detector_config
from video_heatmap.detectors import create_face_detector
from video_heatmap.regions import RegionDetection, select_target_detection


def _detection_record(detection: RegionDetection) -> dict[str, object]:
    return {
        "x": detection.x,
        "y": detection.y,
        "width": detection.width,
        "height": detection.height,
        "confidence": detection.confidence,
        "label": detection.label,
        "track_id": detection.track_id,
    }


def summarise_records(records: list[dict[str, object]]) -> dict[str, object]:
    counts = [len(record["detections"]) for record in records]
    confidences = [
        float(detection["confidence"])
        for record in records
        for detection in record["detections"]
    ]
    runtime = sum(float(record["runtime_seconds"]) for record in records)
    diagnostics: Counter[str] = Counter()
    for record in records:
        diagnostics.update(record.get("diagnostics", {}))
    processed = len(records)
    available = sum(count > 0 for count in counts)
    return {
        "processed_frames": processed,
        "zero_face_frames": sum(count == 0 for count in counts),
        "one_face_frames": sum(count == 1 for count in counts),
        "multiple_face_frames": sum(count > 1 for count in counts),
        "detection_available_frames": available,
        "detection_availability": available / processed if processed else 0.0,
        "detection_count": sum(counts),
        "confidence": {
            "minimum": min(confidences) if confidences else None,
            "mean": statistics.fmean(confidences) if confidences else None,
            "median": statistics.median(confidences) if confidences else None,
            "maximum": max(confidences) if confidences else None,
        },
        "detector_runtime_seconds": runtime,
        "detector_processing_fps": processed / runtime if runtime else None,
        "diagnostic_totals": dict(sorted(diagnostics.items())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    *,
    cv2,
    participant: str,
    video_path: Path,
    detector_id: str,
    config: dict[str, object],
    start_frame: int,
    max_frames: int,
    out_dir: Path,
    save_indices: set[int],
) -> dict[str, object]:
    detector = create_face_detector(detector_id, **config)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = out_dir / "frames"
    evidence_dir.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    first_zero_saved = False
    first_multiple_saved = False
    wall_started = perf_counter()
    try:
        for offset in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = start_frame + offset
            started = perf_counter()
            detections = detector.detect(frame)
            detector_runtime = perf_counter() - started
            target = select_target_detection(detections, width, height)
            diagnostics = dict(getattr(detector, "last_diagnostics", {}))
            record = {
                "participant": participant,
                "provider_id": detector_id,
                "frame_index": frame_index,
                "time_seconds": frame_index / fps if fps else None,
                "runtime_seconds": detector_runtime,
                "detection_count": len(detections),
                "detection_available": bool(detections),
                "detections": [_detection_record(item) for item in detections],
                "selected_target": _detection_record(target) if target else None,
                "diagnostics": diagnostics,
            }
            records.append(record)

            reason = None
            if offset in save_indices:
                reason = "fixed"
            elif not detections and not first_zero_saved:
                reason = "first_zero"
                first_zero_saved = True
            elif len(detections) > 1 and not first_multiple_saved:
                reason = "first_multiple"
                first_multiple_saved = True
            if reason:
                for detection in detections:
                    colour = (0, 220, 0) if detection is target else (0, 165, 255)
                    p0 = (int(round(detection.x)), int(round(detection.y)))
                    p1 = (
                        int(round(detection.x + detection.width)),
                        int(round(detection.y + detection.height)),
                    )
                    cv2.rectangle(frame, p0, p1, colour, 2)
                cv2.putText(
                    frame,
                    f"{detector_id} frame={frame_index} detections={len(detections)}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(evidence_dir / f"frame_{frame_index:06d}_{reason}.jpg"), frame)
    finally:
        cap.release()
    wall_runtime = perf_counter() - wall_started

    raw_path = out_dir / "raw_records.jsonl"
    raw_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema": "face_detector_provider_evidence.v1",
        "participant": participant,
        "provider_id": detector_id,
        "provider_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.items()
        },
        "coordinate_system": "original video pixels; origin at top-left",
        "target_selection_policy": (
            "largest face area; ties by frame-centre distance, confidence, x, y"
        ),
        "input_video": str(video_path),
        "source_video": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": source_frames,
        },
        "selected_range": {
            "start_frame": start_frame,
            "end_frame_exclusive": start_frame + len(records),
        },
        **summarise_records(records),
        "wall_runtime_seconds": wall_runtime,
        "wall_processing_fps": len(records) / wall_runtime if wall_runtime else None,
        "raw_records": str(raw_path),
        "evidence_frames": sorted(str(path) for path in evidence_dir.glob("*.jpg")),
        "claim_boundary": (
            "Availability, runtime, confidence-score distribution and visual-observation "
            "evidence only; no manual ground truth, precision, recall or accuracy claim."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _video_assignment(value: str) -> tuple[str, Path]:
    participant, separator, path = value.partition("=")
    if not separator or not participant or not path:
        raise argparse.ArgumentTypeError("video must use PARTICIPANT=PATH")
    return participant, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched raw face-provider evidence.")
    parser.add_argument("--video", action="append", type=_video_assignment, required=True)
    parser.add_argument("--yunet-model", type=Path, required=True)
    parser.add_argument("--ssd-model", type=Path, required=True)
    parser.add_argument("--ssd-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    if args.start_frame < 0 or args.max_frames <= 0:
        raise SystemExit("frame range must be non-negative and non-empty")
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV is required") from exc

    configs = {
        "opencv-yunet": validate_face_detector_config(
            "opencv-yunet", {"model_path": args.yunet_model}
        ),
        "opencv-dnn-ssd": validate_face_detector_config(
            "opencv-dnn-ssd",
            {"model_path": args.ssd_model, "config_path": args.ssd_config},
        ),
    }
    summaries = []
    for participant, video_path in args.video:
        for detector_id, config in configs.items():
            print(f"Running {participant} {detector_id} [{args.start_frame}, {args.start_frame + args.max_frames})")
            summaries.append(
                _run(
                    cv2=cv2,
                    participant=participant,
                    video_path=video_path,
                    detector_id=detector_id,
                    config=config,
                    start_frame=args.start_frame,
                    max_frames=args.max_frames,
                    out_dir=args.out_dir / participant / detector_id,
                    save_indices={0, args.max_frames // 2, args.max_frames - 1},
                )
            )

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "logical_processor_count": os.cpu_count(),
        "opencv_cpu_count": cv2.getNumberOfCPUs(),
        "opencv_thread_count": cv2.getNumThreads(),
        "execution": (
            "CPU; OpenCV backend selected for opencv-dnn-ssd; "
            "OpenCV 5 new graph engine is CPU-only"
        ),
        "assets": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (args.yunet_model, args.ssd_model, args.ssd_config)
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (args.out_dir / "matched_summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
