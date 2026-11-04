from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "detector_model_evidence.v1"
PROTOCOL_ID = "detector-provider-comparison-v1-2026-08-29"
EXPECTED_PROVIDERS = ("opencv-yunet", "opencv-dnn-ssd")
EXPECTED_PARTICIPANTS = ("P2", "P3", "P4", "P5")


class EvaluationError(ValueError):
    pass


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise EvaluationError("ratio denominator must be positive")
    if numerator < 0 or numerator > denominator:
        raise EvaluationError("ratio numerator must be within [0, denominator]")
    return numerator / denominator


def per_1000_frames(seconds: float, processed_frames: int) -> float:
    if seconds < 0:
        raise EvaluationError("runtime must be non-negative")
    if processed_frames <= 0:
        raise EvaluationError("per-1000-frame denominator must be positive")
    return seconds * 1000.0 / processed_frames


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise EvaluationError(f"missing evidence file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_summary(summary: Mapping[str, object]) -> dict[str, object]:
    required = {
        "participant",
        "provider_id",
        "provider_config",
        "selected_range",
        "processed_frames",
        "zero_face_frames",
        "one_face_frames",
        "multiple_face_frames",
        "detection_available_frames",
        "detection_count",
        "detector_runtime_seconds",
        "wall_runtime_seconds",
        "diagnostic_totals",
        "raw_records",
    }
    missing = sorted(required - summary.keys())
    if missing:
        raise EvaluationError(f"summary missing fields: {', '.join(missing)}")

    processed = int(summary["processed_frames"])
    zero = int(summary["zero_face_frames"])
    one = int(summary["one_face_frames"])
    multiple = int(summary["multiple_face_frames"])
    available = int(summary["detection_available_frames"])
    selected_range = dict(summary["selected_range"])  # type: ignore[arg-type]
    expected_frames = int(selected_range["end_frame_exclusive"]) - int(
        selected_range["start_frame"]
    )
    if processed <= 0:
        raise EvaluationError("zero-frame runs are not valid measured evidence")
    if zero + one + multiple != processed:
        raise EvaluationError("zero/one/multiple partition does not equal processed frames")
    if available != one + multiple:
        raise EvaluationError("availability count does not equal one + multiple frames")
    if processed != expected_frames:
        raise EvaluationError("incomplete run cannot be labelled measured/completed")
    return dict(summary)


def load_matched_summaries(
    path: Path,
    *,
    participants: Sequence[str],
    providers: Sequence[str],
    expected_range: tuple[int, int],
) -> list[dict[str, object]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise EvaluationError("matched summary payload must be a list")
    summaries = [_validate_summary(item) for item in payload]
    keys = [(str(item["participant"]), str(item["provider_id"])) for item in summaries]
    expected_keys = [(p, provider) for p in participants for provider in providers]
    if sorted(keys) != sorted(expected_keys):
        raise EvaluationError(
            f"provider/participant set mismatch: got {sorted(keys)}, expected {sorted(expected_keys)}"
        )
    for item in summaries:
        selected = item["selected_range"]
        actual = (int(selected["start_frame"]), int(selected["end_frame_exclusive"]))  # type: ignore[index]
        if actual != expected_range:
            raise EvaluationError(f"range mismatch for {item['participant']}/{item['provider_id']}: {actual}")
    return summaries


def project_row(summary: Mapping[str, object], cohort: str) -> dict[str, object]:
    processed = int(summary["processed_frames"])
    diagnostics = summary["diagnostic_totals"]
    detector_seconds = float(summary["detector_runtime_seconds"])
    wall_seconds = float(summary["wall_runtime_seconds"])
    available = int(summary["detection_available_frames"])
    return {
        "protocol_id": PROTOCOL_ID,
        "evidence_class": "project_technical",
        "cohort": cohort,
        "participant": summary["participant"],
        "provider_id": summary["provider_id"],
        "status": "measured_completed",
        "start_frame": summary["selected_range"]["start_frame"],  # type: ignore[index]
        "end_frame_exclusive": summary["selected_range"]["end_frame_exclusive"],  # type: ignore[index]
        "processed_frames": processed,
        "zero_face_frames": int(summary["zero_face_frames"]),
        "one_face_frames": int(summary["one_face_frames"]),
        "multiple_face_frames": int(summary["multiple_face_frames"]),
        "detection_available_frames": available,
        "detection_availability_ratio": ratio(available, processed),
        "selected_box_count": available,
        "returned_detection_count": int(summary["detection_count"]),
        "invalid_rejected_boxes": int(diagnostics.get("invalid_rejected", 0)),  # type: ignore[union-attr]
        "clipped_boxes": int(diagnostics.get("clipped_boxes", 0)),  # type: ignore[union-attr]
        "min_size_or_outside_rejected_boxes": int(  # type: ignore[union-attr]
            diagnostics.get("min_size_or_outside_rejected", 0)
        ),
        "detector_runtime_seconds": detector_seconds,
        "detector_processing_fps": processed / detector_seconds if detector_seconds else None,
        "detector_seconds_per_1000_frames": per_1000_frames(detector_seconds, processed),
        "wall_runtime_seconds": wall_seconds,
        "wall_processing_fps": processed / wall_seconds if wall_seconds else None,
        "wall_seconds_per_1000_frames": per_1000_frames(wall_seconds, processed),
        "confidence_minimum": summary["confidence"]["minimum"],  # type: ignore[index]
        "confidence_mean": summary["confidence"]["mean"],  # type: ignore[index]
        "confidence_median": summary["confidence"]["median"],  # type: ignore[index]
        "confidence_maximum": summary["confidence"]["maximum"],  # type: ignore[index]
        "provider_config_json": json.dumps(summary["provider_config"], sort_keys=True),
        "raw_records": summary["raw_records"],
        "warnings": (
            "No manual ground truth; availability and confidence are not precision, recall, "
            "accuracy, or calibrated cross-provider quality."
        ),
    }


def aggregate_rows(rows: Sequence[Mapping[str, object]], cohort: str) -> list[dict[str, object]]:
    if not rows:
        raise EvaluationError("cannot aggregate missing runs")
    if any(row.get("status") != "measured_completed" for row in rows):
        raise EvaluationError("projected/failed/not-run rows cannot enter measured aggregation")
    result = []
    for provider in EXPECTED_PROVIDERS:
        selected = [row for row in rows if row["provider_id"] == provider]
        if not selected:
            raise EvaluationError(f"missing provider in aggregate: {provider}")
        processed = sum(int(row["processed_frames"]) for row in selected)
        detector_seconds = sum(float(row["detector_runtime_seconds"]) for row in selected)
        wall_seconds = sum(float(row["wall_runtime_seconds"]) for row in selected)
        available = sum(int(row["detection_available_frames"]) for row in selected)
        result.append(
            {
                "protocol_id": PROTOCOL_ID,
                "evidence_class": "project_technical",
                "cohort": cohort,
                "participant": "POOLED",
                "provider_id": provider,
                "status": "measured_completed",
                "start_frame": min(int(row["start_frame"]) for row in selected),
                "end_frame_exclusive": max(int(row["end_frame_exclusive"]) for row in selected),
                "processed_frames": processed,
                "zero_face_frames": sum(int(row["zero_face_frames"]) for row in selected),
                "one_face_frames": sum(int(row["one_face_frames"]) for row in selected),
                "multiple_face_frames": sum(int(row["multiple_face_frames"]) for row in selected),
                "detection_available_frames": available,
                "detection_availability_ratio": ratio(available, processed),
                "selected_box_count": available,
                "returned_detection_count": sum(int(row["returned_detection_count"]) for row in selected),
                "invalid_rejected_boxes": sum(int(row["invalid_rejected_boxes"]) for row in selected),
                "clipped_boxes": sum(int(row["clipped_boxes"]) for row in selected),
                "min_size_or_outside_rejected_boxes": sum(
                    int(row["min_size_or_outside_rejected_boxes"]) for row in selected
                ),
                "detector_runtime_seconds": detector_seconds,
                "detector_processing_fps": processed / detector_seconds if detector_seconds else None,
                "detector_seconds_per_1000_frames": per_1000_frames(detector_seconds, processed),
                "wall_runtime_seconds": wall_seconds,
                "wall_processing_fps": processed / wall_seconds if wall_seconds else None,
                "wall_seconds_per_1000_frames": per_1000_frames(wall_seconds, processed),
                "warnings": "Pooled only across the frozen matched range and environment; no ground-truth quality claim.",
            }
        )
    return result


def public_evidence_rows() -> list[dict[str, object]]:
    accessed = "2026-08-29"
    yunet_card = "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md"
    yunet_paper = "https://doi.org/10.1007/s11633-023-1423-y"
    ssd_official = "https://github.com/opencv/opencv/blob/4.x/samples/dnn/README.md#face-detection"
    rows = []
    for split, value in (("easy", 0.8844), ("medium", 0.8656), ("hard", 0.7503)):
        rows.append({
            "evidence_class": "public_official_model_card",
            "provider_id": "opencv-yunet",
            "model_version": "OpenCV Zoo YuNet; project asset face_detection_yunet_2026may.onnx is documented as a dynamic-shape re-export of 2023mar",
            "dataset": "WIDER FACE validation",
            "metric": f"AP_{split} (OpenCV Zoo tools/eval)",
            "value": value,
            "input_size": "Model supports dynamic H/W; evaluation table does not state an input size",
            "hardware_runtime_context": "Not stated for these AP rows",
            "source_url": yunet_card,
            "accessed_date": accessed,
            "comparability": "Not directly comparable with FDDB COCO-style AP/AR or project availability",
        })
    rows.append({
        "evidence_class": "peer_reviewed_paper",
        "provider_id": "opencv-yunet",
        "model_version": "YuNet paper model (not asserted byte-identical to project ONNX asset)",
        "dataset": "WIDER FACE validation hard track",
        "metric": "single-scale mAP",
        "value": 0.811,
        "input_size": "320x320",
        "hardware_runtime_context": "Intel i7-12700K; 1.6 ms/frame reported by paper",
        "source_url": yunet_paper,
        "accessed_date": accessed,
        "comparability": "Context evidence only; model/version and input differ from the project full-resolution OpenCV 5 run",
    })
    for metric, value in (
        ("COCO AP IoU=0.50:0.95, area=all, maxDets=100", 0.408),
        ("COCO AP IoU=0.50, area=all, maxDets=100", 0.849),
        ("COCO AR IoU=0.50:0.95, area=all, maxDets=100", 0.496),
    ):
        rows.append({
            "evidence_class": "public_official_benchmark",
            "provider_id": "opencv-dnn-ssd",
            "model_version": "OpenCV face detector UINT8 TensorFlow graph",
            "dataset": "FDDB",
            "metric": metric,
            "value": value,
            "input_size": "BGR 300x300; mean (104,177,123); scale 1.0",
            "hardware_runtime_context": "No hardware/runtime result stated in the official benchmark entry",
            "source_url": ssd_official,
            "accessed_date": accessed,
            "comparability": "Not directly comparable with WIDER FACE AP/mAP or project availability",
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise EvaluationError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_outputs(evidence_root: Path, out_dir: Path) -> dict[str, object]:
    short = load_matched_summaries(
        evidence_root / "matched_0_300" / "matched_summaries.json",
        participants=EXPECTED_PARTICIPANTS,
        providers=EXPECTED_PROVIDERS,
        expected_range=(0, 300),
    )
    minute = load_matched_summaries(
        evidence_root / "matched_one_minute_P5" / "matched_summaries.json",
        participants=("P5",),
        providers=EXPECTED_PROVIDERS,
        expected_range=(0, 1800),
    )
    short_rows = [project_row(item, "P2-P5 matched [0,300)") for item in short]
    minute_rows = [project_row(item, "P5 matched [0,1800)") for item in minute]
    pooled = aggregate_rows(short_rows, "P2-P5 pooled matched [0,300)")
    public = public_evidence_rows()
    project_rows = short_rows + minute_rows + pooled
    environment = _load_json(evidence_root / "matched_0_300" / "environment.json")
    observations = _load_json(evidence_root / "visual_observations.json")
    protocol = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "frozen_on": "2026-08-29",
        "providers": list(EXPECTED_PROVIDERS),
        "primary_matched_range": {"participants": list(EXPECTED_PARTICIPANTS), "start_frame": 0, "end_frame_exclusive": 300},
        "minute_range": {"participants": ["P5"], "start_frame": 0, "end_frame_exclusive": 1800},
        "exclusions": ["review_frame_* diagnostic reruns", "unmatched ranges", "projected/not-run/failed attempts"],
        "formulas": {
            "availability": "detection_available_frames / processed_frames",
            "detection_available_frames": "one_face_frames + multiple_face_frames",
            "processing_fps": "processed_frames / measured_seconds",
            "seconds_per_1000_frames": "measured_seconds * 1000 / processed_frames",
        },
        "claim_boundary": "No manual ground truth: no project precision, recall, accuracy, calibrated confidence comparison, or universal-best claim.",
        "environment": environment,
    }
    package = {
        "schema": SCHEMA,
        "protocol": protocol,
        "public_evidence": public,
        "project_technical_evidence": project_rows,
        "visual_observations": observations,
        "recommendation": {
            "scope": "first-release provider choice for this project and frozen evidence only",
            "default": "opencv-yunet",
            "reason": "Retain the accepted compatibility/default path and its complete availability on these clips; expose opencv-dnn-ssd as the faster CPU alternative with explicit zero/multiple-output and asset-licensing caveats.",
            "not_a_claim": "Neither provider is established as universally more accurate or generally superior.",
        },
        "validation_errors": [],
    }
    _write_json(out_dir / "protocol.json", protocol)
    _write_json(out_dir / "detector_model_evidence.json", package)
    _write_json(out_dir / "public_evidence.json", public)
    _write_csv(out_dir / "public_evidence.csv", public)
    _write_csv(out_dir / "project_technical_evidence.csv", project_rows)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched detector-provider comparison results.")
    parser.add_argument("--evidence-root", type=Path, default=Path("outputs/detector_provider_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/detector_model_evaluation"))
    args = parser.parse_args()
    package = build_outputs(args.evidence_root, args.out_dir)
    print(json.dumps({"schema": package["schema"], "out_dir": str(args.out_dir), "validation_errors": []}, indent=2))


if __name__ == "__main__":
    main()
