#!/usr/bin/env python3
"""Aggregate baseline, face-relative, optical-flow, and lifecycle results without running core algorithms."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


FACE_CLASSES = (
    "accepted_inside_face",
    "outside_face",
    "no_detection",
    "identity_uncertain",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def interval_from_baseline(summary: dict[str, Any]) -> tuple[int, int]:
    selected = summary["selected_range"]
    return int(selected["start_frame"]), int(selected["end_frame_exclusive"])


def interval_from_face(summary: dict[str, Any]) -> tuple[int, int]:
    selected = summary["selected_range"]
    return int(selected["start_frame"]), int(selected["end_frame_exclusive"])


def interval_from_flow(summary: dict[str, Any]) -> tuple[int, int]:
    selected = summary["selected_range"]
    end = selected.get("end_frame_exclusive", selected.get("end_frame"))
    return int(selected["start_frame"]), int(end)


def accepted_round_trip_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if record.get("classification") != "accepted_inside_face":
            continue
        required = ("screen_x", "screen_y", "draw_back_x", "draw_back_y")
        if any(record.get(key) is None for key in required):
            raise ValueError(f"accepted row lacks round-trip coordinates: {record.get('source_row_id')}")
        dx = float(record["draw_back_x"]) - float(record["screen_x"])
        dy = float(record["draw_back_y"]) - float(record["screen_y"])
        output.append({
            "source_row_id": int(record["source_row_id"]),
            "assigned_frame": int(record["assigned_frame"]),
            "error_pixels": math.hypot(dx, dy),
        })
    return output


def validate_face(summary: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    assignment = summary.get("accumulation_sample_assignment", {})
    if assignment.get("schema_version") != "gaze_assignment.v1":
        raise ValueError("face summary does not use the required unique-source-row assignment schema")
    duplicate = assignment.get("duplicate_consumption_check", {})
    consistency = assignment.get("audit_consistency_check", {})
    if not duplicate.get("passed") or duplicate.get("duplicate_consumption_excess") != 0:
        raise ValueError("duplicate-consumption check failed")
    if not consistency.get("coverage_partition_passed"):
        raise ValueError("face-assignment coverage partition failed")
    if not consistency.get("static_dynamic_canonical_audit_source_set_passed"):
        raise ValueError("face static/dynamic/canonical/audit source sets differ")

    counts = summary["counts"]
    selected = int(counts["selected_valid_source_rows"])
    partition = sum(int(counts[key]) for key in (
        "accepted_inside_face_unique_samples",
        "rejected_outside_face_unique_samples",
        "no_detection_unique_samples",
        "identity_uncertain_unique_samples",
    ))
    if partition != selected:
        raise ValueError(f"face partition {partition} != selected valid {selected}")

    records = audit.get("records")
    if not isinstance(records, list):
        raise ValueError("face audit lacks records list")
    common = [row for row in records if row.get("classification") in FACE_CLASSES]
    ids = [int(row["source_row_id"]) for row in common]
    if len(common) != selected or len(ids) != len(set(ids)):
        raise ValueError("common source-row set is incomplete or non-unique")
    if any(int(row.get("consumption_count", 0)) != 1 for row in common):
        raise ValueError("assigned source row was not consumed exactly once")

    round_trip = accepted_round_trip_rows(records)
    accepted = int(counts["accepted_inside_face_unique_samples"])
    if len(round_trip) != accepted:
        raise ValueError("accepted count differs from round-trip row count")
    errors = [row["error_pixels"] for row in round_trip]
    return {
        "common_rows": common,
        "round_trip_rows": round_trip,
        "round_trip_mean_pixels": sum(errors) / len(errors) if errors else None,
        "round_trip_max_pixels": max(errors) if errors else None,
    }


def validate_flow(summary: dict[str, Any]) -> None:
    if summary.get("schema_version") != "optical_flow_event_semantics.v2":
        raise ValueError("flow results must use the current event schema, not legacy shot-boundary semantics")
    if summary.get("history_mode") != "window":
        raise ValueError("formal flow comparison must use recent-window history")
    terminations = summary.get("termination_counts", {})
    expected = {"window_expired", "flow_or_bounds", "flow_tracking_lost", "confirmed_hard_cut"}
    if set(terminations) != expected:
        raise ValueError("flow termination reasons are missing or conflated")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_evaluation(
    baseline: dict[str, Any],
    face: dict[str, Any],
    audit: dict[str, Any],
    flow: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    face_check = validate_face(face, audit)
    interval = interval_from_face(face)
    if interval_from_baseline(baseline) != interval:
        raise ValueError("baseline and face intervals differ")
    counts = face["counts"]
    selected = int(counts["selected_valid_source_rows"])
    baseline_selected = int(baseline.get("selected_valid_gaze_samples", baseline.get("added_gaze_samples")))
    if baseline_selected != selected:
        raise ValueError("baseline and face valid source-row counts differ")

    if flow is not None:
        validate_flow(flow)
        if interval_from_flow(flow) != interval:
            raise ValueError("flow and primary comparison intervals differ")
        if int(flow["selected_gaze_count"]) != selected:
            raise ValueError("flow and primary selected gaze counts differ")

    common_rows = [{
        "source_row_id": int(row["source_row_id"]),
        "sample_time_seconds": row["sample_time_seconds"],
        "assigned_frame": row["assigned_frame"],
        "baseline_included": True,
        "face_classification": row["classification"],
        "face_accepted": row["classification"] == "accepted_inside_face",
    } for row in face_check["common_rows"]]

    method_rows = [{
        "method": "screen_space_baseline",
        "history_mode": "cumulative",
        "selected_valid_unique_rows": selected,
        "included_or_added_rows": baseline_selected,
        "coverage_ratio": 1.0,
        "runtime_seconds": baseline.get("runtime_seconds"),
        "processing_fps": None,
    }, {
        "method": "face_relative_detector",
        "history_mode": face["parameters"]["history_mode"],
        "selected_valid_unique_rows": selected,
        "included_or_added_rows": int(counts["accepted_inside_face_unique_samples"]),
        "coverage_ratio": face["coverage"]["face_coverage_ratio"],
        "runtime_seconds": face.get("runtime_seconds"),
        "processing_fps": face.get("processing_fps"),
    }]
    if flow is not None:
        method_rows.append({
            "method": "optical_flow_experimental",
            "history_mode": f"window_{flow['parameters']['window_seconds']}_seconds",
            "selected_valid_unique_rows": int(flow["selected_gaze_count"]),
            "included_or_added_rows": int(flow["added_gaze_count"]),
            "coverage_ratio": None,
            "runtime_seconds": flow.get("runtime_seconds"),
            "processing_fps": flow.get("processing_fps"),
        })

    result: dict[str, Any] = {
        "schema_version": "comparative_evaluation.v1",
        "primary_segment": {"participant": "P4", "start_frame": interval[0], "end_frame_exclusive": interval[1]},
        "evidence_policy": {
            "face_source": "unique-source-row assignment results only",
            "superseded_evidence_excluded": "older frame-assignment counts",
            "heat_intensity_ranked_across_methods": False,
        },
        "common_sample_set": {
            "unique_source_row_count": selected,
            "source_row_ids_saved": True,
            "baseline_included_count": baseline_selected,
            "face_partition": {
                "accepted_inside_face": int(counts["accepted_inside_face_unique_samples"]),
                "outside_face": int(counts["rejected_outside_face_unique_samples"]),
                "no_detection": int(counts["no_detection_unique_samples"]),
                "identity_uncertain": int(counts["identity_uncertain_unique_samples"]),
            },
            "face_coverage_ratio": face["coverage"]["face_coverage_ratio"],
        },
        "assignment_integrity": {
            "maximum_samples_assigned_to_one_frame": face["accumulation_sample_assignment"]["maximum_samples_assigned_to_one_frame"],
            "duplicate_consumption_check": face["accumulation_sample_assignment"]["duplicate_consumption_check"],
            "round_trip_accepted_count": len(face_check["round_trip_rows"]),
            "round_trip_mean_pixels": face_check["round_trip_mean_pixels"],
            "round_trip_max_pixels": face_check["round_trip_max_pixels"],
        },
        "face_detection_and_lifecycle_primary": {
            "frames_with_detection": counts["frames_with_detection"],
            "frames_without_detection": counts["frames_without_detection"],
            "detection_availability_ratio": face["coverage"]["detection_availability_ratio"],
            "loss_episode_count": face["tracking_lifecycle"]["loss_episode_count"],
            "reacquisition_count": face["tracking_lifecycle"]["reacquisition_count"],
            "p4_lifecycle_validation_possible": False,
        },
        "qualitative_sanity_check": {
            "reference_agreement": "approximate visual agreement only; not ground-truth accuracy",
            "accepted_rows_round_trip_consistent": len(face_check["round_trip_rows"]) == int(counts["accepted_inside_face_unique_samples"]),
            "heat_rendering": "limited to the face box and moves with the detected face",
            "large_pose_claim": "no stable facial-landmark correspondence claimed",
            "p4_lifecycle_scope": "cannot validate no-face, freeze, hide, or reacquisition",
        },
        "method_metrics": method_rows,
    }
    if flow is not None:
        result["optical_flow_experimental"] = {
            "history_mode": "recent-window",
            "window_seconds": flow["parameters"]["window_seconds"],
            "event_policy": flow["parameters"]["event_policy"],
            "event_counts": flow["event_counts"],
            "termination_counts": flow["termination_counts"],
            "technical_segment_count": flow["technical_segment_count"],
            "flow_interval_count": flow["flow_interval_count"],
            "loss_episode_count": flow["loss_episode_count"],
            "forward_backward_errors": flow["forward_backward_errors"],
        }
    if lifecycle is not None:
        lc_counts = lifecycle["counts"]
        lc = lifecycle["tracking_lifecycle"]
        result["supplementary_lifecycle_stress"] = {
            "label": "supplementary lifecycle stress experiment; threshold 0.94; not attention or accuracy evidence",
            "participant": "P2",
            "interval": interval_from_face(lifecycle),
            "accepted_unique_rows": lc_counts["accepted_inside_face_unique_samples"],
            "assigned_unique_rows": lc_counts["assigned_unique_source_rows"],
            "frames_with_detection": lc_counts["frames_with_detection"],
            "frames_without_detection": lc_counts["frames_without_detection"],
            "loss_episode_count": lc["loss_episode_count"],
            "reacquisition_count": lc["reacquisition_count"],
            "freeze_visible_frames": lc["freeze_visible_frames"],
            "hidden_retained_frames": lc["hidden_retained_frames"],
        }
    return result, method_rows, common_rows, face_check["round_trip_rows"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--face-summary", type=Path, required=True)
    parser.add_argument("--face-audit", type=Path, required=True)
    parser.add_argument("--flow-summary", type=Path)
    parser.add_argument("--lifecycle-summary", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        "baseline_summary": args.baseline_summary,
        "face_summary": args.face_summary,
        "face_audit": args.face_audit,
        "flow_summary": args.flow_summary,
        "lifecycle_summary": args.lifecycle_summary,
    }
    result, methods, common, round_trip = build_evaluation(
        load_json(args.baseline_summary),
        load_json(args.face_summary),
        load_json(args.face_audit),
        load_json(args.flow_summary) if args.flow_summary else None,
        load_json(args.lifecycle_summary) if args.lifecycle_summary else None,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result["source_evidence"] = {key: str(value) if value else None for key, value in sources.items()}
    with (args.out_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(args.out_dir / "method_metrics.csv", [
        "method", "history_mode", "selected_valid_unique_rows", "included_or_added_rows",
        "coverage_ratio", "runtime_seconds", "processing_fps",
    ], methods)
    write_csv(args.out_dir / "common_source_rows.csv", [
        "source_row_id", "sample_time_seconds", "assigned_frame", "baseline_included",
        "face_classification", "face_accepted",
    ], common)
    write_csv(args.out_dir / "accepted_round_trip_errors.csv", [
        "source_row_id", "assigned_frame", "error_pixels",
    ], round_trip)
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "common_unique_rows": len(common),
        "accepted_round_trip_rows": len(round_trip),
        "round_trip_mean_pixels": result["assignment_integrity"]["round_trip_mean_pixels"],
        "round_trip_max_pixels": result["assignment_integrity"]["round_trip_max_pixels"],
    }, indent=2))


if __name__ == "__main__":
    main()
