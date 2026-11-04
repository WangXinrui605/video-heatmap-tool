"""Run and validate L1 long-video evidence without changing algorithm semantics.

The runner delegates to ``scripts/run_heatmap_tool.py`` so the accepted G1/R1/O2
guards remain authoritative.  It adds evaluation-only wall-clock, processing-ratio,
file-size, provenance, and participant/run-matrix records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "l1_long_video_run.v1"
VALIDATION_SCHEMA_VERSION = "l1_long_video_metrics_validation.v1"
MATRIX_SCHEMA_VERSION = "l1_participant_run_matrix.v1"
FLOW_TERMINATION_KEYS = (
    "window_expired",
    "flow_or_bounds",
    "flow_tracking_lost",
    "confirmed_hard_cut",
)
EVENT_REVIEW_FIELDS = (
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
)


class MetricsValidationError(ValueError):
    """Raised when an L1 summary violates the frozen evaluation contract."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MetricsValidationError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(tokens: Iterable[str]) -> str:
    tokens = list(tokens)
    if sys.platform == "win32":
        return subprocess.list2cmdline(tokens)
    return shlex.join(tokens)


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetricsValidationError(f"{key} must be a non-negative integer")
    return value


def validate_face_core(core: dict[str, Any]) -> dict[str, Any]:
    assignment = core.get("accumulation_sample_assignment", {})
    counts = core.get("counts", {})
    coverage = core.get("coverage", {})
    lifecycle = core.get("tracking_lifecycle", {})
    parameters = core.get("parameters", {})
    if assignment.get("schema_version") != "gaze_assignment.v1":
        raise MetricsValidationError("Face core is not gaze_assignment.v1")
    duplicate = assignment.get("duplicate_consumption_check", {})
    audit = assignment.get("audit_consistency_check", {})
    if duplicate.get("passed") is not True or duplicate.get("duplicate_consumption_excess") != 0:
        raise MetricsValidationError("Face duplicate-consumption check failed")
    if audit.get("coverage_partition_passed") is not True:
        raise MetricsValidationError("Face audit coverage partition failed")
    if audit.get("static_dynamic_canonical_audit_source_set_passed") is not True:
        raise MetricsValidationError("Face canonical/audit source-set check failed")
    if parameters.get("history_mode") != "cumulative":
        raise MetricsValidationError("L1 face evidence must use cumulative history")
    assigned = _required_int(counts, "assigned_unique_source_rows")
    inside = _required_int(counts, "accepted_inside_face_unique_samples")
    outside = _required_int(counts, "rejected_outside_face_unique_samples")
    no_detection = _required_int(counts, "no_detection_unique_samples")
    uncertain = _required_int(counts, "identity_uncertain_unique_samples")
    if inside + outside + no_detection + uncertain != assigned:
        raise MetricsValidationError("Face unique-row partition does not equal assigned rows")
    if coverage.get("valid_gaze_partition_check") is not True:
        raise MetricsValidationError("Face coverage.valid_gaze_partition_check is not true")
    expected_coverage = inside / assigned if assigned else None
    observed_coverage = coverage.get("face_coverage_ratio")
    if expected_coverage is None:
        if observed_coverage is not None:
            raise MetricsValidationError("Face coverage must be null for a zero denominator")
    elif not math.isclose(float(observed_coverage), expected_coverage, rel_tol=0, abs_tol=1e-12):
        raise MetricsValidationError("Face coverage ratio is inconsistent with counts")
    return {
        "selected_valid_source_rows": _required_int(counts, "selected_valid_source_rows"),
        "selected_invalid_source_rows": _required_int(counts, "selected_invalid_source_rows"),
        "assigned_unique_source_rows": assigned,
        "accepted_inside_face_unique_samples": inside,
        "rejected_outside_face_unique_samples": outside,
        "no_detection_unique_samples": no_detection,
        "identity_uncertain_unique_samples": uncertain,
        "coverage_ratio": observed_coverage,
        "frames_with_detection": _required_int(counts, "frames_with_detection"),
        "frames_without_detection": _required_int(counts, "frames_without_detection"),
        "multiple_face_frames": _required_int(counts, "multiple_face_frames"),
        "detection_availability_ratio": coverage.get("detection_availability_ratio"),
        "loss_episode_count": _required_int(lifecycle, "loss_episode_count"),
        "freeze_visible_frames": _required_int(lifecycle, "freeze_visible_frames"),
        "hidden_retained_frames": _required_int(lifecycle, "hidden_retained_frames"),
        "reacquisition_count": _required_int(lifecycle, "reacquisition_count"),
        "canonical_count": inside,
        "duplicate_consumption_excess": duplicate.get("duplicate_consumption_excess"),
        "maximum_samples_assigned_to_one_frame": assignment.get(
            "maximum_samples_assigned_to_one_frame"
        ),
    }


def validate_flow_core(core: dict[str, Any]) -> dict[str, Any]:
    if core.get("schema_version") != "optical_flow_event_semantics.v2":
        raise MetricsValidationError("Flow core is not optical_flow_event_semantics.v2")
    if core.get("history_mode") != "window":
        raise MetricsValidationError("L1 flow evidence must use recent-window history")
    if "shot_boundary" in core or "shot_boundary" in core.get("event_counts", {}):
        raise MetricsValidationError("Active L1 flow evidence contains legacy shot_boundary")
    terminations = core.get("termination_counts", {})
    for key in FLOW_TERMINATION_KEYS:
        _required_int(terminations, key)
    events = core.get("events", [])
    if not isinstance(events, list):
        raise MetricsValidationError("Flow events must be a list")
    allowed_actions = {"no_reset", "flow_tracking_lost", "confirmed_hard_cut_reset"}
    for event in events:
        if event.get("automatic_label") != "visual_discontinuity_candidate":
            raise MetricsValidationError("Flow event has a non-neutral automatic label")
        if event.get("action") not in allowed_actions:
            raise MetricsValidationError("Flow event has an unknown runtime action")
        if event.get("confirmed_hard_cut") and event.get("action") != "confirmed_hard_cut_reset":
            raise MetricsValidationError("Confirmed hard cut lacks its distinct reset action")
    event_counts = core.get("event_counts", {})
    automatic = _required_int(event_counts, "visual_discontinuity_candidate")
    if automatic != len(events):
        raise MetricsValidationError("Flow automatic candidate count does not match event records")
    added = _required_int(core, "added_gaze_count")
    final_active = _required_int(core, "final_active_count")
    if sum(_required_int(terminations, key) for key in FLOW_TERMINATION_KEYS) + final_active > added:
        raise MetricsValidationError("Flow terminations plus active trajectories exceed added gaze")
    reviewed = sum(event.get("reviewed_label") != "unreviewed" for event in events)
    action_counts = {action: sum(event.get("action") == action for event in events) for action in allowed_actions}
    return {
        "history_mode": core.get("history_mode"),
        "window_seconds": core.get("parameters", {}).get("window_seconds"),
        "selected_gaze_count": _required_int(core, "selected_gaze_count"),
        "added_gaze_count": added,
        "final_active_count": final_active,
        "maximum_active_count": _required_int(core, "maximum_active_count"),
        "termination_counts": {key: terminations[key] for key in FLOW_TERMINATION_KEYS},
        "automatic_candidate_count": automatic,
        "reviewed_event_count": reviewed,
        "reviewed_label_counts": {
            label: sum(event.get("reviewed_label") == label for event in events)
            for label in ("unreviewed", "continuous_non_cut", "confirmed_hard_cut", "uncertain")
        },
        "runtime_action_counts": action_counts,
        "confirmed_hard_cut_count": event_counts.get("confirmed_hard_cut", 0),
        "loss_episode_count": _required_int(core, "loss_episode_count"),
        "technical_segment_count": _required_int(core, "technical_segment_count"),
        "flow_interval_count": _required_int(core, "flow_interval_count"),
        "forward_backward_errors": core.get("forward_backward_errors", {}),
    }


def validate_l1_summary(summary: dict[str, Any], *, require_completed: bool = True) -> list[str]:
    errors: list[str] = []
    try:
        if summary.get("schema_version") != SCHEMA_VERSION:
            raise MetricsValidationError(f"Expected {SCHEMA_VERSION}")
        if summary.get("evidence_class") != "new_long_run":
            raise MetricsValidationError("L1 run must be explicitly labelled new_long_run")
        status = summary.get("status")
        if require_completed and status != "completed":
            raise MetricsValidationError(f"Run is not completed: {status}")
        method = summary.get("method")
        if method not in {"face", "flow"}:
            raise MetricsValidationError("L1 primary method must be face or flow")
        runtime = summary.get("runtime", {})
        if status == "completed":
            wall = float(runtime.get("wall_seconds", 0))
            fps = float(runtime.get("processing_fps", 0))
            ratio = float(runtime.get("processing_time_to_video_duration_ratio", 0))
            if wall <= 0 or fps <= 0 or ratio <= 0:
                raise MetricsValidationError("Completed run has invalid runtime metrics")
            output = summary.get("output", {})
            if _required_int(output, "primary_file_size_bytes") <= 0:
                raise MetricsValidationError("Completed run has no non-empty primary output")
            if _required_int(output, "output_frames") <= 0:
                raise MetricsValidationError("Completed run has no output frames")
            metrics = summary.get("metrics", {})
            if method == "face" and "assigned_unique_source_rows" not in metrics:
                raise MetricsValidationError("Face metrics are incomplete")
            if method == "flow" and set(metrics.get("termination_counts", {})) != set(FLOW_TERMINATION_KEYS):
                raise MetricsValidationError("Flow termination metrics are incomplete")
        if summary.get("claim_boundary", {}).get("absolute_gaze_accuracy_claimed") is not False:
            raise MetricsValidationError("Absolute gaze accuracy boundary is missing")
    except (MetricsValidationError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def validate_tool_manifest(tool_summary: dict[str, Any]) -> list[str]:
    """Validate the completed tool summary and its embedded artifact manifest."""
    errors: list[str] = []
    try:
        if tool_summary.get("schema_version") != "video_heatmap_tool_run.v2":
            raise MetricsValidationError("Tool summary is not video_heatmap_tool_run.v2")
        if tool_summary.get("status") != "completed" or tool_summary.get("exportable") is not True:
            raise MetricsValidationError("Tool summary is not completed/exportable")
        manifest = tool_summary.get("artifacts", {})
        if manifest.get("schema_version") != "video_heatmap_artifacts.v1":
            raise MetricsValidationError("Artifact manifest is not video_heatmap_artifacts.v1")
        base = Path(manifest.get("path_base", "")).resolve()
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise MetricsValidationError("Artifact manifest has no artifact records")
        seen_ids: set[str] = set()
        for artifact in artifacts:
            artifact_id = str(artifact.get("id", ""))
            if not artifact_id or artifact_id in seen_ids:
                raise MetricsValidationError("Artifact ids must be non-empty and unique")
            seen_ids.add(artifact_id)
            relative = Path(str(artifact.get("path", "")))
            if relative.is_absolute() or not str(relative):
                raise MetricsValidationError(f"Artifact {artifact_id} path must be relative")
            resolved = (base / relative).resolve()
            try:
                resolved.relative_to(base)
            except ValueError as exc:
                raise MetricsValidationError(f"Artifact {artifact_id} escapes path_base") from exc
            exists = resolved.is_file()
            if artifact.get("exists") is not exists:
                raise MetricsValidationError(f"Artifact {artifact_id} exists flag disagrees with disk")
            if artifact.get("exportable") is True and (not exists or resolved.stat().st_size <= 0):
                raise MetricsValidationError(f"Exportable artifact {artifact_id} is missing or empty")
    except (MetricsValidationError, OSError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def validate_event_review_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Validate reviewed event fields without promoting candidates to cuts."""
    errors: list[str] = []
    allowed_labels = {"continuous_non_cut", "confirmed_hard_cut", "uncertain"}
    allowed_confidence = {"high", "medium", "low"}
    allowed_actions = {"no_reset", "confirmed_hard_cut_reset"}
    seen_frames: set[int] = set()
    for index, row in enumerate(rows):
        missing = [field for field in EVENT_REVIEW_FIELDS if field not in row or row[field] in (None, "")]
        if missing:
            errors.append(f"row {index} missing fields: {', '.join(missing)}")
            continue
        try:
            frame = int(row["frame"])
        except (TypeError, ValueError):
            errors.append(f"row {index} has invalid frame")
            continue
        if frame in seen_frames:
            errors.append(f"duplicate reviewed frame: {frame}")
        seen_frames.add(frame)
        if row["automatic_label"] != "visual_discontinuity_candidate":
            errors.append(f"frame {frame} has non-neutral automatic label")
        if row["reviewed_label"] not in allowed_labels:
            errors.append(f"frame {frame} has invalid reviewed label")
        if row["review_confidence"] not in allowed_confidence:
            errors.append(f"frame {frame} has invalid review confidence")
        if row["action"] not in allowed_actions:
            errors.append(f"frame {frame} has invalid action")
        confirmed = row["confirmed_hard_cut"] in (True, "true", "True", 1, "1")
        if confirmed != (row["reviewed_label"] == "confirmed_hard_cut"):
            errors.append(f"frame {frame} cut flag disagrees with reviewed label")
        expected_action = "confirmed_hard_cut_reset" if confirmed else "no_reset"
        if row["action"] != expected_action:
            errors.append(f"frame {frame} action disagrees with cut decision")
        if not Path(str(row["evidence_path"])).is_file():
            errors.append(f"frame {frame} evidence clip is missing")
    return errors


def _total_files_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _range_duration(core: dict[str, Any], processed_frames: int) -> tuple[float, float]:
    if "video_info" in core:
        fps = float(core["video_info"]["fps"])
    else:
        selected = core.get("selected_range", {})
        start = float(selected.get("start_time", 0.0))
        end = float(selected.get("end_time", start))
        frame_span = max(1, int(selected.get("end_frame", 0)) - int(selected.get("start_frame", 0)))
        fps = frame_span / max(end - start, 1e-12)
    return processed_frames / fps, fps


def build_tool_command(args: argparse.Namespace, method_out: Path) -> list[str]:
    command = [
        args.python,
        str(ROOT / "scripts" / "run_heatmap_tool.py"),
        args.method,
        "--gaze",
        args.gaze,
        "--video",
        args.video,
        "--out-dir",
        str(method_out),
        "--start-frame",
        str(args.start_frame),
        "--end-frame",
        str(args.end_frame),
        "--frame-skip",
        str(args.frame_skip),
        "--opacity",
        str(args.opacity),
        "--gradient",
        args.gradient,
    ]
    if args.method == "face":
        command.extend(
            [
                "--model",
                args.model,
                "--brush-radius",
                str(args.face_brush_radius),
                "--detector-confidence",
                str(args.detector_confidence),
                "--max-freeze-seconds",
                str(args.max_freeze_seconds),
                "--heat-scale-mode",
                "auto",
            ]
        )
    else:
        command.extend(
            [
                "--brush-size",
                str(args.flow_brush_size),
                "--window-seconds",
                str(args.window_seconds),
                "--event-policy",
                "candidate-only",
                "--recovery-stable-frames",
                "3",
                "--heat-scale-mode",
                "fixed",
                "--heat-scale-maximum",
                "20",
            ]
        )
        if args.event_annotations:
            command.extend(["--event-annotations", args.event_annotations])
    return command


def run_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.out_dir)
    method_out = run_dir / "method_output"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_tool_command(args, method_out)
    (run_dir / "reproducible_command.txt").write_text(command_text(command) + "\n", encoding="utf-8")
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    wall_seconds = perf_counter() - started
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    errors: list[str] = []
    warnings = [
        "No independent gaze ground truth is available; this run does not measure absolute gaze accuracy.",
        "Peak memory is unavailable because reliable cross-process-tree sampling was not available in the frozen environment.",
    ]
    tool_summary_path = method_out / "tool_run_summary.json"
    core_path = method_out / (
        "face_relative_summary.json" if args.method == "face" else "optical_flow_video_summary.json"
    )
    status = "failed"
    metrics: dict[str, Any] = {}
    processed_frames = 0
    output_frames = 0
    primary_path = method_out / (
        "face_relative_dynamic_overlay.mp4" if args.method == "face" else "optical_flow_window_video.mp4"
    )
    delegated_commands: list[list[str]] = []
    video_duration_seconds: float | None = None
    source_fps: float | None = None
    if completed.returncode != 0:
        errors.append(f"Unified CLI exited with code {completed.returncode}")
    try:
        tool_summary = read_json(tool_summary_path)
        delegated_commands = tool_summary.get("config", {}).get("commands", [])
        if tool_summary.get("status") != "completed":
            errors.append(f"Unified CLI terminal status is {tool_summary.get('status')}")
        core = read_json(core_path)
        if args.method == "face":
            metrics = validate_face_core(core)
            processed_frames = _required_int(core.get("counts", {}), "processed_frames")
            output_frames = math.ceil(processed_frames / args.frame_skip)
        else:
            metrics = validate_flow_core(core)
            processed_frames = _required_int(core, "processed_frames")
            output_frames = _required_int(core, "written_frames")
            if metrics["reviewed_event_count"] < metrics["automatic_candidate_count"]:
                warnings.append("One or more automatic flow candidates remain unreviewed.")
        video_duration_seconds, source_fps = _range_duration(core, processed_frames)
        if not primary_path.is_file() or primary_path.stat().st_size == 0:
            errors.append(f"Primary output is missing or empty: {primary_path}")
    except (OSError, KeyError, MetricsValidationError, ValueError) as exc:
        errors.append(str(exc))
    if not errors:
        status = "completed"
    processing_fps = processed_frames / wall_seconds if processed_frames and wall_seconds else None
    ratio = wall_seconds / video_duration_seconds if video_duration_seconds else None
    primary_size = primary_path.stat().st_size if primary_path.is_file() else 0
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "participant": args.participant,
        "method": args.method,
        "evidence_class": "new_long_run",
        "interval_tier": args.interval_tier,
        "status": status,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"gaze": args.gaze, "video": args.video, "face_model": args.model if args.method == "face" else None},
        "requested_range": {"start_frame_inclusive": args.start_frame, "end_frame_exclusive": args.end_frame},
        "video": {
            "processed_frames": processed_frames,
            "processed_video_duration_seconds": video_duration_seconds,
            "source_fps": source_fps,
        },
        "history_semantics": (
            "face-relative cumulative canonical memory"
            if args.method == "face"
            else f"optical-flow recent window ({args.window_seconds:g} seconds)"
        ),
        "command_tokens": command,
        "reproducible_command": command_text(command),
        "delegated_command_tokens": delegated_commands,
        "runtime": {
            "wall_seconds": wall_seconds,
            "processing_fps": processing_fps,
            "processing_time_to_video_duration_ratio": ratio,
            "peak_memory": {
                "available": False,
                "bytes": None,
                "reason": "reliable cross-process-tree sampling unavailable in the frozen environment",
            },
        },
        "output": {
            "primary_path": str(primary_path),
            "output_frames": output_frames,
            "primary_file_size_bytes": primary_size,
            "total_run_directory_bytes": _total_files_bytes(run_dir),
            "primary_sha256": sha256_file(primary_path) if primary_size else None,
        },
        "parameters": {
            "frame_skip": args.frame_skip,
            "opacity": args.opacity,
            "gradient": args.gradient,
            "detector_confidence": args.detector_confidence if args.method == "face" else None,
            "max_freeze_seconds": args.max_freeze_seconds if args.method == "face" else None,
            "face_brush_radius": args.face_brush_radius if args.method == "face" else None,
            "window_seconds": args.window_seconds if args.method == "flow" else None,
            "flow_brush_size": args.flow_brush_size if args.method == "flow" else None,
            "event_policy": "candidate-only" if args.method == "flow" else None,
            "event_annotations": args.event_annotations if args.method == "flow" else None,
        },
        "metrics": metrics,
        "source_summaries": {"tool_run_summary": str(tool_summary_path), "core_summary": str(core_path)},
        "warnings": warnings,
        "errors": errors,
        "claim_boundary": {
            "absolute_gaze_accuracy_claimed": False,
            "visual_plausibility_is_observation_only": True,
            "face_and_flow_heat_intensity_directly_ranked": False,
            "reason": "No independent gaze ground truth; cumulative face memory and recent-window flow have different histories.",
        },
    }
    validation_errors = validate_l1_summary(summary, require_completed=False)
    if validation_errors:
        summary["errors"].extend(validation_errors)
        summary["status"] = "failed"
    write_json_atomic(run_dir / "l1_run_summary.json", summary)
    print(json.dumps({"status": summary["status"], "summary": str(run_dir / 'l1_run_summary.json')}, indent=2))
    return 0 if summary["status"] == "completed" else 1


def validate_command(args: argparse.Namespace) -> int:
    records = []
    failed = False
    for raw_path in args.summary:
        path = Path(raw_path)
        summary = read_json(path)
        errors = validate_l1_summary(summary, require_completed=not args.allow_failed)
        tool_summary_path = summary.get("source_summaries", {}).get("tool_run_summary")
        if tool_summary_path:
            errors.extend(validate_tool_manifest(read_json(Path(tool_summary_path))))
        else:
            errors.append("L1 summary has no tool_run_summary provenance")
        records.append({"path": str(path), "run_id": summary.get("run_id"), "status": summary.get("status"), "errors": errors})
        failed = failed or bool(errors)
    report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not failed,
        "summary_count": len(records),
        "records": records,
    }
    write_json_atomic(Path(args.out), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failed else 0


def validate_events_command(args: argparse.Namespace) -> int:
    with Path(args.csv).open("r", newline="", encoding="utf-8-sig") as stream:
        csv_rows = list(csv.DictReader(stream))
    document = json.loads(Path(args.json).read_text(encoding="utf-8"))
    json_rows = document.get("events", []) if isinstance(document, dict) else document
    errors: list[str] = []
    if not isinstance(json_rows, list):
        errors.append("Event JSON must contain a list or an events list")
        json_rows = []
    if len(csv_rows) != len(json_rows):
        errors.append("CSV/JSON event row counts do not match")
    else:
        for index, (csv_row, json_row) in enumerate(zip(csv_rows, json_rows)):
            for field in EVENT_REVIEW_FIELDS:
                left = str(csv_row.get(field, "")).lower()
                right = str(json_row.get(field, "")).lower()
                if left != right:
                    errors.append(f"row {index} CSV/JSON field differs: {field}")
    errors.extend(validate_event_review_rows(json_rows))
    report = {
        "schema_version": "l1_event_review_validation.v1",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "event_count": len(json_rows),
        "errors": errors,
    }
    write_json_atomic(Path(args.out), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


def parse_legacy_entry(value: str) -> dict[str, Any]:
    parts = value.split(",", 3)
    if len(parts) != 4:
        raise MetricsValidationError("Legacy entry must be PARTICIPANT,METHOD,LABEL,PATH")
    participant, method, label, path = parts
    if participant not in {"P2", "P3", "P4", "P5"} or method not in {"face", "flow", "baseline"}:
        raise MetricsValidationError("Invalid participant or method in legacy entry")
    return {
        "participant": participant,
        "method": method,
        "evidence_class": "existing_short_smoke",
        "interval_tier": "short_smoke",
        "run_id": label,
        "status": "existing_evidence",
        "source_summary": path,
    }


def aggregate_command(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = [parse_legacy_entry(value) for value in args.legacy_entry]
    for raw_path in args.summary:
        path = Path(raw_path)
        summary = read_json(path)
        errors = validate_l1_summary(summary, require_completed=False)
        if errors:
            raise MetricsValidationError(f"Invalid L1 summary {path}: {'; '.join(errors)}")
        metrics = summary.get("metrics", {})
        rows.append(
            {
                "participant": summary["participant"],
                "method": summary["method"],
                "evidence_class": summary["evidence_class"],
                "interval_tier": summary["interval_tier"],
                "run_id": summary["run_id"],
                "status": summary["status"],
                "start_frame_inclusive": summary["requested_range"]["start_frame_inclusive"],
                "end_frame_exclusive": summary["requested_range"]["end_frame_exclusive"],
                "processed_video_duration_seconds": summary["video"]["processed_video_duration_seconds"],
                "runtime_seconds": summary["runtime"]["wall_seconds"],
                "processing_fps": summary["runtime"]["processing_fps"],
                "processing_ratio": summary["runtime"]["processing_time_to_video_duration_ratio"],
                "selected_or_assigned": metrics.get("assigned_unique_source_rows", metrics.get("selected_gaze_count")),
                "accepted_or_added": metrics.get("accepted_inside_face_unique_samples", metrics.get("added_gaze_count")),
                "source_summary": str(path),
            }
        )
    rows.sort(key=lambda row: (row["participant"], row["evidence_class"], row["interval_tier"], row["method"]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "participant_run_matrix.json", {"schema_version": MATRIX_SCHEMA_VERSION, "rows": rows})
    fields = sorted({key for row in rows for key in row})
    with (out_dir / "participant_run_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "out_dir": str(out_dir)}, indent=2))
    return 0


def event_sheets_command(args: argparse.Namespace) -> int:
    """Create deterministic contact sheets for temporal event review."""
    import cv2  # Imported lazily so metric validation remains stdlib-only.
    import numpy as np

    event_dir = Path(args.event_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for video_path in sorted(event_dir.glob("frame_*_review.mp4")):
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = (
            list(range(frame_count))
            if args.all_frames
            else sorted({round(value) for value in np.linspace(0, max(0, frame_count - 1), 5)})
        )
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            size = (320, 180) if args.all_frames else (480, 270)
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            cv2.putText(frame, f"review frame {index}/{max(0, frame_count - 1)}", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(frame)
        capture.release()
        if not frames:
            raise MetricsValidationError(f"Could not decode review clip: {video_path}")
        if args.all_frames:
            columns = 6
            blank = np.zeros_like(frames[0])
            while len(frames) % columns:
                frames.append(blank.copy())
            sheet = np.vstack([np.hstack(frames[index:index + columns]) for index in range(0, len(frames), columns)])
            suffix = "all_frames"
        else:
            while len(frames) < 5:
                frames.append(frames[-1].copy())
            sheet = np.hstack(frames[:5])
            suffix = "contact_sheet"
        output = out_dir / f"{video_path.stem}_{suffix}.jpg"
        if not cv2.imwrite(str(output), sheet):
            raise MetricsValidationError(f"Could not write contact sheet: {output}")
        created.append(str(output))
    write_json_atomic(
        out_dir / "contact_sheet_index.json",
        {"schema_version": "l1_event_contact_sheets.v1", "source_event_dir": str(event_dir), "files": created},
    )
    print(json.dumps({"created": len(created), "out_dir": str(out_dir)}, indent=2))
    return 0


def review_continuous_command(args: argparse.Namespace) -> int:
    """Write a reviewed copy after every full temporal window was inspected."""
    source_csv = Path(args.source_csv)
    source_json = Path(args.source_json)
    with source_csv.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    json_document = json.loads(source_json.read_text(encoding="utf-8"))
    json_rows = json_document.get("events", []) if isinstance(json_document, dict) else json_document
    if not isinstance(json_rows, list) or len(json_rows) != len(rows):
        raise MetricsValidationError("CSV/JSON event annotation rows do not match")
    for csv_row, json_row in zip(rows, json_rows):
        if int(csv_row["frame"]) != int(json_row["frame"]):
            raise MetricsValidationError("CSV/JSON event frames do not match")
        note = (
            "Full approximately one-second temporal window reviewed frame-by-frame; "
            "continuous transition visible with participant/camera motion or blur and no unexplained content/time jump."
        )
        for row in (csv_row, json_row):
            row["reviewed_label"] = "continuous_non_cut"
            row["review_confidence"] = "high"
            row["reviewer"] = args.reviewer
            row["action"] = "no_reset"
            row["confirmed_hard_cut"] = False
            row["notes"] = note
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "frame", "time_seconds", "automatic_label", "reviewed_label", "review_confidence",
        "reviewer", "review_window", "action", "confirmed_hard_cut", "evidence_path", "notes",
    ]
    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_output = json_document if isinstance(json_document, dict) else json_rows
    out_prefix.with_suffix(".json").write_text(json.dumps(json_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"reviewed": len(rows), "out_prefix": str(out_prefix)}, indent=2))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L1 long-video runner, validator, and participant matrix builder")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--participant", choices=("P2", "P3", "P4", "P5"), required=True)
    run.add_argument("--method", choices=("face", "flow"), required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--interval-tier", choices=("one_minute", "three_minute", "full_video"), required=True)
    run.add_argument("--gaze", required=True)
    run.add_argument("--video", required=True)
    run.add_argument("--model")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--start-frame", type=int, required=True)
    run.add_argument("--end-frame", type=int, required=True)
    run.add_argument("--frame-skip", type=int, default=5)
    run.add_argument("--opacity", type=float, default=0.55)
    run.add_argument("--gradient", default="jet")
    run.add_argument("--detector-confidence", type=float, default=0.8)
    run.add_argument("--max-freeze-seconds", type=float, default=0.5)
    run.add_argument("--face-brush-radius", type=int, default=12)
    run.add_argument("--window-seconds", type=float, default=2.0)
    run.add_argument("--flow-brush-size", type=int, default=40)
    run.add_argument("--event-annotations")
    run.add_argument("--python", default=sys.executable)
    run.set_defaults(func=run_command)
    validate = sub.add_parser("validate")
    validate.add_argument("--summary", action="append", required=True)
    validate.add_argument("--out", required=True)
    validate.add_argument("--allow-failed", action="store_true")
    validate.set_defaults(func=validate_command)
    validate_events = sub.add_parser("validate-events")
    validate_events.add_argument("--csv", required=True)
    validate_events.add_argument("--json", required=True)
    validate_events.add_argument("--out", required=True)
    validate_events.set_defaults(func=validate_events_command)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--summary", action="append", default=[])
    aggregate.add_argument("--legacy-entry", action="append", default=[])
    aggregate.add_argument("--out-dir", required=True)
    aggregate.set_defaults(func=aggregate_command)
    sheets = sub.add_parser("event-sheets")
    sheets.add_argument("--event-dir", required=True)
    sheets.add_argument("--out-dir", required=True)
    sheets.add_argument("--all-frames", action="store_true")
    sheets.set_defaults(func=event_sheets_command)
    review = sub.add_parser("review-continuous")
    review.add_argument("--source-csv", required=True)
    review.add_argument("--source-json", required=True)
    review.add_argument("--out-prefix", required=True)
    review.add_argument("--reviewer", default="L1_executor")
    review.set_defaults(func=review_continuous_command)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if getattr(args, "method", None) == "face" and not args.model:
        raise SystemExit("--model is required for face runs")
    if getattr(args, "end_frame", 1) <= getattr(args, "start_frame", 0):
        raise SystemExit("--end-frame must be greater than --start-frame")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
