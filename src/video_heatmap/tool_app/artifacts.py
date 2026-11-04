from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
from uuid import uuid4

from .config import Method


RUN_SCHEMA_VERSION = "video_heatmap_tool_run.v2"
ARTIFACT_SCHEMA_VERSION = "video_heatmap_artifacts.v1"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class Artifact:
    id: str
    kind: str
    role: str
    label: str
    path: str
    path_base: str
    exists: bool
    media_type: str
    previewable: bool
    exportable: bool


_SPECS: dict[Method, tuple[tuple[str, str, str, str, str, bool, bool], ...]] = {
    Method.BASELINE: (
        ("dynamic", "video", "primary_dynamic", "Cumulative development video", "dynamic/heatmap_development.mp4", "video/mp4", True, True),
        ("overlay", "image", "primary_static", "Final overlay", "static/final_heatmap_overlay.png", "image/png", True, True),
        ("colour", "image", "supporting_static", "Final colour heatmap", "static/final_heatmap_color.png", "image/png", True, True),
        ("static_summary", "summary", "core_summary", "Static summary", "static/final_heatmap_summary.json", "application/json", False, True),
        ("dynamic_summary", "summary", "core_summary", "Dynamic summary", "dynamic/heatmap_development_summary.json", "application/json", False, True),
    ),
    Method.FACE: (
        ("dynamic", "video", "primary_dynamic", "Face-relative cumulative overlay", "face_relative_dynamic_overlay.mp4", "video/mp4", True, True),
        ("debug", "video", "debug", "Tracking/debug overlay", "face_relative_debug_overlay.mp4", "video/mp4", True, True),
        ("static", "image", "primary_static", "Canonical static heatmap", "face_relative_static_heatmap.png", "image/png", True, True),
        ("summary", "summary", "core_summary", "Face-relative summary", "face_relative_summary.json", "application/json", False, True),
        ("assignment_csv", "audit", "audit", "Gaze assignment audit CSV", "gaze_assignment_audit.csv", "text/csv", False, True),
        ("assignment_json", "audit", "audit", "Gaze assignment audit JSON", "gaze_assignment_audit.json", "application/json", False, True),
    ),
    Method.FLOW: (
        ("dynamic", "video", "primary_dynamic", "Optical-flow recent-window video", "optical_flow_window_video.mp4", "video/mp4", True, True),
        ("summary", "summary", "core_summary", "Optical-flow summary", "optical_flow_video_summary.json", "application/json", False, True),
        ("events_csv", "audit", "audit", "Runtime event annotations CSV", "runtime_event_annotations.csv", "text/csv", False, True),
        ("events_json", "audit", "audit", "Runtime event annotations JSON", "runtime_event_annotations.json", "application/json", False, True),
    ),
    Method.MARKER: (
        ("dynamic", "video", "primary_dynamic", "Marker-local cumulative video", "{marker}_marker_heatmap.mp4", "video/mp4", True, True),
        ("summary", "summary", "core_summary", "Marker-local summary", "{marker}_marker_heatmap_summary.json", "application/json", False, True),
    ),
}


def discover_artifacts(method: Method, output_dir: Path, *, marker: str = "Brasilia", completed: bool = False) -> list[Artifact]:
    marker_slug = marker.lower()
    result = []
    specs = (*_SPECS[method], (
        "run_summary", "summary", "run_summary", "Tool run summary",
        "tool_run_summary.json", "application/json", False, True,
    ))
    for artifact_id, kind, role, label, relative, media_type, previewable, intended_exportable in specs:
        relative = relative.format(marker=marker_slug)
        path = output_dir / Path(relative)
        exists = path.is_file()
        result.append(Artifact(
            id=artifact_id, kind=kind, role=role, label=label,
            path=relative.replace("\\", "/"), path_base="run_directory",
            exists=exists, media_type=media_type,
            previewable=bool(completed and exists and previewable),
            exportable=bool(completed and exists and intended_exportable),
        ))
    return result


def artifact_manifest(method: Method, output_dir: Path, *, marker: str = "Brasilia", completed: bool = False) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "path_base": str(output_dir.resolve()),
        "artifacts": [asdict(item) for item in discover_artifacts(method, output_dir, marker=marker, completed=completed)],
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_record(method: Method, config: dict[str, Any], output_dir: Path, *, run_id: str | None = None, marker: str = "Brasilia") -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id or str(uuid4()),
        "status": RunStatus.RUNNING.value,
        "exportable": False,
        "started_at": utc_now(),
        "completed_at": None,
        "method": method.value,
        "config": config,
        "artifacts": artifact_manifest(method, output_dir, marker=marker, completed=False),
        "error": None,
    }


def transition_run(record: dict[str, Any], status: RunStatus, output_dir: Path, *, marker: str = "Brasilia", error: str | None = None) -> dict[str, Any]:
    if record.get("status") != RunStatus.RUNNING.value:
        raise ValueError("Only a running run can transition to a terminal status.")
    if status is RunStatus.RUNNING:
        raise ValueError("Terminal transition requires completed, cancelled, or failed.")
    updated = dict(record)
    updated["status"] = status.value
    updated["completed_at"] = utc_now()
    updated["error"] = error
    method = Method(str(record["method"]))
    completed = status is RunStatus.COMPLETED
    manifest = artifact_manifest(method, output_dir, marker=marker, completed=completed)
    updated["artifacts"] = manifest
    primary_exists = any(
        item["exists"] and item["role"].startswith("primary_")
        for item in manifest["artifacts"]
    )
    updated["exportable"] = bool(completed and primary_exists)
    return updated


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
