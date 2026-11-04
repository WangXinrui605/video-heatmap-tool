from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .artifacts import ARTIFACT_SCHEMA_VERSION, RUN_SCHEMA_VERSION, utc_now, write_json_atomic


EXPORT_SCHEMA_VERSION = "video_heatmap_export.v1"
EXPORT_MANIFEST_NAME = "video_heatmap_export_manifest.json"


@dataclass(frozen=True)
class ExportFile:
    artifact_id: str
    role: str
    label: str
    source: Path
    relative_path: Path


@dataclass(frozen=True)
class ExportPlan:
    destination: Path
    source_run_id: str
    source_method: str
    source_run_summary: Path
    limitations: tuple[str, ...]
    files: tuple[ExportFile, ...]
    conflicts: tuple[Path, ...]


def export_folder_name(summary: dict[str, Any]) -> str:
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(summary.get("run_id") or "run")).strip("._")
    return f"{run_id or 'run'}_export"


def _validated_files(summary: dict[str, Any]) -> tuple[list[ExportFile], Path]:
    if summary.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("Export requires video_heatmap_tool_run.v2.")
    if summary.get("status") != "completed" or summary.get("exportable") is not True:
        raise ValueError("Export is available only for a completed, exportable run.")
    manifest = summary.get("artifacts")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Export requires a validated video_heatmap_artifacts.v1 manifest.")
    base_text = manifest.get("path_base")
    if not isinstance(base_text, str) or not base_text:
        raise ValueError("The artifact manifest does not define its run directory.")
    base = Path(base_text).resolve()
    items = manifest.get("artifacts")
    if not isinstance(items, list):
        raise ValueError("The artifact manifest does not contain an artifact list.")

    files: list[ExportFile] = []
    source_summary: Path | None = None
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("exportable") is not True:
            continue
        artifact_id = str(item.get("id", ""))
        if not artifact_id or artifact_id in seen_ids:
            raise ValueError("Export artifact IDs must be non-empty and unique.")
        if item.get("path_base") != "run_directory":
            raise ValueError(f"Artifact {artifact_id} has an unsupported path base.")
        relative = Path(str(item.get("path", "")))
        if not str(relative) or relative.is_absolute() or relative in seen_paths:
            raise ValueError(f"Artifact {artifact_id} has an invalid or duplicate relative path.")
        source = (base / relative).resolve()
        try:
            source.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Artifact {artifact_id} escapes its run directory.") from exc
        if item.get("exists") is not True or not source.is_file():
            raise ValueError(f"Export artifact is missing: {source}")
        files.append(ExportFile(
            artifact_id=artifact_id,
            role=str(item.get("role", "")),
            label=str(item.get("label", artifact_id)),
            source=source,
            relative_path=relative,
        ))
        if artifact_id == "run_summary":
            source_summary = source
        seen_ids.add(artifact_id)
        seen_paths.add(relative)
    if not files:
        raise ValueError("The completed run has no manifest-declared exportable files.")
    if source_summary is None:
        raise ValueError("The exportable manifest does not include tool_run_summary.json.")
    return files, source_summary


def build_export_plan(summary: dict[str, Any], destination: Path) -> ExportPlan:
    files, source_summary = _validated_files(summary)
    destination = destination.resolve()
    targets = [destination / item.relative_path for item in files]
    manifest_path = destination / EXPORT_MANIFEST_NAME
    conflicts = tuple(path for path in (*targets, manifest_path) if path.exists())
    method_record = summary.get("config", {}).get("method", {})
    limitations = method_record.get("limitations", []) if isinstance(method_record, dict) else []
    if not isinstance(limitations, list):
        limitations = []
    return ExportPlan(
        destination=destination,
        source_run_id=str(summary.get("run_id", "")),
        source_method=str(summary.get("method", "")),
        source_run_summary=source_summary,
        limitations=tuple(str(item) for item in limitations),
        files=tuple(files),
        conflicts=conflicts,
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute_export(plan: ExportPlan, *, overwrite: bool = False) -> Path:
    current_conflicts = tuple(
        path
        for path in (
            *(plan.destination / item.relative_path for item in plan.files),
            plan.destination / EXPORT_MANIFEST_NAME,
        )
        if path.exists()
    )
    if current_conflicts and not overwrite:
        raise FileExistsError("Export target conflicts with existing files: " + ", ".join(map(str, current_conflicts)))
    plan.destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in plan.files:
        target = plan.destination / item.relative_path
        _copy_atomic(item.source, target)
        records.append({
            "artifact_id": item.artifact_id,
            "role": item.role,
            "label": item.label,
            "source_path": str(item.source),
            "export_path": item.relative_path.as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": _digest(target),
        })
    manifest_path = plan.destination / EXPORT_MANIFEST_NAME
    write_json_atomic(manifest_path, {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "source_run_id": plan.source_run_id,
        "method": plan.source_method,
        "source_run_summary": str(plan.source_run_summary),
        "exported_at": utc_now(),
        "export_directory": str(plan.destination),
        "files": records,
        "limitations": list(plan.limitations),
    })
    return manifest_path
