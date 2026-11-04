from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.tool_app.artifacts import RunStatus, new_run_record, transition_run, write_json_atomic
from video_heatmap.tool_app.config import Method
from video_heatmap.tool_app.export import (
    EXPORT_MANIFEST_NAME,
    EXPORT_SCHEMA_VERSION,
    build_export_plan,
    execute_export,
)


def completed_export_run(folder: Path) -> dict:
    dynamic = folder / "dynamic" / "heatmap_development.mp4"
    dynamic.parent.mkdir(parents=True)
    dynamic.write_bytes(b"video-result")
    static_summary = folder / "static" / "final_heatmap_summary.json"
    static_summary.parent.mkdir(parents=True)
    static_summary.write_text('{"samples": 3}', encoding="utf-8")
    (folder / "tool_run_summary.json").write_text("{}", encoding="utf-8")
    record = transition_run(
        new_run_record(Method.BASELINE, {}, folder, run_id="export-run"),
        RunStatus.COMPLETED,
        folder,
    )
    write_json_atomic(folder / "tool_run_summary.json", record)
    return record


class ExportTests(unittest.TestCase):
    def test_export_copies_only_manifest_files_and_writes_traceable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "run"
            run.mkdir()
            summary = completed_export_run(run)
            (run / "unlisted_partial.tmp").write_text("do not export", encoding="utf-8")
            destination = root / "formal-export"
            manifest_path = execute_export(build_export_plan(summary, destination))
            self.assertEqual(manifest_path, destination / EXPORT_MANIFEST_NAME)
            self.assertFalse((destination / "unlisted_partial.tmp").exists())
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], EXPORT_SCHEMA_VERSION)
            self.assertEqual(data["source_run_id"], "export-run")
            self.assertEqual(data["method"], "baseline")
            ids = {item["artifact_id"] for item in data["files"]}
            self.assertEqual(ids, {"dynamic", "static_summary", "run_summary"})
            for item in data["files"]:
                self.assertEqual(len(item["sha256"]), 64)

    def test_conflict_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "run"
            run.mkdir()
            summary = completed_export_run(run)
            destination = root / "export"
            execute_export(build_export_plan(summary, destination))
            conflict_plan = build_export_plan(summary, destination)
            self.assertTrue(conflict_plan.conflicts)
            with self.assertRaises(FileExistsError):
                execute_export(conflict_plan)
            execute_export(conflict_plan, overwrite=True)

    def test_incomplete_run_cannot_be_exported(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            running = new_run_record(Method.FLOW, {}, folder)
            with self.assertRaisesRegex(ValueError, "completed, exportable"):
                build_export_plan(running, folder / "export")

    def test_export_rejects_manifest_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "run"
            run.mkdir()
            summary = completed_export_run(run)
            item = next(item for item in summary["artifacts"]["artifacts"] if item["id"] == "dynamic")
            item["path"] = "../outside.mp4"
            with self.assertRaisesRegex(ValueError, "escapes"):
                build_export_plan(summary, root / "export")


if __name__ == "__main__":
    unittest.main()
