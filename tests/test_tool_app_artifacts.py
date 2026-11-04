from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.tool_app.artifacts import RunStatus, artifact_manifest, new_run_record, transition_run
from video_heatmap.tool_app.config import Method


class ArtifactAndStatusTests(unittest.TestCase):
    def test_manifest_has_stable_fields_without_filename_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            path = folder / "dynamic" / "heatmap_development.mp4"
            path.parent.mkdir()
            path.touch()
            manifest = artifact_manifest(Method.BASELINE, folder, completed=True)
        item = next(item for item in manifest["artifacts"] if item["id"] == "dynamic")
        self.assertTrue(any(item["id"] == "run_summary" for item in manifest["artifacts"]))
        self.assertEqual(
            set(item),
            {"id", "kind", "role", "label", "path", "path_base", "exists", "media_type", "previewable", "exportable"},
        )
        self.assertTrue(item["previewable"])

    def test_running_cancelled_and_failed_are_not_exportable(self) -> None:
        for status in (RunStatus.CANCELLED, RunStatus.FAILED):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as name:
                folder = Path(name)
                (folder / "optical_flow_window_video.mp4").touch()
                running = new_run_record(Method.FLOW, {}, folder, run_id="run-1")
                terminal = transition_run(running, status, folder, error="test")
                self.assertFalse(running["exportable"])
                self.assertFalse(terminal["exportable"])
                self.assertFalse(any(item["previewable"] for item in terminal["artifacts"]["artifacts"]))

    def test_completed_requires_a_primary_artifact_to_be_exportable(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            running = new_run_record(Method.FLOW, {}, folder)
            missing = transition_run(running, RunStatus.COMPLETED, folder)
        self.assertFalse(missing["exportable"])
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "optical_flow_window_video.mp4").touch()
            completed = transition_run(new_run_record(Method.FLOW, {}, folder), RunStatus.COMPLETED, folder)
        self.assertTrue(completed["exportable"])

    def test_hiding_marker_from_gui_does_not_remove_legacy_artifact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "brasilia_marker_heatmap.mp4").touch()
            manifest = artifact_manifest(Method.MARKER, folder, completed=True)
        self.assertEqual(manifest["schema_version"], "video_heatmap_artifacts.v1")
        dynamic = next(item for item in manifest["artifacts"] if item["id"] == "dynamic")
        self.assertEqual(dynamic["path"], "brasilia_marker_heatmap.mp4")
        self.assertTrue(dynamic["previewable"])


if __name__ == "__main__":
    unittest.main()
