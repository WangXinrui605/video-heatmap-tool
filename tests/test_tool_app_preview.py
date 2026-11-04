from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtGui import QColor, QImage
from PySide6.QtMultimedia import QVideoFrame
from PySide6.QtWidgets import QApplication

from video_heatmap.tool_app.artifacts import RunStatus, new_run_record, transition_run, write_json_atomic
from video_heatmap.tool_app.config import Method
from video_heatmap.tool_app.preview import (
    PreviewWidget,
    SCREENSHOT_SCHEMA_VERSION,
    preview_artifacts,
    save_screenshot,
)


def completed_image_run(folder: Path) -> dict:
    image_path = folder / "static" / "final_heatmap_overlay.png"
    image_path.parent.mkdir(parents=True)
    image = QImage(12, 8, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    assert image.save(str(image_path))
    (folder / "tool_run_summary.json").write_text("{}", encoding="utf-8")
    record = transition_run(
        new_run_record(Method.BASELINE, {}, folder, run_id="preview-run"),
        RunStatus.COMPLETED,
        folder,
    )
    write_json_atomic(folder / "tool_run_summary.json", record)
    return record


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_selector_is_manifest_driven_and_rejects_incomplete_runs(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            completed = completed_image_run(folder)
            artifacts = preview_artifacts(completed)
            self.assertEqual([item.id for item in artifacts], ["overlay"])
            running = new_run_record(Method.BASELINE, {}, folder)
            with self.assertRaisesRegex(ValueError, "completed, exportable"):
                preview_artifacts(running)

    def test_widget_fits_image_and_video_speed_is_player_only(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            summary = completed_image_run(Path(name))
            widget = PreviewWidget()
            widget.set_summary(summary)
            self.assertEqual(widget.artifact_combo.count(), 1)
            self.assertEqual(widget.current_artifact.id, "overlay")
            self.assertFalse(widget.current_frame.isNull())
            widget.speed_combo.setCurrentIndex(3)
            self.assertEqual(widget.player.playbackRate(), 2.0)
            self.assertEqual(summary["config"], {})
            widget.clear()
            self.assertEqual(widget.artifact_combo.count(), 0)
            widget.deleteLater()

    def test_selector_label_and_pre_completion_empty_state_are_visible(self) -> None:
        widget = PreviewWidget()
        widget.show()
        self.app.processEvents()
        self.assertEqual(widget.output_to_preview_label.text(), "Output to preview")
        self.assertTrue(widget.preview_empty_state.isVisible())
        self.assertIn("completed, exportable", widget.preview_empty_state.text())
        widget.deleteLater()

    def test_video_artifact_enables_play_seek_and_required_rates(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            video = folder / "dynamic" / "heatmap_development.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"manifest-declared-video")
            (folder / "tool_run_summary.json").write_text("{}", encoding="utf-8")
            summary = transition_run(
                new_run_record(Method.BASELINE, {}, folder, run_id="video-preview"),
                RunStatus.COMPLETED,
                folder,
            )
            write_json_atomic(folder / "tool_run_summary.json", summary)
            widget = PreviewWidget()
            widget.set_summary(summary)
            self.assertEqual(widget.current_artifact.id, "dynamic")
            self.assertTrue(widget.play_button.isEnabled())
            self.assertTrue(widget.seek_slider.isEnabled())
            self.assertEqual(
                [widget.speed_combo.itemData(index) for index in range(widget.speed_combo.count())],
                [0.5, 1.0, 1.5, 2.0],
            )
            widget._duration_changed(5000)
            widget._position_changed(1250)
            self.assertEqual(widget.seek_slider.maximum(), 5000)
            self.assertEqual(widget.time_label.text(), "00:01 / 00:00")
            widget.deleteLater()

    def test_decoded_video_frame_becomes_traceable_screenshot_source(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            video = folder / "dynamic" / "heatmap_development.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"manifest-declared-video")
            (folder / "tool_run_summary.json").write_text("{}", encoding="utf-8")
            summary = transition_run(
                new_run_record(Method.BASELINE, {}, folder, run_id="video-frame-run"),
                RunStatus.COMPLETED,
                folder,
            )
            write_json_atomic(folder / "tool_run_summary.json", summary)
            widget = PreviewWidget()
            widget.set_summary(summary)
            decoded = QImage(9, 7, QImage.Format.Format_RGB32)
            decoded.fill(QColor("blue"))
            widget._video_frame_changed(QVideoFrame(decoded))
            self.assertEqual(widget.current_frame.size(), decoded.size())
            saved, sidecar = save_screenshot(
                widget.current_frame,
                folder / "video_frame.png",
                summary=summary,
                artifact=widget.current_artifact,
                media_time_ms=2468,
            )
            self.assertTrue(saved.is_file())
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8"))["media_time_ms"], 2468)
            widget.deleteLater()

    def test_screenshot_has_identity_time_and_does_not_change_source(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            summary = completed_image_run(folder)
            artifact = preview_artifacts(summary)[0]
            before = sha256(artifact.path.read_bytes()).hexdigest()
            destination = folder / "captures" / "current.png"
            image = QImage(str(artifact.path))
            saved, sidecar = save_screenshot(
                image,
                destination,
                summary=summary,
                artifact=artifact,
                media_time_ms=1234,
            )
            self.assertTrue(saved.is_file())
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], SCREENSHOT_SCHEMA_VERSION)
            self.assertEqual(data["source_run_id"], "preview-run")
            self.assertEqual(data["source_artifact"]["id"], "overlay")
            self.assertEqual(data["media_time_ms"], 1234)
            self.assertEqual(Path(data["saved_path"]), saved)
            self.assertEqual(sha256(artifact.path.read_bytes()).hexdigest(), before)

    def test_manifest_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            summary = completed_image_run(Path(name))
            item = next(item for item in summary["artifacts"]["artifacts"] if item["id"] == "overlay")
            item["path"] = "../outside.png"
            with self.assertRaisesRegex(ValueError, "escapes"):
                preview_artifacts(summary)

    def test_screenshot_rejects_previewable_but_non_exportable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            summary = completed_image_run(Path(name))
            item = next(item for item in summary["artifacts"]["artifacts"] if item["id"] == "overlay")
            item["exportable"] = False
            artifact = preview_artifacts(summary)[0]
            with self.assertRaisesRegex(ValueError, "previewable artifact"):
                save_screenshot(
                    QImage(str(artifact.path)),
                    Path(name) / "capture.png",
                    summary=summary,
                    artifact=artifact,
                    media_time_ms=0,
                )


if __name__ == "__main__":
    unittest.main()
