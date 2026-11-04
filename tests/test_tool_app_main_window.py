from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QComboBox

from video_heatmap.tool_app.adapter import command_tokens
from video_heatmap.tool_app.artifacts import RunStatus, new_run_record, transition_run, write_json_atomic
from video_heatmap.tool_app.config import Method
from video_heatmap.tool_app.main_window import AppState, MainWindow
from video_heatmap.tool_app.process_runner import ProcessRunner
from video_heatmap.video_io import require_cv2


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def make_inputs(self, folder: Path) -> tuple[Path, Path]:
        gaze = folder / "P2_gaze.txt"
        gaze.write_text(
            "rec_time\tDikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze X\t"
            "Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze Y\n"
            "00:00:00.100\t5\t6\n",
            encoding="utf-8",
        )
        video = folder / "scene.avi"
        cv2 = require_cv2()
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
        for _ in range(5):
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
        writer.release()
        return gaze, video

    def set_valid_inputs(self, folder: Path) -> None:
        gaze, video = self.make_inputs(folder)
        self.window.output_edit.setText(str(folder / "outputs"))
        self.window.gaze_picker.edit.setText(str(gaze))
        self.window.video_picker.edit.setText(str(video))
        self.window._input_paths_changed()

    def test_initial_state_and_result_actions(self) -> None:
        self.assertEqual(self.window.state, AppState.EMPTY)
        self.assertFalse(self.window.generate_button.isEnabled())
        self.assertFalse(self.window.cancel_button.isEnabled())
        self.assertTrue(self.window.validate_button.isEnabled())
        self.assertFalse(self.window.preview_button.isEnabled())
        self.assertTrue(self.window.clear_button.isEnabled())
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertFalse(self.window.screenshot_button.isEnabled())
        self.assertEqual(
            [self.window.preview_widget.speed_combo.itemText(index) for index in range(4)],
            ["0.5x", "1.0x", "1.5x", "2.0x"],
        )

    def test_valid_inputs_enter_ready_and_frame_default_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            self.set_valid_inputs(Path(name))
            self.assertEqual(self.window.state, AppState.INPUT_READY)
            self.assertTrue(self.window.generate_button.isEnabled())
            self.assertEqual(self.window.end_frame.value(), 5)
            self.assertEqual(self.window.video_metadata.frame_count, 5)

    def test_invalid_range_disables_generate(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            self.set_valid_inputs(Path(name))
            self.window.start_frame.setValue(4)
            self.window.end_frame.setValue(4)
            self.window._refresh_validation()
            self.assertEqual(self.window.state, AppState.EMPTY)
            self.assertIn("exclusive", self.window.last_validation_error)

    def test_method_switch_and_face_model_validation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            self.set_valid_inputs(Path(name))
            self.window.method_combo.setCurrentIndex(1)
            self.window._refresh_validation()
            self.assertIs(self.window.method_panels.currentWidget(), self.window.method_panels.face)
            self.assertEqual(self.window.state, AppState.EMPTY)
            self.assertIn("Face model", self.window.last_validation_error)

    def test_first_release_selector_has_three_methods_without_marker_or_name_inference(self) -> None:
        methods = [self.window.method_combo.itemData(index) for index in range(self.window.method_combo.count())]
        self.assertEqual(methods, [Method.BASELINE, Method.FACE, Method.FLOW])
        self.assertFalse(hasattr(self.window.method_panels, "marker"))
        self.assertFalse(hasattr(self.window, "_participant_id"))

    def test_processing_and_terminal_states_allow_rerun(self) -> None:
        self.window._set_state(AppState.PROCESSING)
        self.assertFalse(self.window.generate_button.isEnabled())
        self.assertTrue(self.window.cancel_button.isEnabled())
        self.assertFalse(self.window.validate_button.isEnabled())
        self.assertIn("Processing", self.window.state_label.text())
        self.assertEqual((self.window.progress.minimum(), self.window.progress.maximum()), (0, 0))
        for state in (AppState.COMPLETED, AppState.FAILED, AppState.CANCELLED):
            self.window._set_state(state)
            self.assertTrue(self.window.generate_button.isEnabled())
            self.assertFalse(self.window.cancel_button.isEnabled())

    def test_completed_manifest_loads_preview_and_clear_preserves_outputs_and_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.set_valid_inputs(folder)
            run = folder / "completed-run"
            image_path = run / "static" / "final_heatmap_overlay.png"
            image_path.parent.mkdir(parents=True)
            cv2 = require_cv2()
            self.assertTrue(cv2.imwrite(str(image_path), np.full((8, 12, 3), 127, dtype=np.uint8)))
            (run / "tool_run_summary.json").write_text("{}", encoding="utf-8")
            summary = transition_run(
                new_run_record(Method.BASELINE, {}, run, run_id="ui-run"),
                RunStatus.COMPLETED,
                run,
            )
            write_json_atomic(run / "tool_run_summary.json", summary)
            gaze_before = self.window.gaze_picker.edit.text()
            video_before = self.window.video_picker.edit.text()

            self.window._run_terminal("completed", summary)
            self.assertEqual(self.window.state, AppState.COMPLETED)
            self.assertTrue(self.window.preview_button.isEnabled())
            self.assertTrue(self.window.screenshot_button.isEnabled())
            self.assertTrue(self.window.export_button.isEnabled())
            self.assertEqual(self.window.preview_widget.current_artifact.id, "overlay")

            self.window.clear_result()
            self.assertTrue(image_path.is_file())
            self.assertEqual(self.window.gaze_picker.edit.text(), gaze_before)
            self.assertEqual(self.window.video_picker.edit.text(), video_before)
            self.assertEqual(self.window.state, AppState.INPUT_READY)
            self.assertFalse(self.window.export_button.isEnabled())
            self.assertEqual(self.window.preview_widget.artifact_combo.count(), 0)

    def test_preview_speed_does_not_change_backend_command(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.set_valid_inputs(folder)
            config_before = self.window.build_config(output_dir=folder / "run")
            tokens_before = command_tokens(config_before)
            self.window.preview_widget.speed_combo.setCurrentIndex(3)
            config_after = self.window.build_config(output_dir=folder / "run")
            self.assertEqual(command_tokens(config_after), tokens_before)

    def test_wheel_over_common_and_method_specific_controls_scrolls_panel_without_changing_values(self) -> None:
        self.window.resize(700, 480)
        self.window.show()
        self.app.processEvents()
        bar = self.window.controls_scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        bar.setValue(bar.maximum())
        common_before = self.window.start_frame.value()
        common_event = QWheelEvent(
            QPointF(self.window.start_frame.rect().center()),
            QPointF(self.window.start_frame.mapToGlobal(self.window.start_frame.rect().center())),
            QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate, False,
        )
        QApplication.sendEvent(self.window.start_frame, common_event)
        self.app.processEvents()
        self.assertLess(bar.value(), bar.maximum())
        self.assertEqual(self.window.start_frame.value(), common_before)

        bar.setValue(bar.maximum())
        gradient_before = self.window.gradient.currentData()
        gradient_event = QWheelEvent(
            QPointF(self.window.gradient.rect().center()),
            QPointF(self.window.gradient.mapToGlobal(self.window.gradient.rect().center())),
            QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate, False,
        )
        QApplication.sendEvent(self.window.gradient, gradient_event)
        self.app.processEvents()
        self.assertLess(bar.value(), bar.maximum())
        self.assertEqual(self.window.gradient.currentData(), gradient_before)

        self.window.method_combo.setCurrentIndex(1)
        self.app.processEvents()
        bar.setValue(bar.maximum())
        method_before = self.window.method_panels.face.confidence.value()
        method_event = QWheelEvent(
            QPointF(self.window.method_panels.face.confidence.rect().center()),
            QPointF(self.window.method_panels.face.confidence.mapToGlobal(
                self.window.method_panels.face.confidence.rect().center()
            )),
            QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate, False,
        )
        QApplication.sendEvent(self.window.method_panels.face.confidence, method_event)
        self.app.processEvents()
        self.assertLess(bar.value(), bar.maximum())
        self.assertEqual(self.window.method_panels.face.confidence.value(), method_before)

        self.window.method_combo.setCurrentIndex(2)
        self.app.processEvents()
        bar.setValue(bar.maximum())
        policy_before = self.window.method_panels.flow.event_policy.currentText()
        policy_event = QWheelEvent(
            QPointF(self.window.method_panels.flow.event_policy.rect().center()),
            QPointF(self.window.method_panels.flow.event_policy.mapToGlobal(
                self.window.method_panels.flow.event_policy.rect().center()
            )),
            QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate, False,
        )
        QApplication.sendEvent(self.window.method_panels.flow.event_policy, policy_event)
        self.app.processEvents()
        self.assertLess(bar.value(), bar.maximum())
        self.assertEqual(self.window.method_panels.flow.event_policy.currentText(), policy_before)

    def test_cancelled_terminal_with_runconfig_shape_never_enables_results(self) -> None:
        data = {
            "schema_version": "video_heatmap_tool_run.v2",
            "status": "cancelled",
            "exportable": False,
            "config": {"method": "baseline"},
        }
        self.window._run_terminal("cancelled", data)
        self.assertEqual(self.window.state, AppState.CANCELLED)
        self.assertFalse(self.window.preview_button.isEnabled())
        self.assertFalse(self.window.screenshot_button.isEnabled())
        self.assertFalse(self.window.export_button.isEnabled())

    def test_cancelled_and_failed_terminals_preserve_last_real_progress(self) -> None:
        for status, expected_state in (
            ("cancelled", AppState.CANCELLED),
            ("failed", AppState.FAILED),
        ):
            with self.subTest(status=status):
                self.window._set_state(AppState.PROCESSING)
                self.window._progress_received({
                    "schema_version": "video_heatmap_progress.v1",
                    "run_id": f"{status}-run",
                    "method": "flow",
                    "stage": "processing",
                    "processed_input_frames": 3,
                    "total_input_frames": 300,
                    "percentage": 1.0,
                    "elapsed_seconds": 0.5,
                    "estimated_remaining_seconds": None,
                    "timestamp": "2026-08-30T00:00:00+00:00",
                    "sequence": 3,
                })
                self.window._run_terminal(status, {
                    "schema_version": "video_heatmap_tool_run.v2",
                    "status": status,
                    "exportable": False,
                    "config": {"method": "flow"},
                })
                self.assertEqual(self.window.state, expected_state)
                self.assertEqual(self.window.progress.value(), 10)
                self.assertEqual(self.window.progress.format(), "1.0%")
                self.assertIn("Frames: 3/300", self.window.progress_label.text())
                self.assertNotIn("100", self.window.progress_label.text())
                self.assertFalse(self.window.preview_button.isEnabled())
                self.assertFalse(self.window.screenshot_button.isEnabled())
                self.assertFalse(self.window.export_button.isEnabled())

    def test_cancellation_failure_leaves_failed_nonexportable_ui_while_process_settles(self) -> None:
        data = {
            "schema_version": "video_heatmap_tool_run.v2",
            "status": "failed",
            "exportable": False,
            "error": "Targeted windows-job-object cancellation failed with code 5: Access is denied.",
        }
        with __import__("unittest.mock").mock.patch.object(
            ProcessRunner,
            "running",
            new_callable=__import__("unittest.mock").mock.PropertyMock,
            return_value=True,
        ):
            self.window._run_terminal("failed", data)
            self.assertEqual(self.window.state, AppState.FAILED)
            self.assertIn("Access is denied", self.window.summary_view.toPlainText())
            self.assertFalse(self.window.generate_button.isEnabled())
            self.assertFalse(self.window.preview_button.isEnabled())
            self.assertFalse(self.window.screenshot_button.isEnabled())
            self.assertFalse(self.window.export_button.isEnabled())

    def test_no_red_marked_3d_or_unity_controls(self) -> None:
        text = " ".join(widget.text() for widget in self.window.findChildren(__import__("PySide6.QtWidgets").QtWidgets.QLabel))
        for forbidden in ("Render Target Resolution", "Animation Frame", "Segmentation Asset", "3D avatar"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
