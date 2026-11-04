from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import QApplication

from video_heatmap.tool_app.config import Method
from video_heatmap.tool_app.method_panels import FIRST_RELEASE_GUI_METHODS, METHOD_DETAILS, MethodPanels


class MethodPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_switches_to_correct_method_specific_controls(self) -> None:
        panels = MethodPanels()
        panels.set_method(Method.FACE)
        self.assertIs(panels.currentWidget(), panels.face)
        self.assertAlmostEqual(panels.face.confidence.value(), 0.8)
        self.assertAlmostEqual(panels.face.freeze.value(), 0.5)
        self.assertEqual(panels.face.provider.count(), 2)
        self.assertEqual(panels.face.config().detector_id, "opencv-yunet")
        panels.face.provider.setCurrentIndex(1)
        self.assertEqual(panels.face.config().detector_id, "opencv-dnn-ssd")
        self.assertTrue(panels.face.detector_config.isVisibleTo(panels.face))
        panels.set_method(Method.FLOW)
        self.assertIs(panels.currentWidget(), panels.flow)
        self.assertEqual(panels.flow.event_policy.currentText(), "candidate-only")
        self.assertEqual(panels.flow.recovery_frames.value(), 3)
        self.assertFalse(hasattr(panels, "marker"))

    def test_first_release_gui_has_exactly_three_methods_and_no_marker_controls(self) -> None:
        self.assertEqual(FIRST_RELEASE_GUI_METHODS, (Method.BASELINE, Method.FACE, Method.FLOW))
        self.assertNotIn(Method.MARKER, METHOD_DETAILS)
        panels = MethodPanels()
        labels = " ".join(
            widget.text()
            for widget in panels.findChildren(__import__("PySide6.QtWidgets").QtWidgets.QLabel)
        )
        self.assertNotIn("Participant", labels)
        self.assertNotIn("Marker name", labels)

    def test_semantics_keep_cumulative_and_recent_window_distinct(self) -> None:
        self.assertIn("Cumulative", METHOD_DETAILS[Method.BASELINE]["history"])
        self.assertIn("Cumulative", METHOD_DETAILS[Method.FACE]["history"])
        self.assertIn("Recent window", METHOD_DETAILS[Method.FLOW]["history"])
        all_text = " ".join(
            str(value) for details in METHOD_DETAILS.values() for value in details.values()
        ).lower()
        self.assertNotIn("shot_boundary", all_text)


if __name__ == "__main__":
    unittest.main()
