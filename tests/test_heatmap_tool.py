from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_heatmap_tool import (
    build_commands,
    get_face_detector_spec,
    make_parser,
    validate_face_summary,
)


class HeatmapToolTests(unittest.TestCase):
    def base_args(self, mode: str) -> argparse.Namespace:
        return argparse.Namespace(
            mode=mode, gaze="gaze.txt", video="scene.mp4", start_frame=0, end_frame=10,
            start_time=None, end_time=None, opacity=0.55, frame_skip=1, brush_size=40,
            brush_radius=12, model="model.onnx", detector_confidence=0.8,
            max_freeze_seconds=0.5, window_seconds=5.0, event_policy="candidate-only",
            recovery_stable_frames=3, no_auto_event=False, manual_boundaries=None,
            event_annotations=None, marker="Brasilia", brush_radius_marker=1,
        )

    def test_histories_are_fixed_per_mode(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            baseline = build_commands(self.base_args("baseline"), out)
            face = build_commands(self.base_args("face"), out)[0]
            flow = build_commands(self.base_args("flow"), out)[0]
        self.assertEqual(len(baseline), 2)
        self.assertIn("cumulative", face)
        self.assertIn("window", flow)
        self.assertNotIn("cumulative", flow)

    def test_flow_uses_o2_candidate_only_default(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            command = build_commands(self.base_args("flow"), Path(folder))[0]
        index = command.index("--event-policy")
        self.assertEqual(command[index + 1], "candidate-only")
        self.assertNotIn("shot_boundary", " ".join(command))

    def test_marker_keeps_local_float_radius(self) -> None:
        parser_args = __import__("scripts.run_heatmap_tool", fromlist=["make_parser"]).make_parser().parse_args([
            "marker", "--participant", "P2", "--gaze", "gaze.txt", "--video", "scene.mp4",
            "--out-dir", "out", "--brush-radius-marker", "1.5",
        ])
        self.assertEqual(parser_args.brush_radius_marker, 1.5)

    def test_unified_brush_radius_alias_keeps_legacy_destinations(self) -> None:
        parser = make_parser()
        common = ["--gaze", "gaze.txt", "--video", "video.mp4", "--out-dir", "out"]
        baseline = parser.parse_args(["baseline", *common, "--brush-radius", "41"])
        flow = parser.parse_args(["flow", *common, "--brush-radius", "42"])
        marker = parser.parse_args(
            ["marker", *common, "--participant", "P2", "--brush-radius", "1.5"]
        )
        self.assertEqual(baseline.brush_size, 41)
        self.assertEqual(flow.brush_size, 42)
        self.assertEqual(marker.brush_radius_marker, 1.5)

    def test_face_provider_default_confidence_comes_from_registry(self) -> None:
        parser = make_parser()
        common = [
            "--gaze", "gaze.txt", "--video", "video.mp4", "--out-dir", "out", "--model", "model",
        ]
        yunet = parser.parse_args(["face", *common, "--detector-provider", "opencv-yunet"])
        dnn = parser.parse_args(["face", *common, "--detector-provider", "opencv-dnn-ssd"])
        self.assertIsNone(yunet.detector_confidence)
        self.assertIsNone(dnn.detector_confidence)
        self.assertEqual(get_face_detector_spec("opencv-yunet").defaults()["confidence_threshold"], 0.8)
        self.assertEqual(get_face_detector_spec("opencv-dnn-ssd").defaults()["confidence_threshold"], 0.7)

    def test_marker_help_states_research_reproduction_evidence_scope(self) -> None:
        marker_parser = make_parser()._subparsers._group_actions[0].choices["marker"]
        self.assertIn(
            "current validated research-reproduction scope: P2 data only",
            " ".join(marker_parser.format_help().split()),
        )

    def test_current_non_p2_marker_research_entry_still_rejects(self) -> None:
        with self.assertRaises(SystemExit):
            make_parser().parse_args([
                "marker", "--participant", "P3", "--gaze", "gaze.txt",
                "--video", "scene.mp4", "--out-dir", "out",
            ])

    def test_face_summary_rejects_legacy_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "face_relative_summary.json"
            path.write_text(json.dumps({"counts": {}}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "G1"):
                validate_face_summary(Path(folder))

    def test_face_summary_accepts_unique_row_partition(self) -> None:
        summary = {
            "accumulation_sample_assignment": {
                "schema_version": "gaze_assignment.v1",
                "duplicate_consumption_check": {"passed": True},
            },
            "counts": {
                "selected_valid_source_rows": 4,
                "assigned_unique_source_rows": 4,
                "accepted_inside_face_unique_samples": 1,
                "rejected_outside_face_unique_samples": 2,
                "no_detection_unique_samples": 1,
                "identity_uncertain_unique_samples": 0,
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "face_relative_summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(validate_face_summary(Path(folder))["counts"]["assigned_unique_source_rows"], 4)


if __name__ == "__main__":
    unittest.main()
