from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.rendering import HeatmapGradient, HeatScale, HeatScaleMode
from video_heatmap.tool_app.adapter import command_tokens
from video_heatmap.tool_app.config import FaceConfig, FlowConfig, MarkerConfig, Method, RunConfig


class RunConfigTests(unittest.TestCase):
    def make_files(self, folder: Path) -> tuple[Path, Path, Path]:
        paths = tuple(folder / name for name in ("gaze.txt", "scene.mp4", "model.onnx"))
        for path in paths:
            path.touch()
        return paths  # type: ignore[return-value]

    def base(self, method: Method, folder: Path, **extra) -> RunConfig:
        gaze, video, _ = self.make_files(folder)
        return RunConfig(method, gaze, video, folder / "out", 0, 300, 12, **extra)

    def test_method_specific_validation_and_marker_p2_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            with self.assertRaisesRegex(ValueError, "FaceConfig"):
                self.base(Method.FACE, folder).validate()
            with self.assertRaisesRegex(ValueError, "research-reproduction scope: P2 data only"):
                self.base(Method.MARKER, folder, marker=MarkerConfig("P3")).validate()

    def test_backend_four_method_command_mapping_remains_tokenized(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            gaze, video, model = self.make_files(folder)
            configs = [
                RunConfig(Method.BASELINE, gaze, video, folder / "b", 0, 10, 40),
                RunConfig(Method.FACE, gaze, video, folder / "f", 0, 10, 12, face=FaceConfig(model)),
                RunConfig(Method.FLOW, gaze, video, folder / "o", 0, 10, 40, flow=FlowConfig()),
                RunConfig(Method.MARKER, gaze, video, folder / "m", 0, 10, 1.5, marker=MarkerConfig()),
            ]
            commands = [command_tokens(config, python_executable="python") for config in configs]
        self.assertEqual([tokens[2] for tokens in commands], ["baseline", "face", "flow", "marker"])
        self.assertIn("--model", commands[1])
        self.assertEqual(commands[2][commands[2].index("--event-policy") + 1], "candidate-only")
        self.assertEqual(commands[3][commands[3].index("--participant") + 1], "P2")
        self.assertNotIn("shot_boundary", " ".join(commands[2]))

    def test_default_rendering_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            baseline = self.base(Method.BASELINE, folder)
            flow = self.base(Method.FLOW, folder, flow=FlowConfig())
        self.assertEqual(baseline.gradient, HeatmapGradient.JET)
        self.assertEqual(baseline.effective_heat_scale.mode, HeatScaleMode.AUTO)
        self.assertEqual(flow.effective_heat_scale, HeatScale(HeatScaleMode.FIXED, 20.0))

    def test_output_update_interval_maps_only_to_frame_skip(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            tokens = command_tokens(
                self.base(Method.BASELINE, folder, output_update_interval=5),
                python_executable="python",
            )
        self.assertEqual(tokens[tokens.index("--frame-skip") + 1], "5")
        self.assertEqual(tokens.count("5"), 1)

    def test_two_face_providers_map_provider_specific_config(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            gaze, video, model = self.make_files(folder)
            pbtxt = folder / "opencv.pbtxt"
            pbtxt.touch()
            config = RunConfig(
                Method.FACE, gaze, video, folder / "f", 0, 10, 12,
                face=FaceConfig(model, 0.7, 0.5, "opencv-dnn-ssd", pbtxt),
            )
            tokens = command_tokens(config, python_executable="python")
            config.validate()
        self.assertEqual(tokens[tokens.index("--detector-provider") + 1], "opencv-dnn-ssd")
        self.assertEqual(tokens[tokens.index("--detector-config") + 1], str(pbtxt))

    def test_provider_specific_validation_rejects_missing_config_and_unknown_provider(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            gaze, video, model = self.make_files(folder)
            for provider in ("opencv-dnn-ssd", "unknown-provider"):
                with self.subTest(provider=provider), self.assertRaises(ValueError):
                    RunConfig(
                        Method.FACE, gaze, video, folder / "f", 0, 10, 12,
                        face=FaceConfig(model, 0.7, 0.5, provider),
                    ).validate()


if __name__ == "__main__":
    unittest.main()
