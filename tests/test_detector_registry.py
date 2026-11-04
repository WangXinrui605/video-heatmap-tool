from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.detector_registry import (
    FACE_DETECTOR_IDS,
    get_face_detector_spec,
    validate_face_detector_config,
)


class DetectorRegistryTests(unittest.TestCase):
    def test_stable_detector_ids(self) -> None:
        self.assertEqual(FACE_DETECTOR_IDS, ("opencv-yunet", "opencv-dnn-ssd"))

    def test_config_defaults_and_required_assets(self) -> None:
        config = validate_face_detector_config(
            "opencv-dnn-ssd", {"model_path": "model.pb", "config_path": "graph.pbtxt"}
        )
        self.assertEqual(config["confidence_threshold"], 0.7)
        self.assertEqual(config["min_size"], 30)
        self.assertEqual(get_face_detector_spec("opencv-yunet").model_format.split()[0], "YuNet")

    def test_unknown_provider_keys_and_missing_assets_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown face detector"):
            validate_face_detector_config("unknown", {})
        with self.assertRaisesRegex(ValueError, "Missing required config"):
            validate_face_detector_config("opencv-yunet", {})
        with self.assertRaisesRegex(ValueError, "Unknown config keys"):
            validate_face_detector_config(
                "opencv-yunet", {"model_path": "model.onnx", "arbitrary": True}
            )


if __name__ == "__main__":
    unittest.main()
