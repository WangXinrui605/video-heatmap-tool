from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.detectors import OpenCVYuNetFaceDetector, create_face_detector


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "face_detection"
            / "models"
            / "face_detection_yunet_2026may.onnx"
        )

    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            OpenCVYuNetFaceDetector(model_path="missing.onnx", confidence_threshold=1.1)
        with self.assertRaises(ValueError):
            OpenCVYuNetFaceDetector(model_path="missing.onnx", min_size=0)
        with self.assertRaises(ValueError):
            OpenCVYuNetFaceDetector(model_path="missing.onnx", nms_threshold=1.1)

    def test_unknown_detector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown face detector"):
            create_face_detector("unknown")

    def test_missing_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "model file does not exist"):
            OpenCVYuNetFaceDetector(model_path="missing.onnx")

    def test_blank_frame_has_no_faces(self) -> None:
        if not self.model_path.is_file():
            self.skipTest("YuNet model asset is not present")
        import numpy as np

        detector = OpenCVYuNetFaceDetector(model_path=self.model_path)
        self.assertEqual(
            detector.detect(np.zeros((120, 160, 3), dtype=np.uint8)),
            [],
        )

    def test_invalid_frame_is_rejected(self) -> None:
        if not self.model_path.is_file():
            self.skipTest("YuNet model asset is not present")
        detector = OpenCVYuNetFaceDetector(model_path=self.model_path)
        with self.assertRaisesRegex(ValueError, "non-empty image"):
            detector.detect(None)


if __name__ == "__main__":
    unittest.main()
