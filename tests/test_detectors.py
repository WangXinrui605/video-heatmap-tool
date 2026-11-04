from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.detectors import (
    OpenCVDnnSsdFaceDetector,
    OpenCVYuNetFaceDetector,
    _normalise_face_box,
    create_face_detector,
)


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
        cls.ssd_model_path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "F2"
            / "models"
            / "opencv_face_detector_uint8.pb"
        )
        cls.ssd_config_path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "F2"
            / "models"
            / "opencv_face_detector.pbtxt"
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

    def test_dnn_ssd_parameter_and_asset_validation(self) -> None:
        with self.assertRaises(ValueError):
            OpenCVDnnSsdFaceDetector(
                model_path="missing.pb",
                config_path="missing.pbtxt",
                confidence_threshold=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "model file does not exist"):
            OpenCVDnnSsdFaceDetector(
                model_path="missing.pb", config_path="missing.pbtxt"
            )

    def test_dnn_ssd_factory_and_blank_frame(self) -> None:
        if not self.ssd_model_path.is_file() or not self.ssd_config_path.is_file():
            self.skipTest("DNN SSD assets are not present")
        import numpy as np

        detector = create_face_detector(
            "opencv-dnn-ssd",
            model_path=self.ssd_model_path,
            config_path=self.ssd_config_path,
        )
        self.assertIsInstance(detector, OpenCVDnnSsdFaceDetector)
        self.assertEqual(detector.detect(np.zeros((120, 160, 3), dtype=np.uint8)), [])

    def test_provider_box_normalisation_clips_and_rejects(self) -> None:
        clipped, rejection, was_clipped = _normalise_face_box(
            x=-5,
            y=10,
            width=30,
            height=25,
            confidence=1.2,
            frame_width=100,
            frame_height=80,
            min_size=2,
        )
        self.assertEqual(clipped.x, 0)
        self.assertEqual(clipped.confidence, 1.0)
        self.assertIsNone(rejection)
        self.assertTrue(was_clipped)
        rejected, rejection, _ = _normalise_face_box(
            x=99,
            y=79,
            width=10,
            height=10,
            confidence=0.8,
            frame_width=100,
            frame_height=80,
            min_size=2,
        )
        self.assertIsNone(rejected)
        self.assertEqual(rejection, "min_size_or_outside")

    def test_dnn_ssd_normalises_empty_and_multiple_provider_output(self) -> None:
        import numpy as np

        class FakeNet:
            def __init__(self, output):
                self.output = output

            def setInput(self, _blob) -> None:
                pass

            def forward(self):
                return self.output

        class FakeDnn:
            @staticmethod
            def blobFromImage(*_args, **_kwargs):
                return object()

        class FakeCv2:
            dnn = FakeDnn()

        detector = OpenCVDnnSsdFaceDetector.__new__(OpenCVDnnSsdFaceDetector)
        detector._cv2 = FakeCv2()
        detector.confidence_threshold = 0.7
        detector.min_size = 10
        detector._net = FakeNet(np.zeros((1, 1, 0, 7), dtype=float))
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.assertEqual(detector.detect(frame), [])

        detector._net = FakeNet(
            np.array(
                [[[[0, 1, 0.9, -0.1, 0.1, 0.3, 0.5],
                   [0, 1, 0.8, 0.5, 0.2, 0.9, 0.8],
                   [0, 1, 0.2, 0.0, 0.0, 1.0, 1.0]]]],
                dtype=float,
            )
        )
        results = detector.detect(frame)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].x, 0)
        self.assertEqual(detector.last_diagnostics["clipped_boxes"], 1)
        self.assertEqual(detector.last_diagnostics["confidence_rejected"], 1)


if __name__ == "__main__":
    unittest.main()
