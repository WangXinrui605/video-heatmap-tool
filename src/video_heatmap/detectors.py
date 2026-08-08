from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .regions import RegionDetection, clip_detection


class RegionDetector(Protocol):
    def detect(self, frame) -> list[RegionDetection]:
        """Return clipped regions in full-resolution frame pixel coordinates."""


class OpenCVYuNetFaceDetector:
    """CPU face detector backed by OpenCV FaceDetectorYN and a YuNet model."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        confidence_threshold: float = 0.8,
        min_size: int = 30,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if min_size <= 0:
            raise ValueError("min_size must be positive")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for the YuNet face detector") from exc

        resolved_path = Path(model_path)
        if not resolved_path.is_file():
            raise ValueError(f"YuNet model file does not exist: {resolved_path}")
        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError("This OpenCV build does not provide FaceDetectorYN")
        detector = cv2.FaceDetectorYN_create(
            str(resolved_path),
            "",
            (320, 320),
            confidence_threshold,
            nms_threshold,
            top_k,
        )

        self._cv2 = cv2
        self._detector = detector
        self.confidence_threshold = confidence_threshold
        self.min_size = min_size
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.model_path = resolved_path

    def detect(self, frame) -> list[RegionDetection]:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame must be a non-empty image array")
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame must be a non-empty image array")

        if len(frame.shape) == 2:
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_GRAY2BGR)
        self._detector.setInputSize((frame_width, frame_height))
        _status, faces = self._detector.detect(frame)
        if faces is None:
            return []

        detections: list[RegionDetection] = []
        for face in faces:
            x, y, width, height = (float(value) for value in face[:4])
            confidence = float(face[-1])
            if confidence < self.confidence_threshold:
                continue
            clipped = clip_detection(
                RegionDetection(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                    label="face",
                ),
                frame_width,
                frame_height,
                min_size=float(self.min_size),
            )
            if clipped is not None:
                detections.append(clipped)
        return detections


def create_face_detector(name: str = "opencv-yunet", **kwargs) -> RegionDetector:
    if name == "opencv-yunet":
        return OpenCVYuNetFaceDetector(**kwargs)
    raise ValueError(f"Unknown face detector: {name}")
