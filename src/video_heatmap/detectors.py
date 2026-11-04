from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .regions import RegionDetection, clip_detection


class RegionDetector(Protocol):
    def detect(self, frame) -> list[RegionDetection]:
        """Return clipped regions in full-resolution frame pixel coordinates."""


def _validate_common_config(*, confidence_threshold: float, min_size: int) -> None:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if min_size <= 0:
        raise ValueError("min_size must be positive")


def _normalise_face_box(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    confidence: float,
    frame_width: int,
    frame_height: int,
    min_size: int,
) -> tuple[RegionDetection | None, str | None, bool]:
    """Normalise one provider box and report rejection/clipping diagnostics."""

    raw = RegionDetection(x, y, width, height, confidence, "face")
    clipped = clip_detection(raw, frame_width, frame_height, min_size=float(min_size))
    if clipped is None:
        from math import isfinite

        values = (x, y, width, height, confidence)
        if not all(isfinite(value) for value in values) or width <= 0 or height <= 0:
            return None, "invalid", False
        return None, "min_size_or_outside", False
    was_clipped = (
        clipped.x != raw.x
        or clipped.y != raw.y
        or clipped.width != raw.width
        or clipped.height != raw.height
    )
    return clipped, None, was_clipped


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
        _validate_common_config(
            confidence_threshold=confidence_threshold, min_size=min_size
        )
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
        self.last_diagnostics = {
            "raw_boxes": 0,
            "confidence_rejected": 0,
            "invalid_rejected": 0,
            "min_size_or_outside_rejected": 0,
            "clipped_boxes": 0,
        }

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
        diagnostics = {
            "raw_boxes": 0,
            "confidence_rejected": 0,
            "invalid_rejected": 0,
            "min_size_or_outside_rejected": 0,
            "clipped_boxes": 0,
        }
        if faces is None:
            self.last_diagnostics = diagnostics
            return []

        detections: list[RegionDetection] = []
        for face in faces:
            diagnostics["raw_boxes"] += 1
            x, y, width, height = (float(value) for value in face[:4])
            confidence = float(face[-1])
            if confidence < self.confidence_threshold:
                diagnostics["confidence_rejected"] += 1
                continue
            clipped, rejection, was_clipped = _normalise_face_box(
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=confidence,
                frame_width=frame_width,
                frame_height=frame_height,
                min_size=self.min_size,
            )
            if clipped is not None:
                detections.append(clipped)
                diagnostics["clipped_boxes"] += int(was_clipped)
            elif rejection is not None:
                diagnostics[f"{rejection}_rejected"] += 1
        self.last_diagnostics = diagnostics
        return detections


class OpenCVDnnSsdFaceDetector:
    """CPU OpenCV-DNN face detector using the fixed TensorFlow SSD graph."""

    INPUT_SIZE = (300, 300)
    MEAN_BGR = (104.0, 177.0, 123.0)

    def __init__(
        self,
        *,
        model_path: str | Path,
        config_path: str | Path,
        confidence_threshold: float = 0.7,
        min_size: int = 30,
    ) -> None:
        _validate_common_config(
            confidence_threshold=confidence_threshold, min_size=min_size
        )
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for the DNN SSD face detector") from exc

        resolved_model = Path(model_path)
        resolved_config = Path(config_path)
        if not resolved_model.is_file():
            raise ValueError(f"DNN SSD model file does not exist: {resolved_model}")
        if not resolved_config.is_file():
            raise ValueError(f"DNN SSD config file does not exist: {resolved_config}")
        if not hasattr(cv2.dnn, "readNetFromTensorflow"):
            raise RuntimeError("This OpenCV build cannot load TensorFlow DNN graphs")

        self._cv2 = cv2
        self._net = cv2.dnn.readNetFromTensorflow(
            str(resolved_model), str(resolved_config)
        )
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model_path = resolved_model
        self.config_path = resolved_config
        self.confidence_threshold = confidence_threshold
        self.min_size = min_size
        self.last_diagnostics = {
            "raw_boxes": 0,
            "confidence_rejected": 0,
            "invalid_rejected": 0,
            "min_size_or_outside_rejected": 0,
            "clipped_boxes": 0,
        }

    def detect(self, frame) -> list[RegionDetection]:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame must be a non-empty image array")
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame must be a non-empty image array")
        if len(frame.shape) == 2:
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_GRAY2BGR)

        blob = self._cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=self.INPUT_SIZE,
            mean=self.MEAN_BGR,
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        output = self._net.forward()
        rows = output.reshape(-1, 7) if output is not None else []
        diagnostics = {
            "raw_boxes": 0,
            "confidence_rejected": 0,
            "invalid_rejected": 0,
            "min_size_or_outside_rejected": 0,
            "clipped_boxes": 0,
        }
        detections: list[RegionDetection] = []
        for row in rows:
            confidence = float(row[2])
            if confidence < self.confidence_threshold:
                diagnostics["confidence_rejected"] += 1
                continue
            diagnostics["raw_boxes"] += 1
            x0 = float(row[3]) * frame_width
            y0 = float(row[4]) * frame_height
            x1 = float(row[5]) * frame_width
            y1 = float(row[6]) * frame_height
            clipped, rejection, was_clipped = _normalise_face_box(
                x=x0,
                y=y0,
                width=x1 - x0,
                height=y1 - y0,
                confidence=confidence,
                frame_width=frame_width,
                frame_height=frame_height,
                min_size=self.min_size,
            )
            if clipped is not None:
                detections.append(clipped)
                diagnostics["clipped_boxes"] += int(was_clipped)
            elif rejection is not None:
                diagnostics[f"{rejection}_rejected"] += 1
        self.last_diagnostics = diagnostics
        return detections


def create_face_detector(name: str = "opencv-yunet", **kwargs) -> RegionDetector:
    if name == "opencv-yunet":
        return OpenCVYuNetFaceDetector(**kwargs)
    if name == "opencv-dnn-ssd":
        return OpenCVDnnSsdFaceDetector(**kwargs)
    raise ValueError(f"Unknown face detector: {name}")
