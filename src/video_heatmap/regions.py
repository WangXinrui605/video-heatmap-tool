from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class RegionDetection:
    """A detector result in full-resolution frame pixel coordinates."""

    x: float
    y: float
    width: float
    height: float
    confidence: float
    label: str
    track_id: str | None = None

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


def clip_detection(
    detection: RegionDetection,
    frame_width: int,
    frame_height: int,
    *,
    min_size: float = 1.0,
) -> RegionDetection | None:
    """Clip a detection to a frame, returning ``None`` for invalid boxes."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if min_size <= 0:
        raise ValueError("min_size must be positive")

    values = (
        detection.x,
        detection.y,
        detection.width,
        detection.height,
        detection.confidence,
    )
    if not all(isfinite(value) for value in values):
        return None
    if detection.width <= 0 or detection.height <= 0:
        return None
    if not detection.label:
        return None

    x0 = max(0.0, min(float(frame_width), detection.x))
    y0 = max(0.0, min(float(frame_height), detection.y))
    x1 = max(0.0, min(float(frame_width), detection.x + detection.width))
    y1 = max(0.0, min(float(frame_height), detection.y + detection.height))
    width = x1 - x0
    height = y1 - y0
    if width < min_size or height < min_size:
        return None

    return replace(
        detection,
        x=x0,
        y=y0,
        width=width,
        height=height,
        confidence=max(0.0, min(1.0, detection.confidence)),
    )


def select_target_detection(
    detections: Iterable[RegionDetection],
    frame_width: int,
    frame_height: int,
    *,
    label: str = "face",
) -> RegionDetection | None:
    """Select one target deterministically: largest box, then nearest centre."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    frame_centre = (frame_width / 2.0, frame_height / 2.0)
    candidates = [detection for detection in detections if detection.label == label]
    if not candidates:
        return None

    def ranking(detection: RegionDetection) -> tuple[float, float, float, float, float]:
        centre_x, centre_y = detection.center
        distance_squared = (
            (centre_x - frame_centre[0]) ** 2 + (centre_y - frame_centre[1]) ** 2
        )
        return (
            -detection.area,
            distance_squared,
            -detection.confidence,
            detection.x,
            detection.y,
        )

    return min(candidates, key=ranking)
