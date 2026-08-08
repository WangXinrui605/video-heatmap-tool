from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np

from .regions import RegionDetection


RegionPoint = tuple[float, float]
ScreenPoint = tuple[float, float]


def _valid_box(box: RegionDetection) -> bool:
    values = (box.x, box.y, box.width, box.height)
    return all(isfinite(value) for value in values) and box.width > 0 and box.height > 0


def screen_to_region(gaze: ScreenPoint, box: RegionDetection) -> RegionPoint | None:
    """Map an inside-box screen point to normalized region coordinates.

    Both rectangle edges are accepted so the mathematical bottom-right boundary maps
    to ``(1, 1)``. Points outside the box are rejected, never clamped.
    """

    x, y = gaze
    if not _valid_box(box) or not (isfinite(x) and isfinite(y)):
        return None
    if not (box.x <= x <= box.x + box.width and box.y <= y <= box.y + box.height):
        return None
    return ((x - box.x) / box.width, (y - box.y) / box.height)


def region_to_screen(point: RegionPoint, box: RegionDetection) -> ScreenPoint:
    """Map normalized region coordinates back to the current screen-space box."""

    u, v = point
    if not _valid_box(box):
        raise ValueError("box must contain finite coordinates and positive dimensions")
    if not (isfinite(u) and isfinite(v)):
        raise ValueError("region coordinates must be finite")
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise ValueError("region coordinates must be within [0, 1]")
    return (box.x + u * box.width, box.y + v * box.height)


def _brush(
    width: int,
    height: int,
    point: RegionPoint,
    radius: int,
) -> tuple[slice, slice, np.ndarray]:
    u, v = point
    cx = int(round(u * (width - 1)))
    cy = int(round(v * (height - 1)))
    radius = max(1, int(radius))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance_squared = (xx - cx) ** 2 + (yy - cy) ** 2
    sigma = max(radius / 2.0, 1.0)
    values = np.exp(-distance_squared / (2.0 * sigma * sigma)).astype(np.float32)
    values[distance_squared > radius * radius] = 0.0
    return slice(y0, y1), slice(x0, x1), values


@dataclass(frozen=True)
class AccumulatedSample:
    frame_index: int
    point: RegionPoint


class RegionHeatmapAccumulator:
    """Accumulate normalized gaze on a fixed, region-generic canvas."""

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        *,
        radius: int = 12,
        history_mode: str = "cumulative",
        window_frames: int | None = None,
    ) -> None:
        if width <= 1 or height <= 1:
            raise ValueError("canvas dimensions must be greater than one")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if history_mode not in {"cumulative", "window"}:
            raise ValueError("history_mode must be 'cumulative' or 'window'")
        if history_mode == "window" and (window_frames is None or window_frames <= 0):
            raise ValueError("window history requires positive window_frames")

        self.width = int(width)
        self.height = int(height)
        self.radius = int(radius)
        self.history_mode = history_mode
        self.window_frames = int(window_frames) if window_frames is not None else None
        self.canvas = np.zeros((self.height, self.width), dtype=np.float32)
        self._samples: deque[AccumulatedSample] = deque()
        self.total_accepted = 0

    @property
    def active_sample_count(self) -> int:
        return len(self._samples)

    def _apply(self, point: RegionPoint, sign: float) -> None:
        y_slice, x_slice, values = _brush(
            self.width, self.height, point, self.radius
        )
        self.canvas[y_slice, x_slice] += values * sign

    def advance(self, frame_index: int) -> None:
        """Expire windowed samples even on frames that do not accept gaze."""

        if frame_index < 0:
            raise ValueError("frame_index must not be negative")
        if self.history_mode != "window":
            return
        assert self.window_frames is not None
        first_active_frame = frame_index - self.window_frames + 1
        while self._samples and self._samples[0].frame_index < first_active_frame:
            expired = self._samples.popleft()
            self._apply(expired.point, -1.0)
        np.maximum(self.canvas, 0.0, out=self.canvas)

    def add(self, point: RegionPoint, *, frame_index: int) -> None:
        u, v = point
        if not (isfinite(u) and isfinite(v) and 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError("region point must be finite and within [0, 1]")
        self.advance(frame_index)
        sample = AccumulatedSample(frame_index=frame_index, point=(u, v))
        self._samples.append(sample)
        self._apply(sample.point, 1.0)
        self.total_accepted += 1


def normalized_heatmap(canvas: np.ndarray) -> np.ndarray:
    """Convert a floating-point canonical canvas to uint8."""

    if canvas.ndim != 2:
        raise ValueError("canvas must be a two-dimensional array")
    maximum = float(np.nanmax(canvas)) if canvas.size else 0.0
    if not isfinite(maximum) or maximum <= 0.0:
        return np.zeros(canvas.shape, dtype=np.uint8)
    return np.clip(canvas / maximum * 255.0, 0.0, 255.0).astype(np.uint8)


def overlay_region_heatmap(
    frame: np.ndarray,
    canvas: np.ndarray,
    box: RegionDetection,
    *,
    opacity: float = 0.55,
) -> np.ndarray:
    """Resize a canonical heatmap into a box and blend only inside that box."""

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1")
    if not _valid_box(box):
        raise ValueError("box must contain finite coordinates and positive dimensions")

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to render a region heatmap") from exc

    frame_height, frame_width = frame.shape[:2]
    x0 = max(0, min(frame_width, int(round(box.x))))
    y0 = max(0, min(frame_height, int(round(box.y))))
    x1 = max(0, min(frame_width, int(round(box.x + box.width))))
    y1 = max(0, min(frame_height, int(round(box.y + box.height))))
    output = frame.copy()
    if x1 <= x0 or y1 <= y0:
        return output

    heat_uint8 = normalized_heatmap(canvas)
    resized = cv2.resize(heat_uint8, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    colours = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
    alpha = resized.astype(np.float32)[:, :, None] / 255.0 * float(opacity)
    background = output[y0:y1, x0:x1].astype(np.float32)
    blended = background * (1.0 - alpha) + colours.astype(np.float32) * alpha
    output[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return output


def accumulate_points(
    points: Iterable[RegionPoint],
    *,
    width: int = 256,
    height: int = 256,
    radius: int = 12,
) -> np.ndarray:
    """Convenience helper for a static canonical heatmap."""

    accumulator = RegionHeatmapAccumulator(width, height, radius=radius)
    for frame_index, point in enumerate(points):
        accumulator.add(point, frame_index=frame_index)
    return accumulator.canvas
