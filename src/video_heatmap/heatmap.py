from __future__ import annotations

import numpy as np

from .rendering import HeatmapGradient, colorize


def add_brush(heatmap: np.ndarray, x: float, y: float, radius: int, value: float = 1.0) -> None:
    """Add a simple circular brush to a heatmap matrix in-place."""
    height, width = heatmap.shape
    cx = int(round(x))
    cy = int(round(y))
    radius = max(1, int(radius))

    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sigma = max(radius / 2.0, 1.0)
    brush = np.exp(-dist2 / (2.0 * sigma * sigma))
    brush[dist2 > radius * radius] = 0.0
    heatmap[y0:y1, x0:x1] += brush * value


def normalize_to_uint8(heatmap: np.ndarray) -> np.ndarray:
    """Normalize heatmap to 0-255 uint8."""
    max_value = float(np.nanmax(heatmap)) if heatmap.size else 0.0
    if max_value <= 0:
        return np.zeros_like(heatmap, dtype=np.uint8)
    return np.clip((heatmap / max_value) * 255.0, 0, 255).astype(np.uint8)


def normalize_with_max(heatmap: np.ndarray, max_value: float) -> np.ndarray:
    """Map a heatmap to uint8 using a caller-provided stable maximum."""
    if max_value <= 0:
        return np.zeros_like(heatmap, dtype=np.uint8)
    return np.clip((heatmap / float(max_value)) * 255.0, 0, 255).astype(np.uint8)


def colorize_heatmap(
    heatmap_uint8: np.ndarray,
    gradient: HeatmapGradient | str = HeatmapGradient.JET,
) -> np.ndarray:
    """Apply the selected rendering-only colour map; JET remains the default."""
    return colorize(heatmap_uint8, gradient)


def overlay_heatmap(
    background: np.ndarray,
    heatmap_uint8: np.ndarray,
    opacity: float,
    gradient: HeatmapGradient | str = HeatmapGradient.JET,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend a heatmap over a BGR frame using heat-dependent alpha."""
    color_heatmap = colorize_heatmap(heatmap_uint8, gradient)
    alpha = (heatmap_uint8.astype(np.float32) / 255.0) * float(opacity)
    alpha = alpha[:, :, None]
    blended = (
        background.astype(np.float32) * (1.0 - alpha)
        + color_heatmap.astype(np.float32) * alpha
    )
    return blended.clip(0, 255).astype(np.uint8), color_heatmap


def build_final_heatmap(points: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    """Build a final cumulative heatmap from Nx2 gaze points."""
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        add_brush(heatmap, x, y, radius)
    return heatmap
