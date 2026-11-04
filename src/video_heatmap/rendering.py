from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from .video_io import require_cv2


class HeatmapGradient(str, Enum):
    JET = "jet"
    TURBO = "turbo"
    INFERNO = "inferno"
    VIRIDIS = "viridis"


class HeatScaleMode(str, Enum):
    AUTO = "auto"
    FIXED = "fixed"


@dataclass(frozen=True)
class HeatScale:
    mode: HeatScaleMode = HeatScaleMode.AUTO
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.mode is HeatScaleMode.AUTO and self.maximum is not None:
            raise ValueError("Auto heat scale must not define a maximum.")
        if self.mode is HeatScaleMode.FIXED:
            if self.maximum is None or not isfinite(self.maximum) or self.maximum <= 0:
                raise ValueError("Fixed heat scale requires a finite positive maximum.")

    def as_dict(self) -> dict[str, float | str | None]:
        return {"mode": self.mode.value, "maximum": self.maximum}


def resolve_heat_scale(mode: str, maximum: float | None) -> HeatScale:
    try:
        parsed_mode = HeatScaleMode(mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported heat scale mode: {mode}") from exc
    return HeatScale(parsed_mode, maximum)


def scale_heatmap(heatmap: np.ndarray, heat_scale: HeatScale) -> tuple[np.ndarray, float]:
    """Map raw heat to uint8 without modifying the raw array.

    Auto uses the maximum of the supplied rendering domain. Callers decide whether
    that domain is a final cumulative canvas or the current frame's active canvas.
    Fixed uses the declared raw-heat maximum and clips values above it.
    """

    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a two-dimensional array")
    if heat_scale.mode is HeatScaleMode.FIXED:
        assert heat_scale.maximum is not None
        maximum = float(heat_scale.maximum)
    else:
        maximum = float(np.nanmax(heatmap)) if heatmap.size else 0.0
        if not isfinite(maximum) or maximum < 0:
            maximum = 0.0
    if maximum <= 0:
        return np.zeros(heatmap.shape, dtype=np.uint8), maximum
    scaled = np.clip(heatmap / maximum * 255.0, 0.0, 255.0).astype(np.uint8)
    return scaled, maximum


def colorize(heatmap_uint8: np.ndarray, gradient: HeatmapGradient | str) -> np.ndarray:
    cv2 = require_cv2()
    selected = HeatmapGradient(gradient)
    colour_maps = {
        HeatmapGradient.JET: cv2.COLORMAP_JET,
        HeatmapGradient.TURBO: cv2.COLORMAP_TURBO,
        HeatmapGradient.INFERNO: cv2.COLORMAP_INFERNO,
        HeatmapGradient.VIRIDIS: cv2.COLORMAP_VIRIDIS,
    }
    return cv2.applyColorMap(heatmap_uint8, colour_maps[selected])


def rendering_summary(
    gradient: HeatmapGradient | str,
    heat_scale: HeatScale,
    *,
    resolved_maximum: float | None = None,
) -> dict[str, object]:
    return {
        "contract_version": "video_heatmap_rendering.v1",
        "gradient": HeatmapGradient(gradient).value,
        "heat_scale": heat_scale.as_dict(),
        "resolved_maximum": resolved_maximum,
        "raw_heat_invariant": True,
        "counts_and_sample_weights_invariant": True,
        "warning": (
            "Auto normalization may make small sample counts look visually strong; "
            "interpret colour together with exact sample counts."
            if heat_scale.mode is HeatScaleMode.AUTO
            else None
        ),
    }
