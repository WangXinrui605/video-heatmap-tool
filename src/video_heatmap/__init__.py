"""Video heatmap prototype package."""

from .detectors import create_face_detector
from .region_heatmap import (
    RegionHeatmapAccumulator,
    normalized_heatmap,
    overlay_region_heatmap,
    region_to_screen,
    screen_to_region,
)
from .regions import RegionDetection, select_target_detection

__all__ = [
    "RegionDetection",
    "RegionHeatmapAccumulator",
    "create_face_detector",
    "normalized_heatmap",
    "overlay_region_heatmap",
    "region_to_screen",
    "screen_to_region",
    "select_target_detection",
]
