"""Video heatmap prototype package."""

from .detectors import create_face_detector
from .detector_registry import (
    FACE_DETECTOR_IDS,
    FaceDetectorSpec,
    get_face_detector_spec,
    list_face_detector_specs,
    validate_face_detector_config,
)
from .gaze_assignment import (
    AssignmentPlan,
    FrameAssignment,
    SourceGazeSample,
    assign_source_rows_to_frames,
    first_frame_not_earlier_than,
)
from .region_heatmap import (
    RegionHeatmapAccumulator,
    normalized_heatmap,
    overlay_region_heatmap,
    region_to_screen,
    screen_to_region,
)
from .regions import RegionDetection, select_target_detection
from .tracking_state import TrackingDecision, TrackingLifecycle, TrackingPhase

__all__ = [
    "AssignmentPlan",
    "FACE_DETECTOR_IDS",
    "FaceDetectorSpec",
    "FrameAssignment",
    "RegionDetection",
    "RegionHeatmapAccumulator",
    "SourceGazeSample",
    "TrackingDecision",
    "TrackingLifecycle",
    "TrackingPhase",
    "assign_source_rows_to_frames",
    "create_face_detector",
    "first_frame_not_earlier_than",
    "get_face_detector_spec",
    "list_face_detector_specs",
    "normalized_heatmap",
    "overlay_region_heatmap",
    "region_to_screen",
    "screen_to_region",
    "select_target_detection",
    "validate_face_detector_config",
]
