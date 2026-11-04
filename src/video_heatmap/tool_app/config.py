from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from video_heatmap.rendering import HeatmapGradient, HeatScale, HeatScaleMode
from video_heatmap.detector_registry import validate_face_detector_config


class Method(str, Enum):
    BASELINE = "baseline"
    FACE = "face"
    FLOW = "flow"
    MARKER = "marker"


@dataclass(frozen=True)
class FaceConfig:
    model: Path
    detector_confidence: float = 0.8
    max_freeze_seconds: float = 0.5
    detector_id: str = "opencv-yunet"
    config_path: Path | None = None


@dataclass(frozen=True)
class FlowConfig:
    window_seconds: float = 5.0
    event_policy: str = "candidate-only"
    recovery_stable_frames: int = 3
    auto_event_enabled: bool = True
    manual_boundaries: Path | None = None
    event_annotations: Path | None = None


@dataclass(frozen=True)
class MarkerConfig:
    participant: str = "P2"
    marker: str = "Brasilia"


@dataclass(frozen=True)
class RunConfig:
    method: Method
    gaze_path: Path
    video_path: Path
    output_dir: Path
    start_frame: int
    end_frame: int
    brush_radius: float
    opacity: float = 0.55
    output_update_interval: int = 1
    gradient: HeatmapGradient = HeatmapGradient.JET
    heat_scale: HeatScale | None = None
    face: FaceConfig | None = None
    flow: FlowConfig | None = None
    marker: MarkerConfig | None = None

    @property
    def effective_heat_scale(self) -> HeatScale:
        if self.heat_scale is not None:
            return self.heat_scale
        # The optical-flow renderer historically used a fixed raw maximum of 20.
        if self.method is Method.FLOW:
            return HeatScale(HeatScaleMode.FIXED, 20.0)
        return HeatScale(HeatScaleMode.AUTO)

    def validate(self, *, check_paths: bool = True) -> None:
        if check_paths:
            for label, path in (("gaze", self.gaze_path), ("video", self.video_path)):
                if not path.is_file():
                    raise ValueError(f"{label} file does not exist: {path}")
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("Frame range must satisfy 0 <= start_frame < end_frame (end exclusive).")
        if self.brush_radius <= 0:
            raise ValueError("Brush radius must be positive.")
        if not 0 <= self.opacity <= 1:
            raise ValueError("Opacity must be between 0 and 1.")
        if self.output_update_interval < 1:
            raise ValueError("Output Update Interval must be at least one.")
        _ = self.effective_heat_scale

        if self.method is Method.FACE:
            if self.face is None:
                raise ValueError("Face method requires FaceConfig.")
            provider_config: dict[str, object] = {
                "model_path": str(self.face.model),
                "confidence_threshold": self.face.detector_confidence,
            }
            if self.face.config_path is not None:
                provider_config["config_path"] = str(self.face.config_path)
            validate_face_detector_config(self.face.detector_id, provider_config)
            if check_paths and not self.face.model.is_file():
                raise ValueError(f"Face model does not exist: {self.face.model}")
            if check_paths and self.face.config_path is not None and not self.face.config_path.is_file():
                raise ValueError(f"Face detector config does not exist: {self.face.config_path}")
            if not 0 <= self.face.detector_confidence <= 1:
                raise ValueError("Detector confidence must be between 0 and 1.")
            if self.face.max_freeze_seconds < 0:
                raise ValueError("Maximum freeze seconds must not be negative.")
        elif self.face is not None:
            raise ValueError("FaceConfig is only valid for face method.")

        if self.method is Method.FLOW:
            if self.flow is None:
                raise ValueError("Flow method requires FlowConfig.")
            if self.flow.window_seconds <= 0:
                raise ValueError("Flow window seconds must be positive.")
            if self.flow.event_policy not in {"candidate-only", "flow-loss"}:
                raise ValueError("Flow event policy must be candidate-only or flow-loss.")
            if self.flow.recovery_stable_frames < 1:
                raise ValueError("Recovery stable frames must be at least one.")
            if check_paths:
                for label, path in (
                    ("manual boundaries", self.flow.manual_boundaries),
                    ("event annotations", self.flow.event_annotations),
                ):
                    if path is not None and not path.is_file():
                        raise ValueError(f"{label} file does not exist: {path}")
        elif self.flow is not None:
            raise ValueError("FlowConfig is only valid for flow method.")

        if self.method is Method.MARKER:
            if self.marker is None:
                raise ValueError("Marker method requires MarkerConfig.")
            if self.marker.participant != "P2":
                raise ValueError("Marker: current validated research-reproduction scope: P2 data only.")
        elif self.marker is not None:
            raise ValueError("MarkerConfig is only valid for marker method.")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["method"] = self.method.value
        data["gradient"] = self.gradient.value
        data["heat_scale"] = self.effective_heat_scale.as_dict()
        for key in ("gaze_path", "video_path", "output_dir"):
            data[key] = str(data[key])
        for section in ("face", "flow"):
            if data[section]:
                for key, value in data[section].items():
                    if isinstance(value, Path):
                        data[section][key] = str(value)
        return {"schema_version": "video_heatmap_run_config.v1", **data}
