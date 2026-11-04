from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceDetectorSpec:
    detector_id: str
    display_name: str
    model_format: str
    required_config_keys: tuple[str, ...]
    optional_config_defaults: tuple[tuple[str, object], ...]

    def defaults(self) -> dict[str, object]:
        return dict(self.optional_config_defaults)


_SPECS = (
    FaceDetectorSpec(
        detector_id="opencv-yunet",
        display_name="OpenCV YuNet",
        model_format="YuNet ONNX for cv2.FaceDetectorYN",
        required_config_keys=("model_path",),
        optional_config_defaults=(
            ("confidence_threshold", 0.8),
            ("min_size", 30),
            ("nms_threshold", 0.3),
            ("top_k", 5000),
        ),
    ),
    FaceDetectorSpec(
        detector_id="opencv-dnn-ssd",
        display_name="OpenCV DNN SSD (TensorFlow)",
        model_format="OpenCV TensorFlow frozen graph + matching pbtxt",
        required_config_keys=("model_path", "config_path"),
        optional_config_defaults=(
            ("confidence_threshold", 0.7),
            ("min_size", 30),
        ),
    ),
)

FACE_DETECTOR_IDS = tuple(spec.detector_id for spec in _SPECS)


def list_face_detector_specs() -> tuple[FaceDetectorSpec, ...]:
    return _SPECS


def get_face_detector_spec(detector_id: str) -> FaceDetectorSpec:
    for spec in _SPECS:
        if spec.detector_id == detector_id:
            return spec
    raise ValueError(f"Unknown face detector: {detector_id}")


def validate_face_detector_config(
    detector_id: str, config: dict[str, object]
) -> dict[str, object]:
    spec = get_face_detector_spec(detector_id)
    allowed = set(spec.required_config_keys) | set(spec.defaults())
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown config keys for {detector_id}: {', '.join(unknown)}"
        )
    missing = [key for key in spec.required_config_keys if not config.get(key)]
    if missing:
        raise ValueError(
            f"Missing required config keys for {detector_id}: {', '.join(missing)}"
        )
    resolved = spec.defaults()
    resolved.update(config)
    return resolved
