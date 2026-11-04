from __future__ import annotations

from pathlib import Path
import sys

from .config import Method, RunConfig


def command_tokens(config: RunConfig, *, python_executable: str | Path | None = None) -> list[str]:
    """Return the tokenized unified-CLI command consumed by QProcess."""

    config.validate()
    python = str(python_executable or sys.executable)
    root = Path(__file__).resolve().parents[3]
    scale = config.effective_heat_scale
    tokens = [
        python,
        str(root / "scripts" / "run_heatmap_tool.py"),
        config.method.value,
        "--gaze", str(config.gaze_path),
        "--video", str(config.video_path),
        "--out-dir", str(config.output_dir),
        "--start-frame", str(config.start_frame),
        "--end-frame", str(config.end_frame),
        "--opacity", str(config.opacity),
        "--frame-skip", str(config.output_update_interval),
        "--gradient", config.gradient.value,
        "--heat-scale-mode", scale.mode.value,
    ]
    if scale.maximum is not None:
        tokens.extend(["--heat-scale-maximum", str(scale.maximum)])

    if config.method in {Method.BASELINE, Method.FLOW}:
        tokens.extend(["--brush-size", str(int(config.brush_radius))])
    elif config.method is Method.FACE:
        tokens.extend(["--brush-radius", str(int(config.brush_radius))])
    else:
        tokens.extend(["--brush-radius-marker", str(config.brush_radius)])

    if config.method is Method.FACE:
        assert config.face is not None
        tokens.extend([
            "--detector-provider", config.face.detector_id,
            "--model", str(config.face.model),
            "--detector-confidence", str(config.face.detector_confidence),
            "--max-freeze-seconds", str(config.face.max_freeze_seconds),
        ])
        if config.face.config_path is not None:
            tokens.extend(["--detector-config", str(config.face.config_path)])
    elif config.method is Method.FLOW:
        assert config.flow is not None
        tokens.extend([
            "--window-seconds", str(config.flow.window_seconds),
            "--event-policy", config.flow.event_policy,
            "--recovery-stable-frames", str(config.flow.recovery_stable_frames),
        ])
        if not config.flow.auto_event_enabled:
            tokens.append("--no-auto-event")
        if config.flow.manual_boundaries is not None:
            tokens.extend(["--manual-boundaries", str(config.flow.manual_boundaries)])
        if config.flow.event_annotations is not None:
            tokens.extend(["--event-annotations", str(config.flow.event_annotations)])
    elif config.method is Method.MARKER:
        assert config.marker is not None
        tokens.extend(["--participant", config.marker.participant, "--marker", config.marker.marker])
    return tokens
