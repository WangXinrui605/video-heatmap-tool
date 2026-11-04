from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.rendering import HeatmapGradient, resolve_heat_scale
from video_heatmap.detector_registry import (
    FACE_DETECTOR_IDS,
    get_face_detector_spec,
    validate_face_detector_config,
)
from video_heatmap.tool_app.artifacts import RunStatus, new_run_record, transition_run, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "video_heatmap_tool_run.v2"

MODE_INFO = {
    "baseline": {
        "label": "Screen-space cumulative baseline",
        "history": "cumulative from the selected interval start",
        "status": "baseline",
        "limitations": [
            "Uses fixed screen coordinates and does not follow a face or object.",
            "Its cumulative history is not equivalent to optical-flow recent-window history.",
        ],
    },
    "face": {
        "label": "Face-relative cumulative (detector-only main method)",
        "history": "cumulative canonical memory for the declared target identity",
        "status": "main method",
        "limitations": [
            "Uses unique TXT source-row assignment; nearest-frame matches are display-only.",
            "Requires exactly one detected face under the single-conversation-partner identity policy.",
            "A face box is not a facial-landmark coordinate system or accuracy ground truth.",
            "Tracking loss retains canonical memory, briefly freezes display, then hides stale overlay.",
        ],
    },
    "flow": {
        "label": "Optical-flow recent-window (experimental)",
        "history": "recent window of still-propagatable pixel trajectories",
        "status": "experimental",
        "limitations": [
            "Does not assign gaze semantically to a face, body, or object.",
            "Automatic visual-discontinuity candidates do not prove hard cuts and do not reset by default.",
            "window_expired is normal expiry; flow_or_bounds is an early trajectory termination.",
            "This recent-window output is not equivalent to cumulative baseline or face memory.",
        ],
    },
    "marker": {
        "label": "Marker-local research reproduction (validated P2 data only)",
        "history": "cumulative marker-local prototype history",
        "status": "legacy research-reproduction entry",
        "limitations": [
            "current validated research-reproduction scope: P2 data only.",
            "This evidence/data scope is not a claim that the algorithm is theoretically participant-name limited.",
            "A clothing marker is a local planar prototype, not participant-wide motion compensation.",
            "Marker-local evidence does not establish face, head, or full-frame accuracy.",
        ],
    },
}


def existing_file(value: str, option: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{option} file does not exist: {path}")
    return path


def validate_common(args: argparse.Namespace) -> None:
    existing_file(args.gaze, "--gaze")
    existing_file(args.video, "--video")
    if args.start_frame is not None and args.start_time is not None:
        raise ValueError("Use either --start-frame or --start-time, not both.")
    if args.end_frame is not None and args.end_time is not None:
        raise ValueError("Use either --end-frame or --end-time, not both.")
    for name in ("start_frame", "end_frame", "start_time", "end_time"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must not be negative.")
    if args.opacity < 0 or args.opacity > 1:
        raise ValueError("--opacity must be between 0 and 1.")
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be at least one.")


def range_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for name in ("start_frame", "end_frame", "start_time", "end_time"):
        value = getattr(args, name)
        if value is not None:
            result.extend([f"--{name.replace('_', '-')}", str(value)])
    return result


def command(script: str, *parts: Any) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), *(str(part) for part in parts)]


def build_commands(args: argparse.Namespace, out_dir: Path) -> list[list[str]]:
    common = ["--gaze", args.gaze, "--video", args.video]
    gradient = getattr(args, "gradient", "jet")
    heat_scale_mode = getattr(args, "heat_scale_mode", None) or (
        "fixed" if args.mode == "flow" else "auto"
    )
    heat_scale_maximum = getattr(args, "heat_scale_maximum", None)
    if args.mode == "flow" and heat_scale_mode == "fixed" and heat_scale_maximum is None:
        heat_scale_maximum = 20.0
    rendering = ["--gradient", gradient, "--heat-scale-mode", heat_scale_mode]
    if heat_scale_maximum is not None:
        rendering.extend(["--heat-scale-maximum", heat_scale_maximum])
    selected_range = range_args(args)
    if args.mode == "baseline":
        static_dir, dynamic_dir = out_dir / "static", out_dir / "dynamic"
        return [
            command("generate_final_heatmap.py", *common, "--out-dir", static_dir,
                    *selected_range, "--brush-size", args.brush_size, "--opacity", args.opacity, *rendering),
            command("generate_heatmap_video.py", *common, "--out-dir", dynamic_dir,
                    *selected_range, "--brush-size", args.brush_size, "--opacity", args.opacity,
                    "--frame-skip", args.frame_skip, *rendering),
        ]
    if args.mode == "face":
        detector_provider = getattr(args, "detector_provider", "opencv-yunet")
        parts: list[Any] = [
            *common, "--detector-provider", detector_provider, "--model", args.model,
            "--out-dir", out_dir, *selected_range, "--brush-radius", args.brush_radius,
            "--opacity", args.opacity, "--frame-skip", args.frame_skip,
            "--history-mode", "cumulative", "--confidence-threshold", args.detector_confidence,
            "--max-freeze-seconds", args.max_freeze_seconds, "--evidence-purpose", "coverage",
            *rendering,
        ]
        detector_config = getattr(args, "detector_config", None)
        if detector_config:
            parts.extend(["--detector-config", detector_config])
        return [command("generate_face_relative_heatmap.py", *parts)]
    if args.mode == "flow":
        parts: list[Any] = [
            *common, "--out-dir", out_dir, *selected_range, "--history-mode", "window",
            "--window-seconds", args.window_seconds, "--brush-size", args.brush_size,
            "--opacity", args.opacity, "--frame-skip", args.frame_skip,
            "--event-policy", args.event_policy, "--recovery-stable-frames", args.recovery_stable_frames,
            *rendering,
        ]
        if args.no_auto_event:
            parts.append("--no-auto-event")
        if args.manual_boundaries:
            parts.extend(["--manual-boundaries", args.manual_boundaries])
        if args.event_annotations:
            parts.extend(["--event-annotations", args.event_annotations])
        return [command("generate_optical_flow_video.py", *parts)]
    return [command(
        "generate_marker_heatmap_video.py", *common, "--out-dir", out_dir,
        "--marker", args.marker, "--start-frame", args.start_frame or 0,
        "--end-frame", args.end_frame if args.end_frame is not None else 300,
        "--brush-radius-marker", args.brush_radius_marker, "--opacity", args.opacity,
        "--frame-skip", args.frame_skip,
        *rendering,
    )]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_face_summary(out_dir: Path) -> dict[str, Any]:
    summary = read_json(out_dir / "face_relative_summary.json")
    assignment = summary.get("accumulation_sample_assignment", {})
    duplicate = assignment.get("duplicate_consumption_check", {})
    counts = summary.get("counts", {})
    required = (
        "selected_valid_source_rows", "assigned_unique_source_rows",
        "accepted_inside_face_unique_samples", "rejected_outside_face_unique_samples",
        "no_detection_unique_samples", "identity_uncertain_unique_samples",
    )
    missing = [key for key in required if key not in counts]
    if assignment.get("schema_version") != "gaze_assignment.v1" or missing:
        raise RuntimeError(
            "Face output does not use the required unique-source-row schema; "
            f"missing fields: {missing or 'assignment schema_version'}"
        )
    if not duplicate.get("passed", False):
        raise RuntimeError("Face output failed the duplicate-consumption check.")
    partition = sum(counts[key] for key in required[2:])
    if partition != counts["assigned_unique_source_rows"]:
        raise RuntimeError("Face output failed the unique-source-row coverage partition.")
    return summary


def collect_core_summaries(mode: str, out_dir: Path) -> list[dict[str, Any]]:
    if mode == "baseline":
        paths = [out_dir / "static" / "final_heatmap_summary.json",
                 out_dir / "dynamic" / "heatmap_development_summary.json"]
        return [read_json(path) for path in paths]
    if mode == "face":
        return [validate_face_summary(out_dir)]
    if mode == "flow":
        summary = read_json(out_dir / "optical_flow_video_summary.json")
        if summary.get("schema_version") != "optical_flow_event_semantics.v2":
            raise RuntimeError("Optical-flow output does not use the current event schema.")
        if summary.get("history_mode") != "window":
            raise RuntimeError("Integrated optical-flow mode must use recent-window history.")
        return [summary]
    summaries = sorted(out_dir.glob("*_marker_heatmap_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError("Marker core did not produce exactly one summary.")
    return [read_json(summaries[0])]


def add_common(
    parser: argparse.ArgumentParser,
    *,
    brush_name: str = "brush-size",
    brush_type: type = int,
    brush_default: int | float = 40,
) -> None:
    parser.add_argument("--gaze", required=True, help="D-Lab gaze TXT file")
    parser.add_argument("--video", required=True, help="Scene Cam video")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int, help="Exclusive frame end")
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float, help="Inclusive time end in seconds")
    brush_options = ["--brush-radius"]
    legacy_brush_option = f"--{brush_name}"
    if legacy_brush_option not in brush_options:
        brush_options.append(legacy_brush_option)
    parser.add_argument(
        *brush_options,
        dest=brush_name.replace("-", "_"),
        type=brush_type,
        default=brush_default,
        help=(
            "Heat-kernel radius. --brush-radius is the unified spelling; "
            f"{legacy_brush_option} remains a compatibility alias."
        ),
    )
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--gradient", choices=[item.value for item in HeatmapGradient], default="jet")
    parser.add_argument("--heat-scale-mode", choices=("auto", "fixed"), default=None)
    parser.add_argument("--heat-scale-maximum", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the delegated commands")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entry point that delegates to the accepted heatmap pipelines.",
        epilog=(
            "Histories are method-specific: baseline and face are cumulative; optical flow is a "
            "recent-window experiment. They are not interchangeable. Marker remains a legacy "
            "research-reproduction entry with current validated scope: P2 data only."
        ),
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    baseline = sub.add_parser("baseline", help=MODE_INFO["baseline"]["label"])
    add_common(baseline)
    baseline.set_defaults(history_mode="cumulative")

    face = sub.add_parser("face", help=MODE_INFO["face"]["label"])
    add_common(face, brush_name="brush-radius")
    face.add_argument("--detector-provider", choices=FACE_DETECTOR_IDS, default="opencv-yunet")
    face.add_argument("--model", required=True, help="Provider-specific detector model")
    face.add_argument("--detector-config", help="Provider config required by opencv-dnn-ssd")
    face.add_argument(
        "--detector-confidence",
        type=float,
        default=None,
        help="Detection threshold; omitted uses the selected provider registry default.",
    )
    face.add_argument("--max-freeze-seconds", type=float, default=0.5)
    face.set_defaults(history_mode="cumulative")

    flow = sub.add_parser("flow", help=MODE_INFO["flow"]["label"])
    add_common(flow)
    flow.add_argument("--window-seconds", type=float, default=5.0)
    flow.add_argument("--event-policy", choices=("candidate-only", "flow-loss"), default="candidate-only")
    flow.add_argument("--recovery-stable-frames", type=int, default=3)
    flow.add_argument("--no-auto-event", action="store_true")
    flow.add_argument("--manual-boundaries")
    flow.add_argument("--event-annotations")
    flow.set_defaults(history_mode="window")

    marker = sub.add_parser("marker", help=MODE_INFO["marker"]["label"])
    add_common(
        marker,
        brush_name="brush-radius-marker",
        brush_type=float,
        brush_default=1.0,
    )
    marker.add_argument(
        "--participant",
        required=True,
        choices=("P2",),
        help="current validated research-reproduction scope: P2 data only",
    )
    marker.add_argument("--marker", default="Brasilia")
    marker.set_defaults(history_mode="cumulative")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.heat_scale_mode is None:
        args.heat_scale_mode = "fixed" if args.mode == "flow" else "auto"
    if args.mode == "flow" and args.heat_scale_mode == "fixed" and args.heat_scale_maximum is None:
        args.heat_scale_maximum = 20.0
    if args.mode == "face" and args.detector_confidence is None:
        defaults = get_face_detector_spec(args.detector_provider).defaults()
        args.detector_confidence = float(defaults["confidence_threshold"])
    try:
        validate_common(args)
        if args.mode == "face":
            existing_file(args.model, "--model")
            provider_config: dict[str, object] = {
                "model_path": args.model,
                "confidence_threshold": args.detector_confidence,
            }
            if args.detector_config:
                existing_file(args.detector_config, "--detector-config")
                provider_config["config_path"] = args.detector_config
            validate_face_detector_config(args.detector_provider, provider_config)
            if not 0 <= args.detector_confidence <= 1:
                raise ValueError("--detector-confidence must be between 0 and 1.")
            if args.max_freeze_seconds < 0:
                raise ValueError("--max-freeze-seconds must not be negative.")
        if args.mode == "flow":
            if args.window_seconds <= 0:
                raise ValueError("--window-seconds must be positive.")
            for name in ("manual_boundaries", "event_annotations"):
                value = getattr(args, name)
                if value:
                    existing_file(value, f"--{name.replace('_', '-')}")
        if args.mode == "marker" and (args.start_time is not None or args.end_time is not None):
            raise ValueError("Marker experimental mode currently supports frame ranges only.")
        resolve_heat_scale(args.heat_scale_mode, args.heat_scale_maximum)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = Path(args.out_dir)
    commands = build_commands(args, out_dir)
    run_config = {
        "mode": args.mode,
        "method": MODE_INFO[args.mode],
        "inputs": {"gaze": args.gaze, "video": args.video},
        "history_mode": args.history_mode,
        "commands": commands,
        "parameters": {key: value for key, value in vars(args).items() if key not in {"gaze", "video", "out_dir", "dry_run"}},
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps({"schema_version": "video_heatmap_run_config.v1", **run_config}, indent=2, ensure_ascii=False))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    marker_name = getattr(args, "marker", "Brasilia")
    run = new_run_record(
        __import__("video_heatmap.tool_app.config", fromlist=["Method"]).Method(args.mode),
        run_config,
        out_dir,
        marker=marker_name,
    )
    summary_path = out_dir / "tool_run_summary.json"
    progress_path = out_dir / "progress_events.jsonl"
    run["progress_contract"] = {
        "schema_version": "video_heatmap_progress.v1",
        "events_path": str(progress_path),
        "work_semantics": "processed input frames / selected total input frames",
        "frame_skip_invariant": True,
        "legacy_no_event_fallback": "indeterminate",
    }
    write_json_atomic(summary_path, run)
    started = perf_counter()
    delegated_environment = os.environ.copy()
    delegated_environment["VIDEO_HEATMAP_PROGRESS_RUN_ID"] = run["run_id"]
    delegated_environment["VIDEO_HEATMAP_PROGRESS_METHOD"] = args.mode
    delegated_environment["VIDEO_HEATMAP_PROGRESS_PATH"] = str(progress_path.resolve())
    try:
        for delegated in commands:
            subprocess.run(delegated, cwd=ROOT, check=True, env=delegated_environment)
        run["core_summaries"] = collect_core_summaries(args.mode, out_dir)
        run = transition_run(run, RunStatus.COMPLETED, out_dir, marker=marker_name)
    except subprocess.CalledProcessError as exc:
        hint = (
            " Check that the YuNet model is readable and compatible with the installed OpenCV runtime."
            if args.mode == "face"
            else " Review the delegated pipeline error printed above."
        )
        error = f"Delegated pipeline exited with code {exc.returncode}: {exc.cmd}.{hint}"
        run = transition_run(run, RunStatus.FAILED, out_dir, marker=marker_name, error=error)
        run["runtime_seconds"] = perf_counter() - started
        write_json_atomic(summary_path, run)
        raise SystemExit(run["error"]) from exc
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        run = transition_run(run, RunStatus.FAILED, out_dir, marker=marker_name, error=str(exc))
        run["runtime_seconds"] = perf_counter() - started
        write_json_atomic(summary_path, run)
        raise SystemExit(f"Integrated output validation failed: {exc}") from exc
    finally:
        run["runtime_seconds"] = perf_counter() - started
    write_json_atomic(summary_path, run)
    print(json.dumps({"status": run["status"], "summary": str(summary_path), "method": MODE_INFO[args.mode]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
