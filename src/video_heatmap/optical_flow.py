from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .video_io import VideoInfo, require_cv2


O2_SCHEMA_VERSION = "optical_flow_event_semantics.v2"
AUTOMATIC_EVENT_LABEL = "visual_discontinuity_candidate"
REVIEWED_EVENT_LABELS = {
    "unreviewed",
    "continuous_non_cut",
    "confirmed_hard_cut",
    "uncertain",
}
EVENT_POLICIES = {"candidate-only", "flow-loss"}
EVENT_ACTIONS = {"no_reset", "flow_tracking_lost", "confirmed_hard_cut_reset"}


@dataclass(frozen=True)
class FrameRange:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class FarnebackConfig:
    scale: float = 0.125
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 21
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    def validate(self) -> None:
        if not 0.0 < self.scale <= 1.0:
            raise ValueError("Farneback scale must be in (0, 1].")
        if self.levels < 1 or self.winsize < 3 or self.iterations < 1:
            raise ValueError("Invalid Farneback pyramid, window, or iteration settings.")
        if self.poly_n not in (5, 7):
            raise ValueError("Farneback poly_n must be 5 or 7.")


@dataclass(frozen=True)
class FlowPair:
    forward: np.ndarray
    backward: np.ndarray
    scale_x: float
    scale_y: float
    disturbance_score: float
    appearance_change_score: float = 0.0


@dataclass(frozen=True)
class PropagationResult:
    points: np.ndarray
    valid: np.ndarray
    forward_backward_error: np.ndarray


@dataclass(frozen=True)
class EventPolicyState:
    """Minimal deterministic state for the experimental flow-loss policy."""

    loss_active: bool = False
    stable_frames: int = 0
    loss_episode_id: int | None = None
    flow_interval_id: int = 0


@dataclass(frozen=True)
class EventPolicyTransition:
    state: EventPolicyState
    action: str
    entered_loss: bool = False
    recovered: bool = False
    confirmed_hard_cut: bool = False


def resolve_frame_range(
    video_info: VideoInfo,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> FrameRange:
    if video_info.fps <= 0 or video_info.frame_count <= 0:
        raise ValueError("Video FPS and frame count must be available.")
    if start_frame is not None and start_time is not None:
        raise ValueError("Choose either start_frame or start_time, not both.")
    if end_frame is not None and end_time is not None:
        raise ValueError("Choose either end_frame or end_time, not both.")

    if start_time is not None and start_time < 0:
        raise ValueError("start_time must be non-negative.")
    if end_time is not None and end_time < 0:
        raise ValueError("end_time must be non-negative.")

    resolved_start = (
        int(start_frame)
        if start_frame is not None
        else int(np.ceil(float(start_time) * video_info.fps - 1e-9))
        if start_time is not None
        else 0
    )
    resolved_end = (
        int(end_frame)
        if end_frame is not None
        else int(np.floor(float(end_time) * video_info.fps + 1e-9)) + 1
        if end_time is not None
        else video_info.frame_count
    )
    resolved_start = max(0, resolved_start)
    resolved_end = min(video_info.frame_count, resolved_end)
    if resolved_start >= resolved_end:
        raise ValueError("Selected video range is empty or invalid.")

    return FrameRange(
        start_frame=resolved_start,
        end_frame=resolved_end,
        start_time=resolved_start / video_info.fps,
        end_time=(resolved_end - 1) / video_info.fps,
    )


def gaze_frame_indices(times: np.ndarray, fps: float) -> np.ndarray:
    """Map gaze timestamps to the first video frame at or after each sample."""
    values = np.asarray(times, dtype=float)
    return np.ceil(values * float(fps) - 1e-9).astype(np.int64)


def group_gaze_by_frame(
    times: np.ndarray,
    points: np.ndarray,
    fps: float,
    frame_range: FrameRange,
) -> dict[int, np.ndarray]:
    times = np.asarray(times, dtype=float)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) != len(times):
        raise ValueError("points must be an Nx2 array matching times.")
    frames = gaze_frame_indices(times, fps)
    keep = (
        np.isfinite(times)
        & np.isfinite(points).all(axis=1)
        & (frames >= frame_range.start_frame)
        & (frames < frame_range.end_frame)
    )
    grouped: dict[int, list[np.ndarray]] = {}
    for frame, point in zip(frames[keep], points[keep]):
        grouped.setdefault(int(frame), []).append(point)
    return {
        frame: np.asarray(frame_points, dtype=np.float32)
        for frame, frame_points in grouped.items()
    }


def prepare_gray(frame: np.ndarray, scale: float) -> np.ndarray:
    cv2 = require_cv2()
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError("Expected a grayscale or BGR frame.")
    if scale == 1.0:
        return gray
    width = max(2, int(round(gray.shape[1] * scale)))
    height = max(2, int(round(gray.shape[0] * scale)))
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def _farneback(previous: np.ndarray, current: np.ndarray, config: FarnebackConfig) -> np.ndarray:
    cv2 = require_cv2()
    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        config.pyr_scale,
        config.levels,
        config.winsize,
        config.iterations,
        config.poly_n,
        config.poly_sigma,
        config.flags,
    )


def flow_disturbance_score(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    forward_flow: np.ndarray,
) -> float:
    """Return normalized photometric residual after flow compensation."""
    cv2 = require_cv2()
    height, width = previous_gray.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = xx + forward_flow[..., 0]
    map_y = yy + forward_flow[..., 1]
    warped_current = cv2.remap(
        current_gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (
        (map_x >= 0)
        & (map_x <= width - 1)
        & (map_y >= 0)
        & (map_y <= height - 1)
    )
    if not np.any(valid):
        return 1.0
    residual = np.abs(
        previous_gray.astype(np.float32) - warped_current.astype(np.float32)
    )
    return float(np.mean(residual[valid]) / 255.0)


def appearance_change_score(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
) -> float:
    """Compare global grayscale distributions with Bhattacharyya distance."""
    cv2 = require_cv2()
    previous_hist = cv2.calcHist([previous_gray], [0], None, [64], [0, 256])
    current_hist = cv2.calcHist([current_gray], [0], None, [64], [0, 256])
    cv2.normalize(previous_hist, previous_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(current_hist, current_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return float(
        cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA)
    )


def calculate_flow_pair(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    config: FarnebackConfig,
) -> FlowPair:
    config.validate()
    previous_gray = prepare_gray(previous_frame, config.scale)
    current_gray = prepare_gray(current_frame, config.scale)
    forward = _farneback(previous_gray, current_gray, config)
    backward = _farneback(current_gray, previous_gray, config)
    score = flow_disturbance_score(previous_gray, current_gray, forward)
    appearance_score = appearance_change_score(previous_gray, current_gray)
    return FlowPair(
        forward=forward,
        backward=backward,
        scale_x=forward.shape[1] / previous_frame.shape[1],
        scale_y=forward.shape[0] / previous_frame.shape[0],
        disturbance_score=score,
        appearance_change_score=appearance_score,
    )


def is_visual_discontinuity_candidate(
    flow_pair: FlowPair,
    disturbance_threshold: float,
    appearance_threshold: float,
) -> bool:
    """Return an automatic observation, not a semantic cut classification."""
    return (
        flow_pair.disturbance_score >= float(disturbance_threshold)
        and flow_pair.appearance_change_score >= float(appearance_threshold)
    )


def is_shot_boundary(
    flow_pair: FlowPair,
    disturbance_threshold: float,
    appearance_threshold: float,
) -> bool:
    """Legacy O1 compatibility alias.

    New code must call :func:`is_visual_discontinuity_candidate`; the boolean
    produced here is not evidence of a confirmed hard cut.
    """
    return is_visual_discontinuity_candidate(
        flow_pair, disturbance_threshold, appearance_threshold
    )


def advance_event_policy(
    state: EventPolicyState,
    *,
    candidate: bool,
    confirmed_hard_cut: bool,
    event_policy: str,
    recovery_stable_frames: int,
) -> EventPolicyTransition:
    """Advance candidate-only/flow-loss state without interpreting candidates as cuts."""
    if event_policy not in EVENT_POLICIES:
        raise ValueError(f"Unsupported event policy: {event_policy}")
    if recovery_stable_frames < 1:
        raise ValueError("recovery_stable_frames must be at least 1.")

    if confirmed_hard_cut:
        return EventPolicyTransition(
            state=EventPolicyState(
                loss_active=False,
                stable_frames=0,
                loss_episode_id=state.loss_episode_id,
                flow_interval_id=state.flow_interval_id,
            ),
            action="confirmed_hard_cut_reset",
            confirmed_hard_cut=True,
        )

    if event_policy == "candidate-only":
        return EventPolicyTransition(state=state, action="no_reset")

    if candidate:
        if state.loss_active:
            next_state = EventPolicyState(
                loss_active=True,
                stable_frames=0,
                loss_episode_id=state.loss_episode_id,
                flow_interval_id=state.flow_interval_id,
            )
            return EventPolicyTransition(state=next_state, action="flow_tracking_lost")
        episode_id = 0 if state.loss_episode_id is None else state.loss_episode_id + 1
        next_state = EventPolicyState(
            loss_active=True,
            stable_frames=0,
            loss_episode_id=episode_id,
            flow_interval_id=state.flow_interval_id,
        )
        return EventPolicyTransition(
            state=next_state,
            action="flow_tracking_lost",
            entered_loss=True,
        )

    if state.loss_active:
        stable_frames = state.stable_frames + 1
        if stable_frames >= recovery_stable_frames:
            next_state = EventPolicyState(
                loss_active=False,
                stable_frames=0,
                loss_episode_id=state.loss_episode_id,
                flow_interval_id=state.flow_interval_id + 1,
            )
            return EventPolicyTransition(
                state=next_state,
                action="no_reset",
                recovered=True,
            )
        next_state = EventPolicyState(
            loss_active=True,
            stable_frames=stable_frames,
            loss_episode_id=state.loss_episode_id,
            flow_interval_id=state.flow_interval_id,
        )
        return EventPolicyTransition(state=next_state, action="flow_tracking_lost")
    return EventPolicyTransition(state=state, action="no_reset")


def bilinear_sample_flow(
    flow: np.ndarray,
    points: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be Nx2.")

    x = points[:, 0] * float(scale_x)
    y = points[:, 1] * float(scale_y)
    height, width = flow.shape[:2]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= width - 1)
        & (y >= 0)
        & (y <= height - 1)
    )
    sampled = np.zeros((len(points), 2), dtype=np.float32)
    if not np.any(valid):
        return sampled, valid

    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(int)
    y0 = np.floor(yv).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (xv - x0).astype(np.float32)
    wy = (yv - y0).astype(np.float32)
    values = (
        flow[y0, x0] * ((1.0 - wx) * (1.0 - wy))[:, None]
        + flow[y0, x1] * (wx * (1.0 - wy))[:, None]
        + flow[y1, x0] * ((1.0 - wx) * wy)[:, None]
        + flow[y1, x1] * (wx * wy)[:, None]
    )
    values[:, 0] /= float(scale_x)
    values[:, 1] /= float(scale_y)
    sampled[valid] = values
    valid &= np.isfinite(sampled).all(axis=1)
    return sampled, valid


def propagate_points(
    points: np.ndarray,
    flow_pair: FlowPair,
    frame_width: int,
    frame_height: int,
    fb_threshold: float | None,
) -> PropagationResult:
    points = np.asarray(points, dtype=np.float32)
    forward, valid_forward = bilinear_sample_flow(
        flow_pair.forward, points, flow_pair.scale_x, flow_pair.scale_y
    )
    moved = points + forward
    backward, valid_backward = bilinear_sample_flow(
        flow_pair.backward, moved, flow_pair.scale_x, flow_pair.scale_y
    )
    error = np.linalg.norm(forward + backward, axis=1)
    valid = valid_forward & valid_backward & np.isfinite(error)
    valid &= (
        (moved[:, 0] >= 0)
        & (moved[:, 0] < frame_width)
        & (moved[:, 1] >= 0)
        & (moved[:, 1] < frame_height)
    )
    if fb_threshold is not None:
        valid &= error <= float(fb_threshold)
    return PropagationResult(
        points=moved.astype(np.float32),
        valid=valid,
        forward_backward_error=error.astype(np.float32),
    )


def load_manual_boundaries(path: str | Path | None) -> set[int]:
    if path is None:
        return set()
    source = Path(path)
    values: set[int] = set()
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split(",", 1)[0].strip()
        if token.lower() == "frame":
            continue
        values.add(int(token))
    return values


def load_event_annotations(path: str | Path | None) -> dict[int, dict[str, object]]:
    """Load O2 CSV/JSON annotations and validate shared field enums."""
    if path is None:
        return {}
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload.get("events", payload) if isinstance(payload, dict) else payload
    else:
        with source.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    if not isinstance(rows, list):
        raise ValueError("Event annotations must contain a list of events.")
    result: dict[int, dict[str, object]] = {}
    for raw in rows:
        row = dict(raw)
        frame = int(row["frame"])
        automatic_label = str(row.get("automatic_label", ""))
        if automatic_label != AUTOMATIC_EVENT_LABEL:
            raise ValueError(f"Invalid automatic_label at frame {frame}.")
        reviewed_label = str(row.get("reviewed_label", "unreviewed"))
        if reviewed_label not in REVIEWED_EVENT_LABELS:
            raise ValueError(f"Invalid reviewed_label at frame {frame}.")
        action = str(row.get("action", "no_reset"))
        if action not in EVENT_ACTIONS:
            raise ValueError(f"Invalid action at frame {frame}.")
        confirmed = row.get("confirmed_hard_cut", False)
        if isinstance(confirmed, str):
            confirmed = confirmed.strip().lower() in {"1", "true", "yes"}
        row["frame"] = frame
        row["time_seconds"] = float(row.get("time_seconds", 0.0))
        row["confirmed_hard_cut"] = bool(confirmed)
        if bool(confirmed) != (reviewed_label == "confirmed_hard_cut"):
            raise ValueError(
                f"confirmed_hard_cut conflicts with reviewed_label at frame {frame}."
            )
        result[frame] = row
    return result


def read_legacy_event_summary(path: str | Path) -> dict[str, object]:
    """Read O1 evidence without promoting legacy boundaries to confirmed cuts."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") == O2_SCHEMA_VERSION:
        return payload
    legacy = payload.get("boundaries", [])
    return {
        "schema_version": "legacy_o1_compat.v1",
        "legacy_unreviewed_boundary_reset": list(legacy),
        "event_counts": {
            AUTOMATIC_EVENT_LABEL: 0,
            "reviewed_continuous_non_cut": 0,
            "confirmed_hard_cut": 0,
            "uncertain_event": 0,
        },
        "source_summary": str(path),
    }


def summarize_numbers(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }
