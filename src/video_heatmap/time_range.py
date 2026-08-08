from __future__ import annotations

from dataclasses import dataclass
import math

from .video_io import VideoInfo


@dataclass(frozen=True)
class SelectedRange:
    start_frame: int
    end_frame_exclusive: int
    start_time_seconds: float
    end_time_seconds: float

    @property
    def frame_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame

    def as_dict(self) -> dict[str, int | float]:
        return {
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "last_output_frame": self.end_frame_exclusive - 1,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
        }


def resolve_selected_range(
    video_info: VideoInfo,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> SelectedRange:
    """Resolve optional frame/time arguments to one clamped frame range.

    Frame end values are exclusive. Time end values are inclusive: the frame
    whose timestamp is at or immediately before end_time is retained.
    """
    if video_info.fps <= 0:
        raise ValueError("Video FPS must be positive to select a time range.")
    if video_info.frame_count <= 0:
        raise ValueError("Video frame count must be positive.")
    if start_frame is not None and start_time is not None:
        raise ValueError("Use either start_frame or start_time, not both.")
    if end_frame is not None and end_time is not None:
        raise ValueError("Use either end_frame or end_time, not both.")
    if start_time is not None and start_time < 0:
        raise ValueError("start_time cannot be negative.")
    if end_time is not None and end_time < 0:
        raise ValueError("end_time cannot be negative.")
    if start_frame is not None and start_frame < 0:
        raise ValueError("start_frame cannot be negative.")
    if end_frame is not None and end_frame < 0:
        raise ValueError("end_frame cannot be negative.")

    if start_time is not None:
        resolved_start = int(math.ceil(start_time * video_info.fps - 1e-9))
    else:
        resolved_start = 0 if start_frame is None else int(start_frame)

    if end_time is not None:
        resolved_end = int(math.floor(end_time * video_info.fps + 1e-9)) + 1
    else:
        resolved_end = video_info.frame_count if end_frame is None else int(end_frame)

    resolved_start = max(0, min(resolved_start, video_info.frame_count))
    resolved_end = max(0, min(resolved_end, video_info.frame_count))
    if resolved_start >= resolved_end:
        raise ValueError("Selected video range is empty or reversed.")

    return SelectedRange(
        start_frame=resolved_start,
        end_frame_exclusive=resolved_end,
        start_time_seconds=resolved_start / video_info.fps,
        end_time_seconds=(resolved_end - 1) / video_info.fps,
    )
