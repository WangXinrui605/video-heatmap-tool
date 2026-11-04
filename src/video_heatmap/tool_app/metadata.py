from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from video_heatmap.gaze_loader import load_gaze_txt, summarize_gaze
from video_heatmap.video_io import require_cv2


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def read_video_metadata(path: str | Path) -> VideoMetadata:
    """Open a Scene Cam video and return the fields needed by frame validation."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Scene Cam video does not exist: {source}")
    cv2 = require_cv2()
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(
                f"Scene Cam video could not be opened: {source}. "
                "Check that this is a supported MP4, AVI, or MOV file."
            )
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise ValueError(f"Scene Cam video has invalid metadata: {source}")
    return VideoMetadata(width, height, fps, frame_count, frame_count / fps)


def read_gaze_summary(path: str | Path, video: VideoMetadata | None = None) -> dict:
    """Reuse the accepted gaze loader and filtering rules for a basic UI summary."""
    gaze = load_gaze_txt(
        path,
        video_width=video.width if video else None,
        video_height=video.height if video else None,
    )
    return summarize_gaze(gaze)


def validate_frame_range(start: int, end: int, frame_count: int) -> None:
    if not 0 <= start < end <= frame_count:
        raise ValueError(
            "Frame range must satisfy 0 <= start < end <= frame count; "
            "End Frame is exclusive."
        )
