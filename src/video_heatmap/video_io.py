from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    backend: str = "unknown"


def require_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video reading/writing. Install opencv-python in the project environment."
        ) from exc


def get_video_info(path: str | Path) -> VideoInfo:
    video_path = Path(path)
    try:
        cv2 = require_cv2()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return VideoInfo(path=video_path, width=width, height=height, fps=fps, frame_count=frame_count, backend="opencv")
    except RuntimeError:
        return probe_mp4_info(video_path)


def _read_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def _iter_boxes(data: bytes, start: int, end: int):
    offset = start
    while offset + 8 <= end:
        size = _read_u32(data, offset)
        box_type = data[offset + 4 : offset + 8].decode("latin1")
        header = 8
        if size == 1:
            if offset + 16 > end:
                break
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            break
        yield box_type, offset, offset + header, offset + size
        offset += size


def _find_box(data: bytes, path: list[str], start: int = 0, end: int | None = None):
    end = len(data) if end is None else end
    if not path:
        return None
    for box_type, box_start, content_start, box_end in _iter_boxes(data, start, end):
        if box_type == path[0]:
            if len(path) == 1:
                return box_start, content_start, box_end
            return _find_box(data, path[1:], content_start, box_end)
    return None


def probe_mp4_info(path: str | Path) -> VideoInfo:
    """Read basic MP4 width/height/duration without decoding frames."""
    video_path = Path(path)
    data = video_path.read_bytes()

    tkhd = _find_box(data, ["moov", "trak", "tkhd"])
    if not tkhd:
        raise ValueError(f"Could not find MP4 tkhd metadata in {video_path}")
    _, content_start, box_end = tkhd
    version = data[content_start]
    if version == 1:
        width_offset = box_end - 8
        height_offset = box_end - 4
    else:
        width_offset = box_end - 8
        height_offset = box_end - 4
    width = _read_u32(data, width_offset) >> 16
    height = _read_u32(data, height_offset) >> 16

    fps = 0.0
    frame_count = 0
    mvhd = _find_box(data, ["moov", "mvhd"])
    if mvhd:
        _, mvhd_content, _ = mvhd
        version = data[mvhd_content]
        if version == 1:
            timescale_offset = mvhd_content + 20
            duration_offset = mvhd_content + 24
            timescale = _read_u32(data, timescale_offset)
            duration = struct.unpack_from(">Q", data, duration_offset)[0]
        else:
            timescale_offset = mvhd_content + 12
            duration_offset = mvhd_content + 16
            timescale = _read_u32(data, timescale_offset)
            duration = _read_u32(data, duration_offset)
        if timescale:
            duration_seconds = duration / timescale
            fps = 0.0
            frame_count = 0

    return VideoInfo(path=video_path, width=width, height=height, fps=fps, frame_count=frame_count, backend="mp4_metadata")
