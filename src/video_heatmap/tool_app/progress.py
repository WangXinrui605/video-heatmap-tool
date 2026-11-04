from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


PROGRESS_SCHEMA_VERSION = "video_heatmap_progress.v1"
PROGRESS_PREFIX = "VIDEO_HEATMAP_PROGRESS "
PROGRESS_STAGES = frozenset({"preparing", "processing", "finalising"})


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    method: str
    stage: str
    processed_input_frames: int
    total_input_frames: int
    percentage: float
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    timestamp: str
    sequence: int

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": PROGRESS_SCHEMA_VERSION, **self.__dict__}


def parse_progress_line(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROGRESS_PREFIX):
        return None
    payload = json.loads(line[len(PROGRESS_PREFIX):])
    if not isinstance(payload, dict):
        raise ValueError("Progress payload must be a JSON object.")
    return payload


class ProgressValidator:
    """Validate one run's monotonic progress stream; unknown fields are additive."""

    def __init__(self, *, run_id: str | None = None, method: str | None = None) -> None:
        self.run_id = run_id
        self.method = method
        self.last_sequence = -1
        self.last_processed = -1
        self.last_percentage = -1.0
        self.total: int | None = None

    def accept(self, payload: dict[str, Any]) -> dict[str, Any]:
        required = {
            "schema_version", "run_id", "method", "stage", "processed_input_frames",
            "total_input_frames", "percentage", "elapsed_seconds", "timestamp", "sequence",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Progress event is missing: {', '.join(missing)}")
        if payload["schema_version"] != PROGRESS_SCHEMA_VERSION:
            raise ValueError("Unsupported progress schema version.")
        if self.run_id is not None and payload["run_id"] != self.run_id:
            raise ValueError("Progress run_id does not match the current run.")
        if self.run_id is None:
            self.run_id = str(payload["run_id"])
        if self.method is not None and payload["method"] != self.method:
            raise ValueError("Progress method does not match the current run.")
        if payload["stage"] not in PROGRESS_STAGES:
            raise ValueError("Unsupported progress stage.")
        for key in ("processed_input_frames", "total_input_frames", "sequence"):
            if isinstance(payload[key], bool) or not isinstance(payload[key], int):
                raise ValueError(f"{key} must be an integer.")
        processed = payload["processed_input_frames"]
        total = payload["total_input_frames"]
        sequence = payload["sequence"]
        percentage = float(payload["percentage"])
        elapsed = float(payload["elapsed_seconds"])
        if total < 1 or processed < 0 or processed > total:
            raise ValueError("Progress frame counts are outside their valid range.")
        if self.total is not None and total != self.total:
            raise ValueError("Progress total_input_frames changed during the run.")
        if sequence <= self.last_sequence or processed < self.last_processed:
            raise ValueError("Progress event is out of order.")
        expected = processed * 100.0 / total
        if not math.isfinite(percentage) or abs(percentage - expected) > 0.011:
            raise ValueError("Progress percentage does not match processed/total frames.")
        if percentage + 1e-9 < self.last_percentage or not 0 <= percentage <= 100:
            raise ValueError("Progress percentage is not monotonic or in range.")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Progress elapsed_seconds must be finite and non-negative.")
        eta = payload.get("estimated_remaining_seconds")
        if eta is not None and (not isinstance(eta, (int, float)) or not math.isfinite(float(eta)) or eta < 0):
            raise ValueError("Progress ETA must be finite and non-negative when present.")
        self.total = total
        self.last_sequence = sequence
        self.last_processed = processed
        self.last_percentage = percentage
        return payload


class ProgressEmitter:
    def __init__(
        self,
        run_id: str,
        method: str,
        total_input_frames: int,
        *,
        clock: Callable[[], float] = perf_counter,
        sink: Callable[[str], None] = print,
        events_path: Path | None = None,
    ) -> None:
        if total_input_frames < 1:
            raise ValueError("total_input_frames must be positive.")
        self.run_id = run_id
        self.method = method
        self.total = total_input_frames
        self.clock = clock
        self.sink = sink
        self.events_path = events_path
        self.started = clock()
        self.sequence = 0

    @classmethod
    def from_environment(cls, total_input_frames: int) -> "ProgressEmitter | None":
        run_id = os.environ.get("VIDEO_HEATMAP_PROGRESS_RUN_ID")
        method = os.environ.get("VIDEO_HEATMAP_PROGRESS_METHOD")
        if not run_id or not method:
            return None
        path_value = os.environ.get("VIDEO_HEATMAP_PROGRESS_PATH")
        return cls(
            run_id,
            method,
            total_input_frames,
            events_path=Path(path_value) if path_value else None,
        )

    def emit(self, processed: int, stage: str = "processing") -> dict[str, Any]:
        elapsed = max(0.0, self.clock() - self.started)
        eta = None
        # A small warm-up plus positive measured duration avoids invented precision.
        if stage == "processing" and processed >= 3 and elapsed >= 0.25 and processed < self.total:
            rate = processed / elapsed
            if rate > 0 and math.isfinite(rate):
                eta = (self.total - processed) / rate
        event = ProgressEvent(
            run_id=self.run_id,
            method=self.method,
            stage=stage,
            processed_input_frames=processed,
            total_input_frames=self.total,
            percentage=processed * 100.0 / self.total,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=eta,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=self.sequence,
        ).as_dict()
        self.sequence += 1
        line = PROGRESS_PREFIX + json.dumps(event, separators=(",", ":"))
        if self.events_path is not None:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        if self.sink is print:
            print(line, flush=True)
        else:
            self.sink(line)
        return event
