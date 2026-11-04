from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Iterable, Mapping


ASSIGNMENT_SCHEMA_VERSION = "gaze_assignment.v1"


@dataclass(frozen=True)
class SourceGazeSample:
    """One valid, timed TXT source row.

    ``source_row_id`` is source identity. Equal timestamps or coordinates never
    make two rows the same sample.
    """

    source_row_id: int
    time_seconds: float
    gaze_x: float
    gaze_y: float

    def __post_init__(self) -> None:
        if self.source_row_id < 0:
            raise ValueError("source_row_id must not be negative")
        if not all(
            isfinite(value)
            for value in (self.time_seconds, self.gaze_x, self.gaze_y)
        ):
            raise ValueError("sample time and coordinates must be finite")


@dataclass(frozen=True)
class FrameAssignment:
    sample: SourceGazeSample
    frame_index: int
    frame_time_seconds: float


@dataclass(frozen=True)
class AssignmentPlan:
    assignments: tuple[FrameAssignment, ...]
    by_frame: dict[int, tuple[FrameAssignment, ...]]
    output_frame_indices: tuple[int, ...]
    maximum_samples_per_frame: int

    @property
    def assigned_source_row_ids(self) -> tuple[int, ...]:
        return tuple(item.sample.source_row_id for item in self.assignments)


def first_frame_not_earlier_than(sample_time_seconds: float, fps: float) -> int:
    """Return the first video frame whose timestamp is >= the sample time."""

    if not isfinite(sample_time_seconds) or sample_time_seconds < 0:
        raise ValueError("sample_time_seconds must be finite and non-negative")
    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    return max(0, int(ceil(sample_time_seconds * fps - 1e-9)))


def assign_source_rows_to_frames(
    samples: Iterable[SourceGazeSample],
    *,
    fps: float,
    start_frame: int,
    end_frame_exclusive: int,
    frame_skip: int = 1,
) -> AssignmentPlan:
    """Assign every selected sample exactly once using baseline time semantics.

    ``frame_skip`` controls only the listed output frames. It deliberately does
    not alter assignments or accumulation frames.
    """

    if start_frame < 0 or end_frame_exclusive <= start_frame:
        raise ValueError("frame range must be non-empty and non-negative")
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least one")

    seen: set[int] = set()
    ordered: list[FrameAssignment] = []
    for sample in samples:
        if sample.source_row_id in seen:
            raise ValueError(f"duplicate source_row_id: {sample.source_row_id}")
        seen.add(sample.source_row_id)
        frame_index = first_frame_not_earlier_than(sample.time_seconds, fps)
        if not start_frame <= frame_index < end_frame_exclusive:
            raise ValueError(
                f"sample {sample.source_row_id} maps outside selected frame range"
            )
        ordered.append(
            FrameAssignment(
                sample=sample,
                frame_index=frame_index,
                frame_time_seconds=frame_index / fps,
            )
        )

    ordered.sort(
        key=lambda item: (
            item.frame_index,
            item.sample.time_seconds,
            item.sample.source_row_id,
        )
    )
    grouped: dict[int, list[FrameAssignment]] = {}
    for item in ordered:
        grouped.setdefault(item.frame_index, []).append(item)
    by_frame = {frame: tuple(items) for frame, items in grouped.items()}
    maximum = max((len(items) for items in by_frame.values()), default=0)
    output_frames = tuple(range(start_frame, end_frame_exclusive, frame_skip))
    return AssignmentPlan(tuple(ordered), by_frame, output_frames, maximum)


def duplicate_consumption_summary(consumption_counts: dict[int, int]) -> dict[str, object]:
    duplicates = {
        str(source_row_id): count
        for source_row_id, count in sorted(consumption_counts.items())
        if count > 1
    }
    return {
        "passed": not duplicates,
        "duplicate_source_row_count": len(duplicates),
        "duplicate_consumption_excess": sum(count - 1 for count in duplicates.values()),
        "source_rows_consumed_more_than_once": duplicates,
    }


def audit_consistency_summary(
    records: Iterable[Mapping[str, object]], *, canonical_accepted_count: int
) -> dict[str, object]:
    """Validate that audit, coverage and canonical memory use one source set."""

    assigned_classes = {
        "accepted_inside_face",
        "outside_face",
        "no_detection",
        "identity_uncertain",
    }
    assigned = [
        record
        for record in records
        if record.get("classification") in assigned_classes
    ]
    ids = [int(record["source_row_id"]) for record in assigned]
    class_counts = Counter(str(record["classification"]) for record in assigned)
    consumption_is_one = all(
        int(record.get("consumption_count", 0)) == 1 for record in assigned
    )
    unique_ids = len(ids) == len(set(ids))
    accepted = class_counts["accepted_inside_face"]
    return {
        "assigned_audit_record_count": len(assigned),
        "unique_assigned_source_row_count": len(set(ids)),
        "classification_counts": dict(sorted(class_counts.items())),
        "coverage_partition_passed": sum(class_counts.values()) == len(assigned),
        "source_row_identity_unique": unique_ids,
        "all_assigned_consumption_counts_equal_one": consumption_is_one,
        "canonical_accepted_count": canonical_accepted_count,
        "canonical_matches_accepted_audit": accepted == canonical_accepted_count,
        "static_dynamic_canonical_audit_source_set_passed": (
            unique_ids and consumption_is_one and accepted == canonical_accepted_count
        ),
    }
