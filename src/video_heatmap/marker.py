from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional

import numpy as np


# Most D-Lab exports use "Gaze in Marker Coordinates". In the P2 export,
# Brasilia X and Seoul Y instead use "Gaze in  Coordinates" (with the word
# "Marker" missing). Accept both forms while keeping the rest of the column
# name strict enough to avoid matching unrelated gaze columns.
GAZE_IN_MARKER_PATTERN = re.compile(
    r"Gaze in\s+(?:Marker\s+)?Coordinates_"
    r"(?P<name>.+)_Pupil\s+(?P<axis>[XY])$"
)
MARKER_CORNER_PATTERN = re.compile(
    r"Scene Cam_Markers_(?P<group>.+)_(?P<name>.+) "
    r"(?P<axis>[XY]) (?P<corner>[1-4])$"
)


@dataclass(frozen=True)
class MarkerColumns:
    name: str
    gaze_x_col: Optional[str]
    gaze_y_col: Optional[str]
    corner_cols: tuple[tuple[str, str], ...]

    @property
    def has_gaze_coordinates(self) -> bool:
        return self.gaze_x_col is not None and self.gaze_y_col is not None

    @property
    def has_all_corners(self) -> bool:
        return len(self.corner_cols) == 4


def distance_to_marker_square(points: np.ndarray) -> np.ndarray:
    """Return distance from marker-local points to the [0, 1] x [0, 1] square."""
    array = np.asarray(points, dtype=float)
    if array.shape == () or array.shape[-1] != 2:
        raise ValueError("Marker-local points must have a final dimension of 2.")
    outside = np.maximum(np.maximum(-array, array - 1.0), 0.0)
    return np.linalg.norm(outside, axis=-1)


def discover_marker_columns(columns: Iterable[str]) -> list[MarkerColumns]:
    """Find D-Lab marker-relative gaze and ordered marker-corner columns."""
    gaze_columns: dict[str, dict[str, str]] = {}
    corner_columns: dict[str, dict[int, dict[str, str]]] = {}

    for column in columns:
        gaze_match = GAZE_IN_MARKER_PATTERN.search(column)
        if gaze_match:
            name = gaze_match.group("name")
            gaze_columns.setdefault(name, {})[gaze_match.group("axis")] = column
            continue

        corner_match = MARKER_CORNER_PATTERN.search(column)
        if not corner_match:
            continue
        group = corner_match.group("group")
        name = corner_match.group("name")
        if group != name:
            continue
        corner = int(corner_match.group("corner"))
        axis = corner_match.group("axis")
        corner_columns.setdefault(name, {}).setdefault(corner, {})[axis] = column

    names = sorted(set(gaze_columns) | set(corner_columns))
    markers: list[MarkerColumns] = []
    for name in names:
        ordered_corners = []
        for corner in range(1, 5):
            axes = corner_columns.get(name, {}).get(corner, {})
            if "X" not in axes or "Y" not in axes:
                ordered_corners = []
                break
            ordered_corners.append((axes["X"], axes["Y"]))
        gaze_axes = gaze_columns.get(name, {})
        markers.append(
            MarkerColumns(
                name=name,
                gaze_x_col=gaze_axes.get("X"),
                gaze_y_col=gaze_axes.get("Y"),
                corner_cols=tuple(ordered_corners),
            )
        )
    return markers
