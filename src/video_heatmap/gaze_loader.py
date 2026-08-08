from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_GAZE_X_COL = "Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze X"
DEFAULT_GAZE_Y_COL = "Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze Y"


@dataclass(frozen=True)
class GazeData:
    source_path: Path
    dataframe: pd.DataFrame
    x_col: str
    y_col: str

    @property
    def total_rows(self) -> int:
        return len(self.dataframe)

    @property
    def valid_rows(self) -> int:
        return int(self.dataframe["valid"].sum())

    @property
    def invalid_rows(self) -> int:
        return self.total_rows - self.valid_rows

    @property
    def timed_valid_rows(self) -> int:
        return int((self.dataframe["valid"] & self.dataframe["time_valid"]).sum())


def _pick_column(columns: list[str], preferred: str, fallback_suffix: str) -> Optional[str]:
    if preferred in columns:
        return preferred
    exact_suffix_matches = [col for col in columns if col.endswith(fallback_suffix)]
    if exact_suffix_matches:
        return exact_suffix_matches[0]
    loose_matches = [col for col in columns if "Gaze" in col and col.endswith(fallback_suffix[-1])]
    return loose_matches[0] if loose_matches else None


def load_gaze_txt(
    path: str | Path,
    video_width: Optional[int] = None,
    video_height: Optional[int] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
) -> GazeData:
    """Load Dikablis tab-separated gaze data and mark valid gaze rows."""
    source_path = Path(path)
    df = pd.read_csv(source_path, sep="\t", low_memory=False)
    columns = list(df.columns)

    resolved_x = x_col or _pick_column(columns, DEFAULT_GAZE_X_COL, "Gaze X")
    resolved_y = y_col or _pick_column(columns, DEFAULT_GAZE_Y_COL, "Gaze Y")
    if not resolved_x or not resolved_y:
        raise ValueError(
            "Could not find gaze X/Y columns. "
            f"Available columns include: {columns[:20]}"
        )

    df = df.copy()
    df["gaze_x"] = pd.to_numeric(df[resolved_x], errors="coerce")
    df["gaze_y"] = pd.to_numeric(df[resolved_y], errors="coerce")
    if "rec_time" in df.columns:
        df["time_seconds"] = df["rec_time"].map(parse_rec_time_seconds)
    else:
        df["time_seconds"] = np.nan

    missing = df["gaze_x"].isna() | df["gaze_y"].isna()
    non_finite = ~missing & (
        ~np.isfinite(df["gaze_x"]) | ~np.isfinite(df["gaze_y"])
    )
    zero_pair = ~missing & ~non_finite & (df["gaze_x"] == 0) & (df["gaze_y"] == 0)
    out_of_bounds = np.zeros(len(df), dtype=bool)
    if video_width is not None:
        out_of_bounds |= (df["gaze_x"] < 0) | (df["gaze_x"] >= video_width)
    if video_height is not None:
        out_of_bounds |= (df["gaze_y"] < 0) | (df["gaze_y"] >= video_height)
    out_of_bounds &= ~(missing | non_finite | zero_pair)

    valid = ~(missing | non_finite | zero_pair | out_of_bounds)
    # In the demo data, (0,0) appears when the reference video shows no gaze marker.
    df["valid"] = valid
    df["time_valid"] = np.isfinite(df["time_seconds"])
    df["invalid_reason"] = np.select(
        [missing, non_finite, zero_pair, out_of_bounds],
        ["missing_or_non_numeric", "non_finite", "zero_pair", "out_of_bounds"],
        default="valid",
    )
    # This is the source TXT row position, not a video-frame assignment.
    df["frame_index"] = np.arange(len(df), dtype=int)

    return GazeData(source_path=source_path, dataframe=df, x_col=resolved_x, y_col=resolved_y)


def parse_rec_time_seconds(value) -> float:
    """Parse Dikablis rec_time values such as HH:MM:SS.mmm into seconds."""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 3:
        return float("nan")
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return float("nan")
    return hours * 3600.0 + minutes * 60.0 + seconds


def summarize_gaze(gaze: GazeData) -> dict:
    valid_df = gaze.dataframe[gaze.dataframe["valid"]]
    invalid_reason_counts = {
        str(reason): int(count)
        for reason, count in gaze.dataframe["invalid_reason"].value_counts().items()
        if reason != "valid"
    }
    summary = {
        "source_path": str(gaze.source_path),
        "x_col": gaze.x_col,
        "y_col": gaze.y_col,
        "total_rows": gaze.total_rows,
        "valid_rows": gaze.valid_rows,
        "invalid_rows": gaze.invalid_rows,
        "timed_valid_rows": gaze.timed_valid_rows,
        "valid_rows_without_rec_time": gaze.valid_rows - gaze.timed_valid_rows,
        "invalid_reason_counts": invalid_reason_counts,
    }
    if not valid_df.empty:
        summary.update(
            {
                "x_min": float(valid_df["gaze_x"].min()),
                "x_max": float(valid_df["gaze_x"].max()),
                "y_min": float(valid_df["gaze_y"].min()),
                "y_max": float(valid_df["gaze_y"].max()),
            }
        )
    if "time_seconds" in gaze.dataframe.columns:
        valid_time = gaze.dataframe["time_seconds"].dropna()
        if not valid_time.empty:
            summary.update(
                {
                    "time_start_seconds": float(valid_time.min()),
                    "time_end_seconds": float(valid_time.max()),
                    "time_duration_seconds": float(valid_time.max() - valid_time.min()),
                }
            )
    return summary
