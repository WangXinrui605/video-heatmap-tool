from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.marker import discover_marker_columns


def found_or_missing(column: str | None) -> str:
    return column if column is not None else "MISSING"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect D-Lab marker-related columns in a gaze TXT file."
    )
    parser.add_argument("--gaze", required=True, help="Path to the D-Lab TXT file.")
    args = parser.parse_args()

    gaze_path = Path(args.gaze)
    if not gaze_path.is_file():
        raise SystemExit(f"Gaze TXT file not found: {gaze_path}")

    # nrows=0 reads only the header. Marker discovery does not need data rows.
    columns = pd.read_csv(gaze_path, sep="\t", nrows=0).columns
    markers = discover_marker_columns(columns)
    if not markers:
        raise SystemExit("No D-Lab marker columns were found.")

    print(f"File: {gaze_path}")
    print(f"Markers found: {len(markers)}")
    for marker in markers:
        print()
        print(marker.name)
        print(f"  gaze X: {found_or_missing(marker.gaze_x_col)}")
        print(f"  gaze Y: {found_or_missing(marker.gaze_y_col)}")
        print(f"  complete corners: {len(marker.corner_cols)}/4")
        print(f"  ready for reprojection: {marker.has_gaze_coordinates and marker.has_all_corners}")


if __name__ == "__main__":
    main()
