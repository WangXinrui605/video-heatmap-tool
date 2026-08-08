from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_loader import load_gaze_txt, parse_rec_time_seconds, summarize_gaze


class GazeLoaderTests(unittest.TestCase):
    def write_gaze(self, rows: list[dict]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "gaze.txt"
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
        return path

    def test_parses_rec_time(self) -> None:
        self.assertAlmostEqual(parse_rec_time_seconds("01:02:03.250"), 3723.25)
        self.assertTrue(pd.isna(parse_rec_time_seconds("not-a-time")))

    def test_classifies_invalid_coordinates_and_missing_time(self) -> None:
        x_col = "example Gaze X"
        y_col = "example Gaze Y"
        path = self.write_gaze(
            [
                {x_col: 10, y_col: 20, "rec_time": "00:00:00.100"},
                {x_col: 0, y_col: 0, "rec_time": "00:00:00.200"},
                {x_col: "", y_col: 10, "rec_time": "00:00:00.300"},
                {x_col: -1, y_col: 10, "rec_time": "00:00:00.400"},
                {x_col: 10, y_col: 100, "rec_time": "00:00:00.500"},
                {x_col: 20, y_col: 30, "rec_time": ""},
            ]
        )

        gaze = load_gaze_txt(path, video_width=100, video_height=100)
        summary = summarize_gaze(gaze)

        self.assertEqual(gaze.total_rows, 6)
        self.assertEqual(gaze.valid_rows, 2)
        self.assertEqual(gaze.timed_valid_rows, 1)
        self.assertEqual(summary["valid_rows_without_rec_time"], 1)
        self.assertEqual(
            summary["invalid_reason_counts"],
            {
                "out_of_bounds": 2,
                "zero_pair": 1,
                "missing_or_non_numeric": 1,
            },
        )

    def test_one_zero_axis_is_not_the_missing_gaze_sentinel(self) -> None:
        x_col = "example Gaze X"
        y_col = "example Gaze Y"
        path = self.write_gaze(
            [{x_col: 0, y_col: 25, "rec_time": "00:00:00.100"}]
        )

        gaze = load_gaze_txt(path, video_width=100, video_height=100)

        self.assertEqual(gaze.valid_rows, 1)


if __name__ == "__main__":
    unittest.main()
