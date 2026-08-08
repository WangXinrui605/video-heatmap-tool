from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_marker_cross_frame import (
    distance_band,
    evaluate_direction,
    summarize_availability,
    summarize_values,
)
from video_heatmap.video_io import require_cv2


class MarkerCrossFrameSummaryTests(unittest.TestCase):
    def test_summarize_availability_partitions_every_frame(self) -> None:
        first = np.array([True, True, False, False])
        second = np.array([True, False, True, False])

        summary = summarize_availability(first, second)

        self.assertEqual(summary["both_available_frames"], 1)
        self.assertEqual(summary["first_only_frames"], 1)
        self.assertEqual(summary["second_only_frames"], 1)
        self.assertEqual(summary["neither_available_frames"], 1)
        self.assertEqual(summary["first_dropout_frames"], 2)
        self.assertEqual(summary["second_dropout_frames"], 2)

    def test_summarize_availability_rejects_different_shapes(self) -> None:
        with self.assertRaises(ValueError):
            summarize_availability(np.ones(2), np.ones(3))

    def test_distance_bands_have_explicit_boundary(self) -> None:
        self.assertEqual(distance_band(0.0, 2.0), "inside")
        self.assertEqual(distance_band(2.0, 2.0), "near")
        self.assertEqual(distance_band(2.001, 2.0), "far")

    def test_empty_value_summary_uses_null_metrics(self) -> None:
        summary = summarize_values(np.array([]), "pixels")

        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["mean_pixels"])
        self.assertIsNone(summary["max_pixels"])

    def test_cross_frame_landmark_stays_aligned_for_planar_translation(self) -> None:
        cv2 = require_cv2()
        anchor = np.array(
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[5, 3], [15, 3], [15, 13], [5, 13]],
            ],
            dtype=float,
        )
        target = np.array(
            [
                [[19, 4], [21, 4], [21, 6], [19, 6]],
                [[24, 7], [26, 7], [26, 9], [24, 9]],
            ],
            dtype=float,
        )
        marker_data = {
            "Anchor": {"corners": anchor},
            "Target": {"corners": target},
        }
        matched_rows = {
            "Anchor": np.array([0, 1]),
            "Target": np.array([0, 1]),
        }
        availability = {
            "Anchor": np.array([True, True]),
            "Target": np.array([True, True]),
        }

        records, summary = evaluate_direction(
            cv2,
            "Anchor",
            "Target",
            marker_data,
            np.array([0, 1]),
            np.array([0.0, 1.0]),
            matched_rows,
            availability,
        )

        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(summary["max_pixels"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
