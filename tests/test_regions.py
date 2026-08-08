from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.regions import (
    RegionDetection,
    clip_detection,
    select_target_detection,
)


class RegionTests(unittest.TestCase):
    def test_clip_detection_to_frame(self) -> None:
        result = clip_detection(
            RegionDetection(-5, 10, 30, 25, 1.2, "face"), 100, 80
        )
        self.assertEqual(
            result,
            RegionDetection(0, 10, 25, 25, 1.0, "face"),
        )

    def test_clip_rejects_non_finite_or_too_small_box(self) -> None:
        self.assertIsNone(
            clip_detection(RegionDetection(math.nan, 0, 10, 10, 0.8, "face"), 100, 80)
        )
        self.assertIsNone(
            clip_detection(RegionDetection(99, 79, 10, 10, 0.8, "face"), 100, 80, min_size=2)
        )

    def test_selects_largest_face(self) -> None:
        small_central = RegionDetection(40, 30, 20, 20, 0.99, "face")
        large_edge = RegionDetection(0, 0, 40, 40, 0.7, "face")
        self.assertIs(
            select_target_detection([small_central, large_edge], 100, 80),
            large_edge,
        )

    def test_selection_tie_uses_frame_centre_then_confidence(self) -> None:
        edge = RegionDetection(0, 0, 20, 20, 0.99, "face")
        central_low = RegionDetection(40, 30, 20, 20, 0.6, "face")
        central_high = RegionDetection(40, 30, 20, 20, 0.8, "face")
        self.assertIs(
            select_target_detection([edge, central_low, central_high], 100, 80),
            central_high,
        )

    def test_selection_ignores_other_labels_and_handles_empty(self) -> None:
        person = RegionDetection(0, 0, 50, 50, 0.9, "person")
        self.assertIsNone(select_target_detection([person], 100, 80))
        self.assertIsNone(select_target_detection([], 100, 80))


if __name__ == "__main__":
    unittest.main()
