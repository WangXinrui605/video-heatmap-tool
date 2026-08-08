from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.region_heatmap import (
    RegionHeatmapAccumulator,
    overlay_region_heatmap,
    region_to_screen,
    screen_to_region,
)
from video_heatmap.regions import RegionDetection


def region(label: str = "face") -> RegionDetection:
    return RegionDetection(10.0, 20.0, 100.0, 200.0, 0.9, label)


class RegionCoordinateTests(unittest.TestCase):
    def test_box_centre_maps_to_half(self) -> None:
        self.assertEqual(screen_to_region((60.0, 120.0), region()), (0.5, 0.5))

    def test_box_boundaries_map_to_zero_and_one(self) -> None:
        self.assertEqual(screen_to_region((10.0, 20.0), region()), (0.0, 0.0))
        self.assertEqual(screen_to_region((110.0, 220.0), region()), (1.0, 1.0))

    def test_outside_point_is_rejected_not_clamped(self) -> None:
        self.assertIsNone(screen_to_region((9.999, 120.0), region()))
        self.assertIsNone(screen_to_region((60.0, 220.001), region()))

    def test_round_trip(self) -> None:
        box = region()
        point = (42.25, 178.75)
        mapped = screen_to_region(point, box)
        self.assertIsNotNone(mapped)
        restored = region_to_screen(mapped, box)  # type: ignore[arg-type]
        self.assertAlmostEqual(restored[0], point[0], places=10)
        self.assertAlmostEqual(restored[1], point[1], places=10)


class RegionAccumulatorTests(unittest.TestCase):
    def test_non_face_region_uses_same_core(self) -> None:
        body = region("body")
        point = screen_to_region((60.0, 120.0), body)
        self.assertEqual(point, (0.5, 0.5))
        accumulator = RegionHeatmapAccumulator(33, 33, radius=3)
        accumulator.add(point, frame_index=0)  # type: ignore[arg-type]
        self.assertGreater(float(accumulator.canvas[16, 16]), 0.0)

    def test_rejected_gaze_does_not_change_canvas(self) -> None:
        accumulator = RegionHeatmapAccumulator(32, 32, radius=3)
        before = accumulator.canvas.copy()
        mapped = screen_to_region((500.0, 500.0), region())
        if mapped is not None:
            accumulator.add(mapped, frame_index=0)
        np.testing.assert_array_equal(accumulator.canvas, before)
        self.assertEqual(accumulator.total_accepted, 0)

    def test_window_history_expires_old_samples(self) -> None:
        accumulator = RegionHeatmapAccumulator(
            33, 33, radius=2, history_mode="window", window_frames=2
        )
        accumulator.add((0.25, 0.25), frame_index=0)
        accumulator.add((0.75, 0.75), frame_index=1)
        old_value = float(accumulator.canvas[8, 8])
        self.assertGreater(old_value, 0.0)
        accumulator.advance(2)
        self.assertAlmostEqual(float(accumulator.canvas[8, 8]), 0.0, places=6)
        self.assertEqual(accumulator.active_sample_count, 1)
        self.assertEqual(accumulator.total_accepted, 2)

    def test_overlay_changes_only_clipped_region(self) -> None:
        frame = np.full((20, 30, 3), 80, dtype=np.uint8)
        canvas = np.zeros((8, 8), dtype=np.float32)
        canvas[4, 4] = 1.0
        box = RegionDetection(-5.0, 5.0, 15.0, 10.0, 1.0, "object")
        output = overlay_region_heatmap(frame, canvas, box, opacity=1.0)
        np.testing.assert_array_equal(output[:5], frame[:5])
        np.testing.assert_array_equal(output[15:], frame[15:])
        np.testing.assert_array_equal(output[:, 10:], frame[:, 10:])
        self.assertTrue(np.any(output[5:15, :10] != frame[5:15, :10]))


if __name__ == "__main__":
    unittest.main()
