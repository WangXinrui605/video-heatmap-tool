from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.heatmap import add_brush, build_final_heatmap, normalize_to_uint8


class HeatmapTests(unittest.TestCase):
    def test_empty_points_produce_zero_heatmap(self) -> None:
        heatmap = build_final_heatmap(np.empty((0, 2)), width=8, height=6, radius=2)

        self.assertEqual(heatmap.shape, (6, 8))
        self.assertFalse(heatmap.any())
        self.assertFalse(normalize_to_uint8(heatmap).any())

    def test_outside_point_does_not_modify_heatmap(self) -> None:
        heatmap = np.zeros((6, 8), dtype=np.float32)

        add_brush(heatmap, -100, -100, radius=2)

        self.assertFalse(heatmap.any())

    def test_edge_brush_is_clipped_without_wrapping(self) -> None:
        heatmap = np.zeros((6, 8), dtype=np.float32)

        add_brush(heatmap, 0, 0, radius=2)

        self.assertGreater(heatmap[0, 0], 0)
        self.assertEqual(float(heatmap[-1, -1]), 0.0)

    def test_normalization_maps_maximum_to_255(self) -> None:
        heatmap = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)

        normalized = normalize_to_uint8(heatmap)

        self.assertEqual(int(normalized[0, 0]), 0)
        self.assertEqual(int(normalized[1, 1]), 255)


if __name__ == "__main__":
    unittest.main()
