from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.rendering import HeatmapGradient, HeatScale, HeatScaleMode, colorize, scale_heatmap


class RenderingContractTests(unittest.TestCase):
    def test_gradient_changes_colour_not_raw_heat_or_scale(self) -> None:
        raw = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
        original = raw.copy()
        scaled, resolved = scale_heatmap(raw, HeatScale())
        jet = colorize(scaled, HeatmapGradient.JET)
        viridis = colorize(scaled, HeatmapGradient.VIRIDIS)
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(resolved, 4.0)
        self.assertFalse(np.array_equal(jet, viridis))

    def test_fixed_maximum_clips_without_changing_raw_heat(self) -> None:
        raw = np.array([[0.0, 2.0, 4.0]], dtype=np.float32)
        original = raw.copy()
        scaled, resolved = scale_heatmap(raw, HeatScale(HeatScaleMode.FIXED, 2.0))
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(resolved, 2.0)
        self.assertEqual(scaled.tolist(), [[0, 255, 255]])

    def test_fixed_requires_explicit_positive_maximum(self) -> None:
        with self.assertRaises(ValueError):
            HeatScale(HeatScaleMode.FIXED, None)


if __name__ == "__main__":
    unittest.main()
