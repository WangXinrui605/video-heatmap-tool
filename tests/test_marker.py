from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.marker import distance_to_marker_square, discover_marker_columns


class DiscoverMarkerColumnsTests(unittest.TestCase):
    def test_accepts_standard_dlab_marker_gaze_columns(self) -> None:
        columns = [
            "Device_Gaze in Marker Coordinates_Brasilia_Pupil X",
            "Device_Gaze in Marker Coordinates_Brasilia_Pupil Y",
        ]

        markers = discover_marker_columns(columns)

        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].name, "Brasilia")
        self.assertTrue(markers[0].has_gaze_coordinates)

    def test_accepts_p2_columns_with_missing_marker_word(self) -> None:
        columns = [
            "Device_Gaze in  Coordinates_Brasilia_Pupil X",
            "Device_Gaze in Marker Coordinates_Brasilia_Pupil Y",
            "Device_Gaze in Marker Coordinates_Seoul_Pupil X",
            "Device_Gaze in  Coordinates_Seoul_Pupil Y",
        ]

        markers = {marker.name: marker for marker in discover_marker_columns(columns)}

        self.assertTrue(markers["Brasilia"].has_gaze_coordinates)
        self.assertTrue(markers["Seoul"].has_gaze_coordinates)


class DistanceToMarkerSquareTests(unittest.TestCase):
    def test_inside_and_boundary_points_have_zero_distance(self) -> None:
        points = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.25]])

        distances = distance_to_marker_square(points)

        np.testing.assert_allclose(distances, [0.0, 0.0, 0.0])

    def test_outside_distance_is_euclidean(self) -> None:
        points = np.array([[3.0, 0.5], [-1.0, 2.0]])

        distances = distance_to_marker_square(points)

        np.testing.assert_allclose(distances, [2.0, np.sqrt(2.0)])

    def test_rejects_values_without_xy_pairs(self) -> None:
        with self.assertRaises(ValueError):
            distance_to_marker_square(np.array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
