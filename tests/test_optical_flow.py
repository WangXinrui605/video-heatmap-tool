from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.optical_flow import (
    FarnebackConfig,
    FlowPair,
    bilinear_sample_flow,
    calculate_flow_pair,
    gaze_frame_indices,
    is_shot_boundary,
    propagate_points,
    resolve_frame_range,
)
from video_heatmap.video_io import VideoInfo


class ResolveFrameRangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video = VideoInfo(
            path=Path("video.mp4"),
            width=1920,
            height=1080,
            fps=25.0,
            frame_count=250,
        )

    def test_defaults_to_full_video(self) -> None:
        frame_range = resolve_frame_range(self.video)

        self.assertEqual(frame_range.start_frame, 0)
        self.assertEqual(frame_range.end_frame, 250)
        self.assertEqual(frame_range.end_time, 249 / 25)

    def test_time_range_is_inclusive_at_end(self) -> None:
        frame_range = resolve_frame_range(
            self.video, start_time=1.0, end_time=2.0
        )

        self.assertEqual(frame_range.start_frame, 25)
        self.assertEqual(frame_range.end_frame, 51)

    def test_rejects_mixed_start_units(self) -> None:
        with self.assertRaises(ValueError):
            resolve_frame_range(self.video, start_frame=0, start_time=0.0)


class GazeFrameIndexTests(unittest.TestCase):
    def test_assigns_sample_to_first_frame_at_or_after_timestamp(self) -> None:
        frames = gaze_frame_indices(np.array([0.0, 0.001, 0.04]), fps=25.0)

        np.testing.assert_array_equal(frames, [0, 1, 1])


class FlowSamplingTests(unittest.TestCase):
    def test_bilinear_sampling_converts_scaled_flow_to_full_resolution(self) -> None:
        flow = np.zeros((4, 4, 2), dtype=np.float32)
        flow[..., 0] = 1.0
        flow[..., 1] = 0.5

        sampled, valid = bilinear_sample_flow(
            flow,
            np.array([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32),
            scale_x=0.5,
            scale_y=0.5,
        )

        np.testing.assert_allclose(sampled, [[2.0, 1.0], [2.0, 1.0]])
        np.testing.assert_array_equal(valid, [True, True])

    def test_forward_backward_propagation_moves_points_and_rejects_bad_flow(self) -> None:
        forward = np.zeros((20, 20, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = 3.0
        backward[..., 0] = -3.0
        pair = FlowPair(
            forward=forward,
            backward=backward,
            scale_x=1.0,
            scale_y=1.0,
            disturbance_score=0.0,
        )

        result = propagate_points(
            np.array([[5.0, 5.0]], dtype=np.float32),
            pair,
            frame_width=20,
            frame_height=20,
            fb_threshold=0.1,
        )

        np.testing.assert_allclose(result.points, [[8.0, 5.0]])
        np.testing.assert_array_equal(result.valid, [True])
        np.testing.assert_allclose(result.forward_backward_error, [0.0])

        bad_pair = FlowPair(
            forward=forward,
            backward=np.zeros_like(backward),
            scale_x=1.0,
            scale_y=1.0,
            disturbance_score=0.0,
        )
        bad_result = propagate_points(
            np.array([[5.0, 5.0]], dtype=np.float32),
            bad_pair,
            frame_width=20,
            frame_height=20,
            fb_threshold=0.1,
        )
        np.testing.assert_array_equal(bad_result.valid, [False])


class FlowDisturbanceTests(unittest.TestCase):
    def test_unrelated_frames_have_higher_disturbance_than_identical_frames(self) -> None:
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
        unrelated = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
        config = FarnebackConfig(scale=1.0)

        identical_score = calculate_flow_pair(
            frame, frame.copy(), config
        ).disturbance_score
        unrelated_score = calculate_flow_pair(
            frame, unrelated, config
        )

        self.assertLess(identical_score, 1e-3)
        self.assertGreater(unrelated_score.disturbance_score, 0.05)
        self.assertTrue(is_shot_boundary(unrelated_score, 0.03, 0.0))


if __name__ == "__main__":
    unittest.main()
