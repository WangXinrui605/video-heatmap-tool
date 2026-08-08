from __future__ import annotations

import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.time_range import resolve_selected_range
from video_heatmap.video_io import VideoInfo


class ResolveSelectedRangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video = VideoInfo(
            path=Path("video.mp4"),
            width=1920,
            height=1080,
            fps=25.0,
            frame_count=250,
        )

    def test_defaults_to_the_whole_video(self) -> None:
        selected = resolve_selected_range(self.video)

        self.assertEqual(selected.start_frame, 0)
        self.assertEqual(selected.end_frame_exclusive, 250)
        self.assertEqual(selected.frame_count, 250)

    def test_accepts_frame_range_with_exclusive_end(self) -> None:
        selected = resolve_selected_range(self.video, start_frame=10, end_frame=20)

        self.assertEqual(selected.start_frame, 10)
        self.assertEqual(selected.end_frame_exclusive, 20)

    def test_converts_inclusive_time_range_to_frames(self) -> None:
        selected = resolve_selected_range(self.video, start_time=1.0, end_time=2.0)

        self.assertEqual(selected.start_frame, 25)
        self.assertEqual(selected.end_frame_exclusive, 51)
        self.assertAlmostEqual(selected.end_time_seconds, 2.0)

    def test_rejects_mixed_start_arguments(self) -> None:
        with self.assertRaises(ValueError):
            resolve_selected_range(self.video, start_frame=1, start_time=1.0)

    def test_rejects_reversed_range(self) -> None:
        with self.assertRaises(ValueError):
            resolve_selected_range(self.video, start_frame=20, end_frame=10)

    def test_clamps_end_frame_to_video_length(self) -> None:
        selected = resolve_selected_range(self.video, start_frame=240, end_frame=999)

        self.assertEqual(selected.end_frame_exclusive, 250)
        self.assertEqual(selected.frame_count, 10)

    def test_fractional_start_time_uses_first_frame_at_or_after_time(self) -> None:
        selected = resolve_selected_range(self.video, start_time=1.001, end_time=1.05)

        self.assertEqual(selected.start_frame, 26)
        self.assertEqual(selected.end_frame_exclusive, 27)

    def test_rejects_negative_frame(self) -> None:
        with self.assertRaises(ValueError):
            resolve_selected_range(self.video, start_frame=-1)

    def test_serializes_consistent_exclusive_range(self) -> None:
        selected = resolve_selected_range(self.video, start_frame=10, end_frame=20)

        self.assertEqual(
            selected.as_dict(),
            {
                "start_frame": 10,
                "end_frame_exclusive": 20,
                "last_output_frame": 19,
                "start_time_seconds": 0.4,
                "end_time_seconds": 0.76,
            },
        )


if __name__ == "__main__":
    unittest.main()
