from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.tool_app.metadata import read_video_metadata, validate_frame_range
from video_heatmap.video_io import require_cv2


class MetadataTests(unittest.TestCase):
    def test_reads_video_metadata_and_duration(self) -> None:
        cv2 = require_cv2()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "clip.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.release()
            metadata = read_video_metadata(path)
        self.assertEqual((metadata.width, metadata.height, metadata.frame_count), (32, 24, 5))
        self.assertAlmostEqual(metadata.fps, 10.0, places=2)
        self.assertAlmostEqual(metadata.duration_seconds, 0.5, places=2)

    def test_frame_range_is_strict_and_end_exclusive(self) -> None:
        validate_frame_range(0, 10, 10)
        for start, end in ((-1, 1), (0, 0), (4, 3), (0, 11)):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "exclusive"):
                validate_frame_range(start, end, 10)


if __name__ == "__main__":
    unittest.main()
