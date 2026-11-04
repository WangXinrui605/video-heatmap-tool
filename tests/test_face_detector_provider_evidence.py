from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_face_detector_providers import summarise_records


class ProviderEvidenceTests(unittest.TestCase):
    def test_summary_partitions_availability_and_aggregates_diagnostics(self) -> None:
        records = [
            {
                "detections": [],
                "runtime_seconds": 0.1,
                "diagnostics": {"clipped_boxes": 0, "invalid_rejected": 1},
            },
            {
                "detections": [{"confidence": 0.8}],
                "runtime_seconds": 0.2,
                "diagnostics": {"clipped_boxes": 1, "invalid_rejected": 0},
            },
            {
                "detections": [{"confidence": 0.7}, {"confidence": 0.9}],
                "runtime_seconds": 0.3,
                "diagnostics": {"clipped_boxes": 0, "invalid_rejected": 0},
            },
        ]
        summary = summarise_records(records)
        self.assertEqual(summary["processed_frames"], 3)
        self.assertEqual(summary["zero_face_frames"], 1)
        self.assertEqual(summary["one_face_frames"], 1)
        self.assertEqual(summary["multiple_face_frames"], 1)
        self.assertAlmostEqual(summary["detection_availability"], 2 / 3)
        self.assertEqual(summary["diagnostic_totals"]["clipped_boxes"], 1)
