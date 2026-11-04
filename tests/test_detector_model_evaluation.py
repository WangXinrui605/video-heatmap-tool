from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_detector_models import (
    EvaluationError,
    aggregate_rows,
    load_matched_summaries,
    per_1000_frames,
    project_row,
    ratio,
)


def summary(participant: str = "P2", provider: str = "opencv-yunet") -> dict[str, object]:
    return {
        "participant": participant,
        "provider_id": provider,
        "provider_config": {"confidence_threshold": 0.8},
        "selected_range": {"start_frame": 0, "end_frame_exclusive": 4},
        "processed_frames": 4,
        "zero_face_frames": 1,
        "one_face_frames": 2,
        "multiple_face_frames": 1,
        "detection_available_frames": 3,
        "detection_count": 4,
        "confidence": {"minimum": 0.8, "mean": 0.85, "median": 0.85, "maximum": 0.9},
        "detector_runtime_seconds": 2.0,
        "wall_runtime_seconds": 4.0,
        "diagnostic_totals": {"invalid_rejected": 1, "clipped_boxes": 2, "min_size_or_outside_rejected": 3},
        "raw_records": "raw.jsonl",
    }


class DetectorModelEvaluationTests(unittest.TestCase):
    def test_ratio_and_per_1000_frame_formulas(self) -> None:
        self.assertEqual(ratio(3, 4), 0.75)
        self.assertEqual(per_1000_frames(2.0, 4), 500.0)

    def test_zero_denominators_are_rejected(self) -> None:
        with self.assertRaises(EvaluationError):
            ratio(0, 0)
        with self.assertRaises(EvaluationError):
            per_1000_frames(0.0, 0)

    def test_project_row_preserves_measured_runtime_and_partition(self) -> None:
        row = project_row(summary(), "fixture")
        self.assertEqual(row["status"], "measured_completed")
        self.assertEqual(row["selected_box_count"], 3)
        self.assertEqual(row["detector_seconds_per_1000_frames"], 500.0)
        self.assertEqual(row["wall_seconds_per_1000_frames"], 1000.0)

    def test_partition_mismatch_is_rejected(self) -> None:
        bad = summary()
        bad["zero_face_frames"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matched.json"
            path.write_text(json.dumps([bad]), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "partition"):
                load_matched_summaries(path, participants=("P2",), providers=("opencv-yunet",), expected_range=(0, 4))

    def test_missing_provider_or_range_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matched.json"
            path.write_text(json.dumps([summary()]), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "set mismatch"):
                load_matched_summaries(path, participants=("P2",), providers=("opencv-yunet", "opencv-dnn-ssd"), expected_range=(0, 4))
            with self.assertRaisesRegex(EvaluationError, "range mismatch"):
                load_matched_summaries(path, participants=("P2",), providers=("opencv-yunet",), expected_range=(1, 5))

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "missing evidence file"):
            load_matched_summaries(Path("does-not-exist.json"), participants=(), providers=(), expected_range=(0, 1))

    def test_projected_or_failed_status_cannot_enter_measured_aggregate(self) -> None:
        row = project_row(summary(), "fixture")
        row["status"] = "projected"
        with self.assertRaisesRegex(EvaluationError, "projected/failed/not-run"):
            aggregate_rows([row], "fixture")


if __name__ == "__main__":
    unittest.main()
