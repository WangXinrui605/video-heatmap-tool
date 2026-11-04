from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.gaze_assignment import (
    SourceGazeSample,
    assign_source_rows_to_frames,
    audit_consistency_summary,
    duplicate_consumption_summary,
    first_frame_not_earlier_than,
)


def sample(row: int, time_seconds: float) -> SourceGazeSample:
    return SourceGazeSample(row, time_seconds, 100.0 + row, 200.0)


class GazeAssignmentTests(unittest.TestCase):
    def test_sparse_sample_near_two_frames_is_assigned_once(self) -> None:
        plan = assign_source_rows_to_frames(
            [sample(7, 0.049)], fps=30.0, start_frame=0, end_frame_exclusive=5
        )
        self.assertEqual(len(plan.assignments), 1)
        self.assertEqual(plan.assignments[0].frame_index, 2)
        self.assertEqual(plan.assigned_source_row_ids, (7,))

    def test_multiple_samples_on_one_frame_are_all_preserved(self) -> None:
        plan = assign_source_rows_to_frames(
            [sample(1, 0.034), sample(2, 0.049), sample(3, 0.060)],
            fps=30.0,
            start_frame=0,
            end_frame_exclusive=5,
        )
        self.assertEqual([item.sample.source_row_id for item in plan.by_frame[2]], [1, 2, 3])
        self.assertEqual(plan.maximum_samples_per_frame, 3)

    def test_equal_timestamps_do_not_merge_distinct_source_rows(self) -> None:
        plan = assign_source_rows_to_frames(
            [sample(10, 0.1), sample(11, 0.1)],
            fps=30.0,
            start_frame=0,
            end_frame_exclusive=10,
        )
        self.assertEqual(plan.assigned_source_row_ids, (10, 11))

    def test_range_boundaries_and_video_end(self) -> None:
        self.assertEqual(first_frame_not_earlier_than(0.0, 30.0), 0)
        self.assertEqual(first_frame_not_earlier_than(1.0 / 30.0, 30.0), 1)
        plan = assign_source_rows_to_frames(
            [sample(1, 0.0), sample(2, 2.0 / 30.0)],
            fps=30.0,
            start_frame=0,
            end_frame_exclusive=3,
        )
        self.assertEqual([item.frame_index for item in plan.assignments], [0, 2])
        with self.assertRaises(ValueError):
            assign_source_rows_to_frames(
                [sample(3, 3.0 / 30.0)],
                fps=30.0,
                start_frame=0,
                end_frame_exclusive=3,
            )

    def test_frame_skip_only_changes_output_frames(self) -> None:
        samples = [sample(1, 0.01), sample(2, 0.04), sample(3, 0.08)]
        full = assign_source_rows_to_frames(
            samples, fps=30.0, start_frame=0, end_frame_exclusive=6, frame_skip=1
        )
        skipped = assign_source_rows_to_frames(
            samples, fps=30.0, start_frame=0, end_frame_exclusive=6, frame_skip=3
        )
        self.assertEqual(full.assignments, skipped.assignments)
        self.assertNotEqual(full.output_frame_indices, skipped.output_frame_indices)

    def test_duplicate_source_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate source_row_id"):
            assign_source_rows_to_frames(
                [sample(2, 0.1), sample(2, 0.2)],
                fps=30.0,
                start_frame=0,
                end_frame_exclusive=20,
            )

    def test_duplicate_consumption_check(self) -> None:
        self.assertTrue(duplicate_consumption_summary({1: 1, 2: 1})["passed"])
        summary = duplicate_consumption_summary({1: 2, 2: 1})
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["duplicate_consumption_excess"], 1)

    def test_audit_partition_and_canonical_use_same_source_set(self) -> None:
        records = [
            {"source_row_id": 1, "classification": "accepted_inside_face", "consumption_count": 1},
            {"source_row_id": 2, "classification": "outside_face", "consumption_count": 1},
            {"source_row_id": 3, "classification": "no_detection", "consumption_count": 1},
            {"source_row_id": 4, "classification": "identity_uncertain", "consumption_count": 1},
            {"source_row_id": 5, "classification": "invalid_source_row", "consumption_count": 0},
        ]
        summary = audit_consistency_summary(records, canonical_accepted_count=1)
        self.assertTrue(summary["coverage_partition_passed"])
        self.assertTrue(summary["static_dynamic_canonical_audit_source_set_passed"])
        self.assertEqual(summary["assigned_audit_record_count"], 4)

    def test_audit_detects_duplicate_or_canonical_mismatch(self) -> None:
        records = [
            {"source_row_id": 1, "classification": "accepted_inside_face", "consumption_count": 2},
            {"source_row_id": 1, "classification": "outside_face", "consumption_count": 1},
        ]
        summary = audit_consistency_summary(records, canonical_accepted_count=2)
        self.assertFalse(summary["source_row_identity_unique"])
        self.assertFalse(summary["all_assigned_consumption_counts_equal_one"])
        self.assertFalse(summary["canonical_matches_accepted_audit"])
        self.assertFalse(summary["static_dynamic_canonical_audit_source_set_passed"])


if __name__ == "__main__":
    unittest.main()
