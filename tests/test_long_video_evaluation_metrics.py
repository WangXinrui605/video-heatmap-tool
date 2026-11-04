import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_long_video_runs import (
    MetricsValidationError,
    parse_legacy_entry,
    validate_face_core,
    validate_flow_core,
    validate_l1_summary,
    validate_event_review_rows,
    validate_tool_manifest,
)


def face_core():
    return {
        "accumulation_sample_assignment": {
            "schema_version": "gaze_assignment.v1",
            "maximum_samples_assigned_to_one_frame": 2,
            "duplicate_consumption_check": {"passed": True, "duplicate_consumption_excess": 0},
            "audit_consistency_check": {
                "coverage_partition_passed": True,
                "static_dynamic_canonical_audit_source_set_passed": True,
            },
        },
        "parameters": {"history_mode": "cumulative"},
        "counts": {
            "processed_frames": 1800,
            "frames_with_detection": 1700,
            "frames_without_detection": 100,
            "multiple_face_frames": 4,
            "selected_valid_source_rows": 3000,
            "selected_invalid_source_rows": 20,
            "assigned_unique_source_rows": 3000,
            "accepted_inside_face_unique_samples": 1000,
            "rejected_outside_face_unique_samples": 1900,
            "no_detection_unique_samples": 90,
            "identity_uncertain_unique_samples": 10,
        },
        "coverage": {
            "face_coverage_ratio": 1 / 3,
            "valid_gaze_partition_check": True,
            "detection_availability_ratio": 1700 / 1800,
        },
        "tracking_lifecycle": {
            "loss_episode_count": 3,
            "freeze_visible_frames": 25,
            "hidden_retained_frames": 75,
            "reacquisition_count": 3,
        },
    }


def flow_core():
    return {
        "schema_version": "optical_flow_event_semantics.v2",
        "history_mode": "window",
        "processed_frames": 1800,
        "written_frames": 360,
        "selected_gaze_count": 3000,
        "added_gaze_count": 3000,
        "final_active_count": 100,
        "maximum_active_count": 150,
        "event_counts": {
            "visual_discontinuity_candidate": 1,
            "reviewed_continuous_non_cut": 1,
            "confirmed_hard_cut": 0,
            "uncertain_event": 0,
        },
        "events": [
            {
                "automatic_label": "visual_discontinuity_candidate",
                "reviewed_label": "continuous_non_cut",
                "action": "no_reset",
                "confirmed_hard_cut": False,
            }
        ],
        "termination_counts": {
            "window_expired": 2800,
            "flow_or_bounds": 100,
            "flow_tracking_lost": 0,
            "confirmed_hard_cut": 0,
        },
        "loss_episode_count": 0,
        "technical_segment_count": 1,
        "flow_interval_count": 1,
        "forward_backward_errors": {"count": 50, "median": 0.1},
        "parameters": {"window_seconds": 2.0},
    }


class LongVideoMetricsTests(unittest.TestCase):
    def test_face_partition_and_lifecycle_are_extracted(self):
        metrics = validate_face_core(face_core())
        self.assertEqual(metrics["assigned_unique_source_rows"], 3000)
        self.assertEqual(metrics["loss_episode_count"], 3)
        self.assertEqual(metrics["canonical_count"], 1000)

    def test_face_duplicate_consumption_is_rejected(self):
        core = face_core()
        core["accumulation_sample_assignment"]["duplicate_consumption_check"]["passed"] = False
        with self.assertRaises(MetricsValidationError):
            validate_face_core(core)

    def test_face_partition_mismatch_is_rejected(self):
        core = face_core()
        core["counts"]["identity_uncertain_unique_samples"] = 11
        with self.assertRaises(MetricsValidationError):
            validate_face_core(core)

    def test_flow_termination_and_review_are_extracted(self):
        metrics = validate_flow_core(flow_core())
        self.assertEqual(metrics["termination_counts"]["window_expired"], 2800)
        self.assertEqual(metrics["reviewed_event_count"], 1)
        self.assertEqual(metrics["runtime_action_counts"]["no_reset"], 1)

    def test_flow_legacy_boundary_is_rejected(self):
        core = flow_core()
        core["shot_boundary"] = 1
        with self.assertRaises(MetricsValidationError):
            validate_flow_core(core)

    def test_flow_candidate_count_must_match_records(self):
        core = flow_core()
        core["event_counts"]["visual_discontinuity_candidate"] = 2
        with self.assertRaises(MetricsValidationError):
            validate_flow_core(core)

    def test_l1_summary_requires_claim_boundary_and_runtime(self):
        summary = {
            "schema_version": "l1_long_video_run.v1",
            "evidence_class": "new_long_run",
            "status": "completed",
            "method": "flow",
            "runtime": {
                "wall_seconds": 10.0,
                "processing_fps": 180.0,
                "processing_time_to_video_duration_ratio": 1 / 6,
            },
            "output": {"primary_file_size_bytes": 10, "output_frames": 360},
            "metrics": {"termination_counts": flow_core()["termination_counts"]},
            "claim_boundary": {"absolute_gaze_accuracy_claimed": False},
        }
        self.assertEqual(validate_l1_summary(summary), [])
        summary["claim_boundary"]["absolute_gaze_accuracy_claimed"] = True
        self.assertTrue(validate_l1_summary(summary))

    def test_legacy_matrix_entry_is_explicitly_short(self):
        entry = parse_legacy_entry("P4,face,face_assignment_P4_0_300,outputs/example.json")
        self.assertEqual(entry["evidence_class"], "existing_short_smoke")
        self.assertEqual(entry["interval_tier"], "short_smoke")

    def test_tool_manifest_checks_declared_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("{}", encoding="utf-8")
            summary = {
                "schema_version": "video_heatmap_tool_run.v2",
                "status": "completed",
                "exportable": True,
                "artifacts": {
                    "schema_version": "video_heatmap_artifacts.v1",
                    "path_base": directory,
                    "artifacts": [
                        {"id": "summary", "path": "result.json", "exists": True, "exportable": True}
                    ],
                },
            }
            self.assertEqual(validate_tool_manifest(summary), [])
            summary["artifacts"]["artifacts"][0]["path"] = "missing.json"
            self.assertTrue(validate_tool_manifest(summary))

    def test_event_review_requires_consistent_non_cut_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "review.mp4"
            evidence.write_bytes(b"evidence")
            row = {
                "frame": 42,
                "time_seconds": 1.4,
                "automatic_label": "visual_discontinuity_candidate",
                "reviewed_label": "continuous_non_cut",
                "review_confidence": "high",
                "reviewer": "tester",
                "review_window": "[27,57]",
                "action": "no_reset",
                "confirmed_hard_cut": False,
                "evidence_path": str(evidence),
                "notes": "Continuous transition visible.",
            }
            self.assertEqual(validate_event_review_rows([row]), [])
            row["action"] = "confirmed_hard_cut_reset"
            self.assertTrue(validate_event_review_rows([row]))


if __name__ == "__main__":
    unittest.main()
