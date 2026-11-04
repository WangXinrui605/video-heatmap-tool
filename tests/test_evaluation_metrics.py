import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_methods.py"
SPEC = importlib.util.spec_from_file_location("evaluate_methods", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def face_fixture():
    records = [
        {"source_row_id": 10, "sample_time_seconds": 0.1, "assigned_frame": 1,
         "classification": "accepted_inside_face", "consumption_count": 1,
         "screen_x": 5.0, "screen_y": 7.0, "draw_back_x": 5.0, "draw_back_y": 7.0},
        {"source_row_id": 11, "sample_time_seconds": 0.2, "assigned_frame": 2,
         "classification": "outside_face", "consumption_count": 1},
    ]
    summary = {
        "selected_range": {"start_frame": 0, "end_frame_exclusive": 3},
        "accumulation_sample_assignment": {
            "schema_version": "gaze_assignment.v1",
            "maximum_samples_assigned_to_one_frame": 1,
            "duplicate_consumption_check": {"passed": True, "duplicate_consumption_excess": 0},
            "audit_consistency_check": {
                "coverage_partition_passed": True,
                "static_dynamic_canonical_audit_source_set_passed": True,
            },
        },
        "counts": {
            "selected_valid_source_rows": 2, "selected_invalid_source_rows": 1,
            "assigned_unique_source_rows": 2, "accepted_inside_face_unique_samples": 1,
            "rejected_outside_face_unique_samples": 1, "no_detection_unique_samples": 0,
            "identity_uncertain_unique_samples": 0, "frames_with_detection": 3,
            "frames_without_detection": 0,
        },
        "coverage": {"face_coverage_ratio": 0.5, "detection_availability_ratio": 1.0},
        "parameters": {"history_mode": "cumulative"},
        "tracking_lifecycle": {"loss_episode_count": 0, "reacquisition_count": 0},
    }
    return summary, {"records": records}


class EvaluationMetricTests(unittest.TestCase):
    def test_round_trip_error_is_pixel_euclidean_distance(self):
        rows = MODULE.accepted_round_trip_rows([{
            "source_row_id": 1, "assigned_frame": 2, "classification": "accepted_inside_face",
            "screen_x": 1.0, "screen_y": 2.0, "draw_back_x": 4.0, "draw_back_y": 6.0,
        }])
        self.assertEqual(rows[0]["error_pixels"], 5.0)

    def test_corrected_unique_partition_builds_common_set(self):
        face, audit = face_fixture()
        baseline = {"selected_range": {"start_frame": 0, "end_frame_exclusive": 3},
                    "selected_valid_gaze_samples": 2, "runtime_seconds": 1.0}
        result, _, common, round_trip = MODULE.build_evaluation(baseline, face, audit)
        self.assertEqual(result["common_sample_set"]["unique_source_row_count"], 2)
        self.assertEqual({row["source_row_id"] for row in common}, {10, 11})
        self.assertEqual(len(round_trip), 1)

    def test_rejects_superseded_face_schema(self):
        face, audit = face_fixture()
        face["accumulation_sample_assignment"]["schema_version"] = "frame_assignment.legacy"
        with self.assertRaisesRegex(ValueError, "G1 corrected"):
            MODULE.validate_face(face, audit)

    def test_rejects_duplicate_consumption(self):
        face, audit = face_fixture()
        face["accumulation_sample_assignment"]["duplicate_consumption_check"] = {
            "passed": False, "duplicate_consumption_excess": 1}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.validate_face(face, audit)

    def test_rejects_mismatched_common_interval(self):
        face, audit = face_fixture()
        baseline = {"selected_range": {"start_frame": 1, "end_frame_exclusive": 3},
                    "selected_valid_gaze_samples": 2}
        with self.assertRaisesRegex(ValueError, "intervals differ"):
            MODULE.build_evaluation(baseline, face, audit)

    def test_flow_requires_v2_window_and_separate_terminations(self):
        flow = {"schema_version": "optical_flow_event_semantics.v2", "history_mode": "window",
                "termination_counts": {"window_expired": 1, "flow_or_bounds": 2,
                                       "flow_tracking_lost": 3, "confirmed_hard_cut": 4}}
        MODULE.validate_flow(flow)
        flow["termination_counts"].pop("window_expired")
        with self.assertRaisesRegex(ValueError, "missing or conflated"):
            MODULE.validate_flow(flow)

    def test_zero_accepted_rows_have_no_round_trip_statistic(self):
        face, audit = face_fixture()
        audit["records"][0]["classification"] = "outside_face"
        face["counts"]["accepted_inside_face_unique_samples"] = 0
        face["counts"]["rejected_outside_face_unique_samples"] = 2
        face["coverage"]["face_coverage_ratio"] = 0.0
        checked = MODULE.validate_face(face, audit)
        self.assertEqual(checked["round_trip_rows"], [])
        self.assertIsNone(checked["round_trip_mean_pixels"])


if __name__ == "__main__":
    unittest.main()
