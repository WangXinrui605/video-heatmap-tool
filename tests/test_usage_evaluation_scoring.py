"""Tests use synthetic unit-test fixtures, never participant responses."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_usage_evaluation.py"
SPEC = importlib.util.spec_from_file_location("score_usage_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def versioned(**values):
    return {
        "protocol_version": MODULE.PROTOCOL_VERSION,
        "build_scope": MODULE.BUILD_SCOPE,
        **values,
    }


class UsageEvaluationScoringTests(unittest.TestCase):
    def test_sus_standard_scoring(self):
        self.assertEqual(MODULE.score_sus([5, 1, 5, 1, 5, 1, 5, 1, 5, 1]), 100.0)
        self.assertEqual(MODULE.score_sus([1, 5, 1, 5, 1, 5, 1, 5, 1, 5]), 0.0)
        self.assertEqual(MODULE.score_sus([3] * 10), 50.0)

    def test_sus_rejects_missing_and_invalid_responses(self):
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            MODULE.score_sus([3] * 9)
        with self.assertRaisesRegex(ValueError, "1 to 5"):
            MODULE.score_sus([3] * 9 + [6])
        with self.assertRaisesRegex(ValueError, "integer"):
            MODULE.score_sus([3] * 9 + [2.5])

    def test_incomplete_sus_is_reported_missing_not_scored_as_zero(self):
        row = versioned(participant_id="participant-001", sus_responses=[3] * 9)
        result = MODULE.summarize_sus([row])
        self.assertEqual(result["individual"], [])
        self.assertEqual(result["missing"][0]["participant_id"], "participant-001")
        self.assertIsNone(result["small_sample_descriptive"]["median"])

    def test_task_summary_preserves_individual_time_and_descriptive_range(self):
        rows = [
            versioned(participant_id="participant-001", task_id="task-01", outcome="success", duration_seconds=10,
                      error_count=0, help_count=0, prompt_level=0, error_types="", error_severity=""),
            versioned(participant_id="participant-002", task_id="task-01", outcome="partial", duration_seconds=20,
                      error_count=1, help_count=1, prompt_level=1,
                      error_types="navigation;interpretation", error_severity="minor"),
        ]
        result = MODULE.summarize_tasks(rows)
        self.assertEqual(result["outcomes"], {"partial": 1, "success": 1})
        self.assertEqual(result["time_by_task"]["task-01"]["median"], 15.0)
        self.assertEqual(result["time_by_task"]["task-01"]["minimum"], 10.0)
        self.assertEqual(result["errors"]["by_type"]["navigation"], 1)

    def test_task_summary_rejects_missing_invalid_data(self):
        bad = versioned(participant_id="participant-001", task_id="task-01", outcome="complete",
                        duration_seconds=-1, error_count=0, help_count=0, prompt_level=0)
        with self.assertRaisesRegex(ValueError, "invalid task outcome"):
            MODULE.summarize_tasks([bad])

    def test_not_attempted_task_keeps_missing_time_missing(self):
        row = versioned(participant_id="participant-001", task_id="task-15", outcome="not_attempted",
                        duration_seconds="", error_count=0, help_count=0, prompt_level=0)
        result = MODULE.summarize_tasks([row])
        self.assertEqual(result["outcomes"], {"not_attempted": 1})
        self.assertEqual(result["individual_times"], [])

    def test_usefulness_is_itemwise_without_composite(self):
        rows = [
            versioned(participant_id="participant-001", item_id="usefulness-1", response=5),
            versioned(participant_id="participant-002", item_id="usefulness-1", response=3),
            versioned(participant_id="participant-001", item_id="usefulness-2", response=2),
        ]
        result = MODULE.summarize_usefulness(rows)
        self.assertEqual(result["items"]["usefulness-1"]["median"], 4.0)
        self.assertEqual(result["items"]["usefulness-1"]["distribution_1_to_5"]["5"], 1)
        self.assertIsNone(result["overall_score"])
        self.assertFalse(result["overall_score_validated"])

    def test_protocol_and_build_mismatch_are_rejected(self):
        row = versioned(participant_id="participant-001", sus_responses=[3] * 10)
        row["protocol_version"] = "usage-evaluation-v0.9"
        with self.assertRaisesRegex(ValueError, "protocol version mismatch"):
            MODULE.summarize_sus([row])
        row["protocol_version"] = MODULE.PROTOCOL_VERSION
        row["build_scope"] = "legacy-four-method-gui"
        with self.assertRaisesRegex(ValueError, "build scope mismatch"):
            MODULE.summarize_sus([row])

    def test_real_data_requires_ethics_gate(self):
        payload = {"data_class": "consented_anonymized_session_data", "participants": [],
                   "tasks": [], "usefulness": [], "ethics_consent_gate_confirmed": False}
        with self.assertRaisesRegex(ValueError, "ethics/consent"):
            MODULE.build_summary(payload)

    def test_real_participant_requires_record_level_consent(self):
        payload = {"data_class": "consented_anonymized_session_data", "participants": [
            versioned(participant_id="participant-001", sus_responses=[3] * 10, consent_confirmed=False)],
            "tasks": [], "usefulness": [], "ethics_consent_gate_confirmed": True}
        with self.assertRaisesRegex(ValueError, "consent_confirmed"):
            MODULE.build_summary(payload)

    def test_synthetic_fixture_is_explicit_and_supported(self):
        payload = {"data_class": "synthetic_unit_test_fixture", "participants": [],
                   "tasks": [], "usefulness": []}
        result = MODULE.build_summary(payload)
        self.assertEqual(result["data_class"], "synthetic_unit_test_fixture")


if __name__ == "__main__":
    unittest.main()
