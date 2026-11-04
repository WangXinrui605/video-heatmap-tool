from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.tool_app.progress import (
    PROGRESS_PREFIX,
    ProgressEmitter,
    ProgressValidator,
    parse_progress_line,
)


class ProgressTests(unittest.TestCase):
    def event(self, processed: int, total: int, sequence: int, **updates):
        value = {
            "schema_version": "video_heatmap_progress.v1",
            "run_id": "run-1",
            "method": "face",
            "stage": "processing",
            "processed_input_frames": processed,
            "total_input_frames": total,
            "percentage": processed * 100.0 / total,
            "elapsed_seconds": float(sequence),
            "estimated_remaining_seconds": None,
            "timestamp": "2026-08-29T00:00:00+00:00",
            "sequence": sequence,
        }
        value.update(updates)
        return value

    def test_zero_one_and_many_frames(self) -> None:
        validator = ProgressValidator(run_id="run-1", method="face")
        for event in (self.event(0, 5, 0, stage="preparing"), self.event(1, 5, 1), self.event(5, 5, 2, stage="finalising")):
            self.assertIs(validator.accept(event), event)
        self.assertEqual(validator.last_percentage, 100.0)

    def test_malformed_and_out_of_order_are_rejected(self) -> None:
        validator = ProgressValidator()
        validator.accept(self.event(1, 5, 1))
        for bad in (
            self.event(2, 5, 1),
            self.event(0, 5, 2),
            self.event(2, 6, 2),
            self.event(2, 5, 2, percentage=99),
        ):
            with self.assertRaises(ValueError):
                validator.accept(bad)

    def test_eta_warmup_unknown_then_measured(self) -> None:
        times = iter((0.0, 0.0, 0.1, 1.0, 1.0))
        lines: list[str] = []
        emitter = ProgressEmitter("run-1", "flow", 10, clock=lambda: next(times), sink=lines.append)
        first = emitter.emit(0, "preparing")
        warm = emitter.emit(1)
        measured = emitter.emit(4)
        final = emitter.emit(10, "finalising")
        self.assertIsNone(first["estimated_remaining_seconds"])
        self.assertIsNone(warm["estimated_remaining_seconds"])
        self.assertAlmostEqual(measured["estimated_remaining_seconds"], 1.5)
        self.assertIsNone(final["estimated_remaining_seconds"])
        self.assertTrue(all(line.startswith(PROGRESS_PREFIX) for line in lines))

    def test_eta_is_unknown_at_zero_processed_rate(self) -> None:
        times = iter((0.0, 5.0))
        emitter = ProgressEmitter("run-1", "baseline", 10, clock=lambda: next(times), sink=lambda _line: None)
        event = emitter.emit(0, "processing")
        self.assertIsNone(event["estimated_remaining_seconds"])

    def test_parser_accepts_unknown_additive_fields(self) -> None:
        payload = self.event(1, 1, 0, future_field="allowed")
        parsed = parse_progress_line(PROGRESS_PREFIX + json.dumps(payload))
        self.assertEqual(parsed["future_field"], "allowed")
        self.assertIsNone(parse_progress_line("ordinary log"))

    def test_frame_skip_does_not_enter_progress_math(self) -> None:
        sequences = []
        for _frame_skip in (1, 2, 7):
            validator = ProgressValidator()
            for sequence, processed in enumerate(range(0, 11)):
                validator.accept(self.event(processed, 10, sequence))
            sequences.append((validator.last_processed, validator.total, validator.last_percentage))
        self.assertEqual(sequences, [(10, 10, 100.0)] * 3)


if __name__ == "__main__":
    unittest.main()
