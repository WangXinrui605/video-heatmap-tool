from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from video_heatmap.tool_app.artifacts import new_run_record, write_json_atomic
from video_heatmap.tool_app.cancellation import CancellationResult
from video_heatmap.tool_app.config import Method, RunConfig
from video_heatmap.tool_app.process_runner import ProcessRunner


class ProcessRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def make_config(self, folder: Path) -> RunConfig:
        gaze = folder / "gaze.txt"
        video = folder / "scene.mp4"
        gaze.touch()
        video.touch()
        return RunConfig(Method.BASELINE, gaze, video, folder / "run", 0, 10, 40)

    def test_start_uses_frozen_tokenized_command(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            config = self.make_config(Path(name))
            runner = ProcessRunner()
            with patch.object(runner.process, "start") as start:
                tokens = runner.start(config, python_executable="python-test")
        start.assert_called_once_with(tokens[0], tokens[1:])
        self.assertEqual(tokens[0], "python-test")
        self.assertTrue(tokens[1].endswith("scripts\\run_heatmap_tool.py") or tokens[1].endswith("scripts/run_heatmap_tool.py"))

    def test_summary_schema_and_status_are_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            config = self.make_config(folder)
            config.output_dir.mkdir()
            runner = ProcessRunner()
            runner.config = config
            record = new_run_record(Method.BASELINE, {}, config.output_dir)
            write_json_atomic(config.output_dir / "tool_run_summary.json", record)
            self.assertEqual(runner._load_summary()["status"], "running")
            record["schema_version"] = "legacy"
            write_json_atomic(config.output_dir / "tool_run_summary.json", record)
            with self.assertRaisesRegex(RuntimeError, "v2"):
                runner._load_summary()

    def test_cancel_targets_retained_root_pid_and_marks_v2_record(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            config = self.make_config(folder)
            config.output_dir.mkdir()
            write_json_atomic(
                config.output_dir / "tool_run_summary.json",
                new_run_record(Method.BASELINE, {}, config.output_dir),
            )
            runner = ProcessRunner()
            runner.config = config
            runner.root_pid = 43210
            with patch.object(ProcessRunner, "running", new_callable=__import__("unittest.mock").mock.PropertyMock, return_value=True), patch.object(
                runner._cancellation_scope,
                "cancel",
                return_value=CancellationResult(43210, 0, "terminated", "", "windows-job-object"),
            ) as cancel:
                runner.cancel()
                runner._write_cancelled_record()
            cancel.assert_called_once_with()
            summary = json.loads((config.output_dir / "tool_run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "cancelled")
        self.assertFalse(summary["exportable"])

    def test_early_cancel_creates_atomic_v2_cancelled_record(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            config = self.make_config(folder)
            config.output_dir.mkdir()
            runner = ProcessRunner()
            runner.config = config
            runner.root_pid = 12345
            with patch.object(ProcessRunner, "running", new_callable=__import__("unittest.mock").mock.PropertyMock, return_value=True), patch.object(
                runner._cancellation_scope,
                "cancel",
                return_value=CancellationResult(12345, 0, "terminated", "", "windows-job-object"),
            ):
                runner.cancel()
                runner._write_cancelled_record()
            summary = json.loads((config.output_dir / "tool_run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], "video_heatmap_tool_run.v2")
        self.assertEqual(summary["status"], "cancelled")
        self.assertFalse(summary["exportable"])

    def test_cancel_failure_atomically_marks_failed_and_emits_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            config = self.make_config(folder)
            config.output_dir.mkdir()
            write_json_atomic(
                config.output_dir / "tool_run_summary.json",
                new_run_record(Method.BASELINE, {}, config.output_dir),
            )
            runner = ProcessRunner()
            runner.config = config
            runner.root_pid = 54321
            terminal: list[tuple[str, dict]] = []
            runner.terminal.connect(lambda status, summary: terminal.append((status, summary)))
            with patch.object(ProcessRunner, "running", new_callable=__import__("unittest.mock").mock.PropertyMock, return_value=True), patch.object(
                runner._cancellation_scope,
                "cancel",
                return_value=CancellationResult(
                    54321, 5, "", "Access is denied.", "windows-job-object",
                ),
            ):
                result = runner.cancel()
            summary = json.loads((config.output_dir / "tool_run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 5)
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["exportable"])
        self.assertIn("Access is denied", summary["error"])
        self.assertEqual(terminal[0][0], "failed")

    def test_natural_completion_wins_cancel_race(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            config = self.make_config(folder)
            config.output_dir.mkdir()
            record = new_run_record(Method.BASELINE, {}, config.output_dir)
            primary = config.output_dir / "dynamic" / "heatmap_development.mp4"
            primary.parent.mkdir(parents=True)
            primary.touch()
            from video_heatmap.tool_app.artifacts import RunStatus, transition_run
            write_json_atomic(
                config.output_dir / "tool_run_summary.json",
                transition_run(record, RunStatus.COMPLETED, config.output_dir),
            )
            runner = ProcessRunner()
            runner.config = config
            runner.root_pid = 65432
            with patch.object(ProcessRunner, "running", new_callable=__import__("unittest.mock").mock.PropertyMock, return_value=True), patch.object(
                runner._cancellation_scope, "cancel",
            ) as cancel:
                result = runner.cancel()
            summary = json.loads((config.output_dir / "tool_run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result.mechanism, "completion-race")
        cancel.assert_not_called()
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["exportable"])

    def run_fixture(self, status: str) -> tuple[list[tuple[str, str]], str, dict]:
        with tempfile.TemporaryDirectory() as name:
            config = self.make_config(Path(name))
            fixture = Path(__file__).parent / "fixtures" / "tool_runner_backend.py"
            runner = ProcessRunner()
            logs: list[tuple[str, str]] = []
            result: list[tuple[str, dict]] = []
            runner.log_received.connect(lambda channel, text: logs.append((channel, text)))
            loop = QEventLoop()
            runner.terminal.connect(lambda state, summary: (result.append((state, summary)), loop.quit()))
            with patch(
                "video_heatmap.tool_app.process_runner.command_tokens",
                return_value=[sys.executable, str(fixture), status, str(config.output_dir)],
            ):
                runner.start(config)
                QTimer.singleShot(10_000, loop.quit)
                loop.exec()
            self.assertTrue(result, "QProcess fixture did not reach a terminal signal")
            terminal_status, summary = result[0]
        return logs, terminal_status, summary

    def test_qprocess_captures_stdout_stderr_and_completed_summary(self) -> None:
        logs, status, summary = self.run_fixture("completed")
        combined = "".join(text for _, text in logs)
        self.assertIn("fixture stdout", combined)
        self.assertIn("fixture stderr", combined)
        self.assertEqual(status, "completed")
        self.assertTrue(summary["exportable"])

    def test_qprocess_uses_failed_and_cancelled_summary_status(self) -> None:
        for expected in ("failed", "cancelled"):
            with self.subTest(expected=expected):
                _, status, summary = self.run_fixture(expected)
                self.assertEqual(status, expected)
                self.assertFalse(summary["exportable"])


if __name__ == "__main__":
    unittest.main()
