from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, Signal

from .adapter import command_tokens
from .artifacts import RUN_SCHEMA_VERSION, RunStatus, new_run_record, transition_run, write_json_atomic
from .cancellation import CancellationResult, CancellationScope
from .config import RunConfig
from .progress import PROGRESS_PREFIX, ProgressValidator, parse_progress_line


class ProcessRunner(QObject):
    log_received = Signal(str, str)
    stage_changed = Signal(str)
    progress_received = Signal(object)
    started = Signal(int)
    terminal = Signal(str, object)
    settled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.started.connect(self._on_started)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)
        self.config: RunConfig | None = None
        self.root_pid = 0
        self.cancel_requested = False
        self._terminal_emitted = False
        self._cancellation_scope: CancellationScope | None = CancellationScope()
        self._stdout_buffer = ""
        self._progress_validator: ProgressValidator | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, config: RunConfig, *, python_executable: str | Path | None = None) -> list[str]:
        if self.running:
            raise RuntimeError("A heatmap run is already processing.")
        tokens = command_tokens(config, python_executable=python_executable)
        self.config = config
        self.root_pid = 0
        self.cancel_requested = False
        self._terminal_emitted = False
        self._stdout_buffer = ""
        self._progress_validator = ProgressValidator(method=config.method.value)
        if self._cancellation_scope is not None:
            self._cancellation_scope.close()
        self._cancellation_scope = CancellationScope()
        self.stage_changed.emit("Starting unified backend")
        self.log_received.emit("command", repr(tokens))
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        self.process.start(tokens[0], tokens[1:])
        return tokens

    def cancel(self) -> CancellationResult | None:
        if not self.running or self.root_pid <= 0:
            return None
        try:
            summary = self._load_summary()
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
            summary = None
        if summary is not None and summary.get("status") == RunStatus.COMPLETED.value:
            self.log_received.emit("cancel", "Run completed before cancellation could take effect.")
            return CancellationResult(
                self.root_pid, 0, "Natural completion won the cancellation race.", "", "completion-race",
            )
        self.cancel_requested = True
        self.stage_changed.emit("Cancelling current run process tree")
        if self._cancellation_scope is None:
            result = CancellationResult(
                self.root_pid, 6, "", "The current run has no cancellation scope.", "unavailable",
            )
        else:
            result = self._cancellation_scope.cancel()
        self.log_received.emit("cancel", (result.stdout + result.stderr).strip())
        if result.returncode != 0:
            error = (
                f"Targeted {result.mechanism} cancellation failed with code "
                f"{result.returncode}: {result.stderr or 'no OS error text'}"
            )
            self._write_failed_record(error)
            self._emit_failed_terminal(error)
        return result

    def _summary_path(self) -> Path | None:
        return self.config.output_dir / "tool_run_summary.json" if self.config else None

    def _load_summary(self) -> dict[str, Any]:
        path = self._summary_path()
        if path is None or not path.is_file():
            raise RuntimeError("The backend did not produce tool_run_summary.json for this run.")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("schema_version") != RUN_SCHEMA_VERSION:
            raise RuntimeError("The run summary is not video_heatmap_tool_run.v2.")
        artifacts = summary.get("artifacts", {})
        if artifacts.get("schema_version") != "video_heatmap_artifacts.v1":
            raise RuntimeError("The run summary does not contain a validated v1 artifact manifest.")
        if summary.get("status") not in {item.value for item in RunStatus}:
            raise RuntimeError("The run summary has an unsupported status.")
        return summary

    def _write_cancelled_record(self) -> None:
        self._write_terminal_record(RunStatus.CANCELLED)

    def _write_failed_record(self, error: str) -> None:
        self._write_terminal_record(RunStatus.FAILED, error=error)

    def _write_terminal_record(self, status: RunStatus, *, error: str | None = None) -> None:
        if self.config is None:
            return
        path = self._summary_path()
        if path is None:
            return
        try:
            if path.is_file():
                record = self._load_summary()
            else:
                marker = self.config.marker.marker if self.config.marker else "Brasilia"
                record = new_run_record(
                    self.config.method,
                    self.config.as_dict(),
                    self.config.output_dir,
                    marker=marker,
                )
            if record["status"] == RunStatus.RUNNING.value:
                marker = self.config.marker.marker if self.config.marker else "Brasilia"
                record = transition_run(
                    record, status, self.config.output_dir, marker=marker, error=error,
                )
                write_json_atomic(path, record)
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            self.log_received.emit("stderr", f"Could not finalize {status.value} run record: {exc}")

    def _emit_failed_terminal(self, error: str) -> None:
        if self._terminal_emitted:
            return
        try:
            summary = self._load_summary()
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
            summary = {
                "schema_version": RUN_SCHEMA_VERSION,
                "status": RunStatus.FAILED.value,
                "exportable": False,
                "error": error,
            }
        self._terminal_emitted = True
        self.stage_changed.emit("Cancellation failed")
        self.terminal.emit(RunStatus.FAILED.value, summary)

    def _read_stdout(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self._stdout_buffer += text
        lines = self._stdout_buffer.splitlines(keepends=True)
        self._stdout_buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._stdout_buffer = lines.pop()
        for line in lines:
            self._consume_stdout_line(line.rstrip("\r\n"))

    def _consume_stdout_line(self, line: str) -> None:
        if line.startswith(PROGRESS_PREFIX):
            try:
                payload = parse_progress_line(line)
                assert payload is not None and self._progress_validator is not None
                accepted = self._progress_validator.accept(payload)
            except (AssertionError, ValueError, json.JSONDecodeError) as exc:
                self.log_received.emit("progress-rejected", f"{exc}\n")
                return
            self.stage_changed.emit(str(accepted["stage"]).replace("_", " ").title())
            self.progress_received.emit(accepted)
            return
        if line:
            self.log_received.emit("stdout", line + "\n")

    def _flush_stdout_buffer(self) -> None:
        if self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer, ""
            self._consume_stdout_line(line)

    def _read_stderr(self) -> None:
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self.log_received.emit("stderr", text)

    def _on_started(self) -> None:
        self.root_pid = int(self.process.processId())
        try:
            if self._cancellation_scope is None:
                raise RuntimeError("Cancellation scope was not created before QProcess startup.")
            self._cancellation_scope.attach(self.root_pid)
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                summary = self._load_summary()
            except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
                summary = None
            if summary is not None and summary.get("status") in {
                RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value,
            }:
                self.log_received.emit(
                    "process",
                    "Backend reached an atomic terminal summary before its short-lived root "
                    "could be attached to the cancellation scope.",
                )
                return
            error = f"Could not isolate current run for safe cancellation: {exc}"
            self.log_received.emit("stderr", error)
            self.process.kill()
            self._write_failed_record(error)
            self._emit_failed_terminal(error)
            return
        self.stage_changed.emit("Backend processing")
        self.started.emit(self.root_pid)

    def _on_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._read_stdout()
        self._flush_stdout_buffer()
        self._read_stderr()
        if self._terminal_emitted:
            if self._cancellation_scope is not None:
                self._cancellation_scope.close()
            self.settled.emit()
            return
        try:
            if self.cancel_requested:
                self._write_cancelled_record()
            summary = self._load_summary()
            status = str(summary["status"])
            if self.cancel_requested and status == RunStatus.COMPLETED.value:
                # Natural completion won the race; the validated record is authoritative.
                self.log_received.emit("cancel", "Run completed before cancellation could take effect.")
            elif self.cancel_requested and status == RunStatus.RUNNING.value:
                self._write_cancelled_record()
                summary = self._load_summary()
                status = str(summary["status"])
            if status == RunStatus.RUNNING.value:
                raise RuntimeError(f"Backend exited with code {exit_code} but left a running summary.")
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            status = RunStatus.FAILED.value
            summary = {"schema_version": RUN_SCHEMA_VERSION, "status": status, "exportable": False, "error": str(exc)}
        self._terminal_emitted = True
        if self._cancellation_scope is not None:
            self._cancellation_scope.close()
        self.stage_changed.emit(status.title())
        self.terminal.emit(status, summary)
        self.settled.emit()

    def _on_error(self, error: QProcess.ProcessError) -> None:
        self.log_received.emit("stderr", f"QProcess error: {error.name}")
        if error == QProcess.ProcessError.FailedToStart and not self._terminal_emitted:
            self._terminal_emitted = True
            summary = {
                "schema_version": RUN_SCHEMA_VERSION,
                "status": RunStatus.FAILED.value,
                "exportable": False,
                "error": "The unified backend process could not be started.",
            }
            self.terminal.emit(RunStatus.FAILED.value, summary)
