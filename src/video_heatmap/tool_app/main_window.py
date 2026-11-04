from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
import re
from time import monotonic

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from video_heatmap.rendering import HeatmapGradient, HeatScale, HeatScaleMode

from .artifacts import RUN_SCHEMA_VERSION
from .config import Method, RunConfig
from .export import build_export_plan, execute_export, export_folder_name
from .metadata import VideoMetadata, read_gaze_summary, read_video_metadata, validate_frame_range
from .method_panels import (
    FIRST_RELEASE_GUI_METHODS,
    METHOD_DETAILS,
    MethodPanels,
    ParameterComboBox,
    ParameterDoubleSpinBox,
    ParameterSpinBox,
    PathPicker,
)
from .preview import PreviewWidget, save_screenshot
from .process_runner import ProcessRunner


class AppState(str, Enum):
    EMPTY = "EMPTY"
    INPUT_READY = "INPUT_READY"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Heatmap Tool")
        self.resize(1180, 820)
        self.state = AppState.EMPTY
        self.video_metadata: VideoMetadata | None = None
        self.gaze_summary: dict | None = None
        self.last_validation_error = "Select the required inputs."
        self.current_config: RunConfig | None = None
        self.last_summary: dict | None = None
        self.last_export_dir: Path | None = None
        self.started_at = 0.0
        self.last_progress_event: dict | None = None
        self._build_ui()
        self.runner = ProcessRunner(self)
        self.runner.log_received.connect(self._append_log)
        self.runner.stage_changed.connect(self._set_stage)
        self.runner.progress_received.connect(self._progress_received)
        self.runner.started.connect(lambda pid: self._append_log("process", f"Current run root PID: {pid}"))
        self.runner.terminal.connect(self._run_terminal)
        self.runner.settled.connect(lambda: self._set_state(self.state))
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(250)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self._connect_validation()
        self._method_changed()
        self._refresh_validation()

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        self.setCentralWidget(root)

        inputs = QGroupBox("Inputs")
        form = QFormLayout(inputs)
        self.gaze_picker = PathPicker("Select D-Lab gaze TXT", "Text files (*.txt)")
        self.video_picker = PathPicker("Select Scene Cam video", "Video files (*.mp4 *.avi *.mov)")
        self.output_edit = QLineEdit()
        self.output_button = QPushButton("Browse…")
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(self.output_button)
        self.metadata_label = QLabel("Width: —  Height: —  FPS: —  Frames: —  Duration: —")
        self.gaze_label = QLabel("Gaze rows: —")
        form.addRow("Gaze TXT", self.gaze_picker)
        form.addRow("Scene Cam video", self.video_picker)
        form.addRow("Output/run destination", output_row)
        form.addRow("Video metadata", self.metadata_label)
        form.addRow("Gaze summary", self.gaze_label)
        outer.addWidget(inputs)

        middle = QHBoxLayout()
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        self.controls_scroll.setWidget(controls)
        middle.addWidget(self.controls_scroll, 3)

        method_box = QGroupBox("Method")
        method_form = QFormLayout(method_box)
        self.method_combo = QComboBox()
        for method in FIRST_RELEASE_GUI_METHODS:
            self.method_combo.addItem(METHOD_DETAILS[method]["label"], method)
        self.method_status = QLabel()
        self.method_history = QLabel()
        self.method_history.setWordWrap(True)
        self.method_limitations = QLabel()
        self.method_limitations.setWordWrap(True)
        method_form.addRow("Method", self.method_combo)
        method_form.addRow("Status", self.method_status)
        method_form.addRow("History semantics", self.method_history)
        method_form.addRow("Method limitations", self.method_limitations)
        controls_layout.addWidget(method_box)

        common = QGroupBox("Common parameters")
        common_form = QFormLayout(common)
        self.start_frame = ParameterSpinBox()
        self.end_frame = ParameterSpinBox()
        for field in (self.start_frame, self.end_frame):
            field.setRange(0, 2_147_483_647)
        self.end_frame.setValue(300)
        self.end_exclusive = QLabel("Exclusive: frame end is not processed")
        self.brush = ParameterDoubleSpinBox()
        self.brush.setRange(0.01, 10000)
        self.brush.setValue(40)
        self.brush.setDecimals(2)
        self.opacity = ParameterDoubleSpinBox()
        self.opacity.setRange(0, 1)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(0.55)
        self.update_interval = ParameterSpinBox()
        self.update_interval.setRange(1, 10000)
        self.update_interval.setValue(1)
        self.update_interval.setToolTip(
            "All frames still drive gaze assignment, detection, lifecycle and optical flow.\n"
            "This setting only reduces written output frames."
        )
        self.gradient = ParameterComboBox()
        for gradient in HeatmapGradient:
            self.gradient.addItem(gradient.value.upper(), gradient)
        self.heat_auto = QRadioButton("Auto")
        self.heat_fixed = QRadioButton("Fixed maximum")
        self.heat_auto.setChecked(True)
        self.heat_group = QButtonGroup(self)
        self.heat_group.addButton(self.heat_auto)
        self.heat_group.addButton(self.heat_fixed)
        self.heat_maximum = ParameterDoubleSpinBox()
        self.heat_maximum.setRange(0.0001, 1e9)
        self.heat_maximum.setValue(20)
        self.heat_maximum.setEnabled(False)
        heat_row = QWidget()
        heat_layout = QHBoxLayout(heat_row)
        heat_layout.setContentsMargins(0, 0, 0, 0)
        heat_layout.addWidget(self.heat_auto)
        heat_layout.addWidget(self.heat_fixed)
        heat_layout.addWidget(self.heat_maximum)
        self.heat_warning = QLabel(
            "Auto scaling may make a small number of samples look visually strong. "
            "Interpret colour together with exact sample counts."
        )
        self.heat_warning.setWordWrap(True)
        common_form.addRow("Start Frame (inclusive)", self.start_frame)
        common_form.addRow("End Frame", self.end_frame)
        common_form.addRow(self.end_exclusive)
        self.brush_label = QLabel("Brush Size")
        common_form.addRow(self.brush_label, self.brush)
        common_form.addRow("Opacity", self.opacity)
        common_form.addRow("Output Update Interval", self.update_interval)
        common_form.addRow("Gradient", self.gradient)
        common_form.addRow("Heat Scale", heat_row)
        common_form.addRow(self.heat_warning)
        controls_layout.addWidget(common)

        method_specific = QGroupBox("Method-specific parameters")
        method_layout = QVBoxLayout(method_specific)
        self.method_panels = MethodPanels()
        method_layout.addWidget(self.method_panels)
        controls_layout.addWidget(method_specific)
        controls_layout.addStretch(1)

        preview = QGroupBox("Manifest-driven Preview")
        preview_layout = QVBoxLayout(preview)
        self.preview_widget = PreviewWidget()
        preview_layout.addWidget(self.preview_widget)
        preview_actions = QHBoxLayout()
        self.preview_button = QPushButton("Preview Selected")
        self.screenshot_button = QPushButton("Save Screenshot")
        self.clear_button = QPushButton("Clear Preview")
        self.export_button = QPushButton("Export Results")
        self.open_output_button = QPushButton("Open Output Folder")
        for button in (
            self.preview_button,
            self.screenshot_button,
            self.clear_button,
            self.export_button,
            self.open_output_button,
        ):
            preview_actions.addWidget(button)
        preview_layout.addLayout(preview_actions)
        middle.addWidget(preview, 2)
        outer.addLayout(middle, 1)

        actions = QHBoxLayout()
        self.validate_button = QPushButton("Validate")
        self.generate_button = QPushButton("Generate")
        self.cancel_button = QPushButton("Cancel")
        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        actions.addWidget(self.validate_button)
        actions.addWidget(self.generate_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.validation_label, 1)
        outer.addLayout(actions)

        status_row = QHBoxLayout()
        self.state_label = QLabel("Status: EMPTY")
        self.stage_label = QLabel("Current stage: Waiting for inputs")
        self.elapsed_label = QLabel("Elapsed: 00:00")
        self.progress_label = QLabel("Progress: waiting")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        status_row.addWidget(self.state_label)
        status_row.addWidget(self.stage_label, 1)
        status_row.addWidget(self.elapsed_label)
        status_row.addWidget(self.progress_label)
        status_row.addWidget(self.progress)
        outer.addLayout(status_row)

        self.tabs = QTabWidget()
        self.summary_view = QTextEdit()
        self.limitations_view = QTextEdit()
        self.log_view = QTextEdit()
        for view in (self.summary_view, self.limitations_view, self.log_view):
            view.setReadOnly(True)
        self.tabs.addTab(self.summary_view, "Summary")
        self.tabs.addTab(self.limitations_view, "Limitations")
        self.tabs.addTab(self.log_view, "Log")
        outer.addWidget(self.tabs, 1)

    def _connect_validation(self) -> None:
        self.gaze_picker.changed.connect(self._input_paths_changed)
        self.video_picker.changed.connect(self._input_paths_changed)
        self.output_edit.textChanged.connect(self._refresh_validation)
        self.output_button.clicked.connect(self._browse_output)
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        self.method_panels.changed.connect(self._refresh_validation)
        for signal in (
            self.start_frame.valueChanged,
            self.end_frame.valueChanged,
            self.brush.valueChanged,
            self.opacity.valueChanged,
            self.update_interval.valueChanged,
            self.gradient.currentIndexChanged,
            self.heat_auto.toggled,
            self.heat_maximum.valueChanged,
        ):
            signal.connect(self._refresh_validation)
        self.heat_auto.toggled.connect(lambda checked: self.heat_maximum.setEnabled(not checked))
        self.generate_button.clicked.connect(self.generate)
        self.cancel_button.clicked.connect(self.cancel)
        self.validate_button.clicked.connect(self._validate_clicked)
        self.preview_button.clicked.connect(self.preview_widget.load_current)
        self.screenshot_button.clicked.connect(self._save_screenshot)
        self.clear_button.clicked.connect(self.clear_result)
        self.export_button.clicked.connect(self._export_results)
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.preview_widget.frame_availability_changed.connect(lambda _available: self._refresh_result_actions())
        self.preview_widget.artifact_availability_changed.connect(lambda _available: self._refresh_result_actions())
        self.preview_widget.error_occurred.connect(lambda message: self._append_log("preview", message))

    def current_method(self) -> Method:
        return Method(self.method_combo.currentData())

    def _browse_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select output/run destination", self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)

    def _input_paths_changed(self) -> None:
        video_path = self.video_picker.path()
        self.video_metadata = None
        if video_path and video_path.is_file():
            try:
                self.video_metadata = read_video_metadata(video_path)
                metadata = self.video_metadata
                self.metadata_label.setText(
                    f"Width: {metadata.width}  Height: {metadata.height}  FPS: {metadata.fps:.3f}  "
                    f"Frames: {metadata.frame_count}  Duration: {metadata.duration_seconds:.3f} s"
                )
                self.start_frame.setMaximum(metadata.frame_count)
                self.end_frame.setMaximum(metadata.frame_count)
                self.start_frame.setValue(0)
                self.end_frame.setValue(min(300, metadata.frame_count))
            except ValueError as exc:
                self.metadata_label.setText(str(exc))
        else:
            self.metadata_label.setText("Width: —  Height: —  FPS: —  Frames: —  Duration: —")

        gaze_path = self.gaze_picker.path()
        self.gaze_summary = None
        if gaze_path and gaze_path.is_file():
            try:
                self.gaze_summary = read_gaze_summary(gaze_path, self.video_metadata)
                summary = self.gaze_summary
                self.gaze_label.setText(
                    f"Total: {summary['total_rows']}  Valid: {summary['valid_rows']}  "
                    f"Invalid: {summary['invalid_rows']}  Timed valid: {summary['timed_valid_rows']}"
                )
            except (OSError, ValueError) as exc:
                self.gaze_label.setText(f"Gaze TXT could not be read: {exc}")
        else:
            self.gaze_label.setText("Gaze rows: —")
        self._refresh_validation()

    def _method_changed(self) -> None:
        method = self.current_method()
        details = METHOD_DETAILS[method]
        self.method_status.setText(details["status"])
        self.method_history.setText(details["history"])
        self.method_limitations.setText("\n".join(f"• {item}" for item in details["limitations"]))
        self.brush_label.setText(details["brush"])
        self.method_panels.set_method(method)
        if method is Method.FACE:
            self.brush.setDecimals(0)
            self.brush.setValue(12)
        else:
            self.brush.setDecimals(0)
            self.brush.setValue(40)
        if method is Method.FLOW:
            self.heat_fixed.setChecked(True)
            self.heat_maximum.setValue(20)
        else:
            self.heat_auto.setChecked(True)
        self._refresh_validation()

    def _base_output(self) -> Path:
        text = self.output_edit.text().strip()
        return Path(text) if text else Path("")

    @staticmethod
    def _safe_stem(path: Path) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")
        return safe[:48] or "run"

    def _unique_run_dir(self) -> Path:
        base = self._base_output()
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return base / f"{self._safe_stem(self.gaze_picker.path() or Path('run'))}_{self.current_method().value}_{timestamp}"

    def build_config(self, *, output_dir: Path | None = None) -> RunConfig:
        if self.video_metadata is None:
            raise ValueError("Select a Scene Cam video that can be opened.")
        gaze = self.gaze_picker.path()
        video = self.video_picker.path()
        base_output = self._base_output()
        if gaze is None or not gaze.is_file():
            raise ValueError("Select an existing gaze TXT file.")
        if video is None or not video.is_file():
            raise ValueError("Select an existing Scene Cam video.")
        if not self.output_edit.text().strip():
            raise ValueError("Select an output/run destination.")
        validate_frame_range(self.start_frame.value(), self.end_frame.value(), self.video_metadata.frame_count)
        try:
            base_output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Output destination cannot be created: {base_output}: {exc}") from exc
        if not base_output.is_dir():
            raise ValueError(f"Output destination is not a directory: {base_output}")
        scale = (
            HeatScale(HeatScaleMode.AUTO)
            if self.heat_auto.isChecked()
            else HeatScale(HeatScaleMode.FIXED, self.heat_maximum.value())
        )
        method = self.current_method()
        config = RunConfig(
            method=method,
            gaze_path=gaze,
            video_path=video,
            output_dir=output_dir or base_output / ".validation",
            start_frame=self.start_frame.value(),
            end_frame=self.end_frame.value(),
            brush_radius=self.brush.value(),
            opacity=self.opacity.value(),
            output_update_interval=self.update_interval.value(),
            gradient=HeatmapGradient(self.gradient.currentData()),
            heat_scale=scale,
            face=self.method_panels.face.config() if method is Method.FACE else None,
            flow=self.method_panels.flow.config() if method is Method.FLOW else None,
            marker=None,
        )
        config.validate()
        return config

    def validate_inputs(self) -> tuple[bool, str]:
        try:
            self.build_config()
        except (OSError, ValueError) as exc:
            return False, str(exc)
        return True, "Inputs and parameters are valid. Ready to generate a unique run directory."

    def _refresh_validation(self) -> None:
        if self.state is AppState.PROCESSING:
            return
        valid, message = self.validate_inputs()
        self.last_validation_error = "" if valid else message
        self.validation_label.setText(message)
        self._set_state(AppState.INPUT_READY if valid else AppState.EMPTY)

    def _validate_clicked(self) -> None:
        valid, message = self.validate_inputs()
        self.validation_label.setText(message)
        self._set_state(AppState.INPUT_READY if valid else AppState.EMPTY)
        if not valid:
            QMessageBox.warning(self, "Validation failed", message)

    def _set_state(self, state: AppState) -> None:
        self.state = state
        self.state_label.setText("Status: Processing…" if state is AppState.PROCESSING else f"Status: {state.value}")
        runner_active = bool(getattr(self, "runner", None) and self.runner.running)
        self.generate_button.setEnabled(
            state is not AppState.PROCESSING and state is not AppState.EMPTY and not runner_active
        )
        self.cancel_button.setEnabled(state is AppState.PROCESSING)
        self.validate_button.setEnabled(state is not AppState.PROCESSING and not runner_active)
        processing = state is AppState.PROCESSING
        if processing and self.last_progress_event is None:
            self.progress.setRange(0, 0)
            self.progress.setTextVisible(False)
            self.progress_label.setText("Progress: waiting for structured events")
        elif not processing and state in {AppState.EMPTY, AppState.INPUT_READY}:
            self.progress.setRange(0, 1000)
            self.progress.setValue(0)
            self.progress.setTextVisible(False)
            self.progress_label.setText("Progress: waiting")
        elif not processing and self.last_progress_event is None:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setTextVisible(False)
            self.progress_label.setText("Progress: unavailable (legacy/no events)")
        self._refresh_result_actions()

    def _completed_exportable_result(self) -> bool:
        return bool(
            self.state is AppState.COMPLETED
            and isinstance(self.last_summary, dict)
            and self.last_summary.get("schema_version") == RUN_SCHEMA_VERSION
            and self.last_summary.get("status") == "completed"
            and self.last_summary.get("exportable") is True
        )

    def _refresh_result_actions(self) -> None:
        processing = self.state is AppState.PROCESSING or bool(
            getattr(self, "runner", None) and self.runner.running
        )
        completed = self._completed_exportable_result()
        has_artifacts = bool(getattr(self, "preview_widget", None) and self.preview_widget.artifacts)
        artifact = self.preview_widget.current_artifact if getattr(self, "preview_widget", None) else None
        has_frame = bool(
            artifact is not None and artifact.exportable and not self.preview_widget.current_frame.isNull()
        )
        self.preview_button.setEnabled(completed and has_artifacts)
        self.screenshot_button.setEnabled(completed and has_frame)
        self.clear_button.setEnabled(not processing)
        self.export_button.setEnabled(completed)
        self.open_output_button.setEnabled(
            not processing and (self.last_export_dir is not None or (completed and self.current_config is not None))
        )

    def generate(self) -> None:
        valid, message = self.validate_inputs()
        if not valid:
            self._set_state(AppState.EMPTY)
            QMessageBox.warning(self, "Cannot generate", message)
            return
        run_dir = self._unique_run_dir()
        if run_dir.exists():
            QMessageBox.warning(self, "Output conflict", f"Run directory already exists: {run_dir}")
            return
        try:
            self.current_config = self.build_config(output_dir=run_dir)
            self.current_config.validate()
            run_dir.mkdir(parents=True, exist_ok=False)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot generate", str(exc))
            return
        self.last_summary = None
        self.last_progress_event = None
        self.last_export_dir = None
        self.preview_widget.clear()
        self.summary_view.clear()
        self.limitations_view.setPlainText("\n".join(METHOD_DETAILS[self.current_method()]["limitations"]))
        self.log_view.clear()
        self.tabs.setCurrentWidget(self.log_view)
        self.started_at = monotonic()
        self.elapsed_timer.start()
        self._set_state(AppState.PROCESSING)
        self._set_stage("Starting unified backend")
        try:
            self.runner.start(self.current_config)
        except (OSError, RuntimeError, ValueError) as exc:
            self.elapsed_timer.stop()
            self._set_state(AppState.FAILED)
            QMessageBox.critical(self, "Backend start failed", str(exc))

    def cancel(self) -> None:
        try:
            result = self.runner.cancel()
        except (OSError, ValueError) as exc:
            self._append_log("stderr", f"Cancellation failed: {exc}")
            return
        if result is None:
            self._append_log("cancel", "The current run is no longer active.")
        elif result.returncode != 0:
            self._append_log("stderr", f"Targeted tree cancellation returned {result.returncode}: {result.stderr}")

    def _run_terminal(self, status: str, summary: object) -> None:
        self.elapsed_timer.stop()
        data = summary if isinstance(summary, dict) else {}
        self.last_summary = data
        state_map = {
            "completed": AppState.COMPLETED,
            "failed": AppState.FAILED,
            "cancelled": AppState.CANCELLED,
        }
        self._set_state(state_map.get(status, AppState.FAILED))
        if self.last_progress_event is not None:
            processed = self.last_progress_event["processed_input_frames"]
            total = self.last_progress_event["total_input_frames"]
            terminal_text = "Complete" if status == "completed" else status.title()
            self.progress_label.setText(f"Frames: {processed}/{total}  {terminal_text}")
        self.summary_view.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
        method_record = data.get("config", {}).get("method", {})
        limitations = method_record.get("limitations") if isinstance(method_record, dict) else None
        if not isinstance(limitations, list):
            limitations = METHOD_DETAILS[self.current_method()]["limitations"]
        self.limitations_view.setPlainText("\n\n".join(str(item) for item in limitations))
        self.tabs.setCurrentWidget(self.summary_view if status == "completed" else self.log_view)
        if data.get("schema_version") != RUN_SCHEMA_VERSION:
            self._append_log("stderr", "Terminal result did not validate as video_heatmap_tool_run.v2.")
        if status == "completed":
            try:
                self.preview_widget.set_summary(data)
            except (OSError, ValueError) as exc:
                self.preview_widget.clear()
                self._append_log("stderr", f"Completed result cannot be previewed: {exc}")
        else:
            self.preview_widget.clear()
        self._refresh_result_actions()

    def _progress_received(self, event: object) -> None:
        if not isinstance(event, dict) or self.state is not AppState.PROCESSING:
            return
        self.last_progress_event = event
        processed = int(event["processed_input_frames"])
        total = int(event["total_input_frames"])
        percentage = float(event["percentage"])
        self.progress.setRange(0, 1000)
        self.progress.setValue(round(percentage * 10))
        self.progress.setFormat(f"{percentage:.1f}%")
        self.progress.setTextVisible(True)
        eta = event.get("estimated_remaining_seconds")
        if event.get("stage") == "finalising":
            eta_text = "— (finalising outputs)"
        else:
            eta_text = "Estimating…" if eta is None else self._format_duration(float(eta))
        self.progress_label.setText(f"Frames: {processed}/{total}  ETA: {eta_text}")
        self.elapsed_label.setText(f"Elapsed: {self._format_duration(float(event['elapsed_seconds']))}")

    @staticmethod
    def _format_duration(seconds_value: float) -> str:
        elapsed = max(0, int(round(seconds_value)))
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def clear_result(self) -> None:
        if self.state is AppState.PROCESSING:
            return
        self.preview_widget.clear()
        self.summary_view.clear()
        self.limitations_view.clear()
        self.log_view.clear()
        self.last_summary = None
        self.last_progress_event = None
        self.last_export_dir = None
        self.current_config = None
        self._refresh_validation()

    def _save_screenshot(self) -> None:
        if not self._completed_exportable_result() or self.preview_widget.current_artifact is None:
            QMessageBox.warning(self, "Screenshot unavailable", "Load a completed, exportable image or video frame first.")
            return
        assert self.last_summary is not None
        artifact = self.preview_widget.current_artifact
        run_dir = artifact.path.parent
        default = run_dir / "screenshots" / f"{artifact.id}_{self.preview_widget.media_time_ms:09d}ms.png"
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save current preview screenshot",
            str(default),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg)",
        )
        if not selected:
            return
        try:
            image_path, sidecar = save_screenshot(
                self.preview_widget.current_frame,
                Path(selected),
                summary=self.last_summary,
                artifact=artifact,
                media_time_ms=self.preview_widget.media_time_ms,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Screenshot failed", str(exc))
            return
        self._append_log("screenshot", f"Saved {image_path} with identity record {sidecar}")
        QMessageBox.information(self, "Screenshot saved", f"Screenshot: {image_path}\nIdentity record: {sidecar}")

    def _export_results(self) -> None:
        if not self._completed_exportable_result():
            QMessageBox.warning(self, "Export unavailable", "Only a completed, exportable run can be exported.")
            return
        assert self.last_summary is not None
        selected = QFileDialog.getExistingDirectory(self, "Select parent folder for exported result package")
        if not selected:
            return
        destination = Path(selected) / export_folder_name(self.last_summary)
        try:
            plan = build_export_plan(self.last_summary, destination)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        overwrite = False
        if plan.conflicts:
            preview = "\n".join(str(path) for path in plan.conflicts[:5])
            if len(plan.conflicts) > 5:
                preview += f"\n…and {len(plan.conflicts) - 5} more"
            answer = QMessageBox.question(
                self,
                "Export conflicts",
                "The following export targets already exist:\n"
                f"{preview}\n\nOverwrite only these manifest-declared files?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        try:
            manifest = execute_export(plan, overwrite=overwrite)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.last_export_dir = plan.destination
        self._refresh_result_actions()
        self._append_log("export", f"Exported result package: {manifest}")
        QMessageBox.information(self, "Export completed", f"Result package: {plan.destination}\nManifest: {manifest}")

    def _open_output_folder(self) -> None:
        folder = self.last_export_dir
        if folder is None and self.current_config is not None:
            folder = self.current_config.output_dir
        if folder is None or not folder.is_dir():
            QMessageBox.warning(self, "Folder unavailable", "No completed run or export folder is available.")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve()))):
            QMessageBox.warning(self, "Could not open folder", str(folder))

    def _append_log(self, channel: str, text: str) -> None:
        if text:
            self.log_view.moveCursor(self.log_view.textCursor().MoveOperation.End)
            self.log_view.insertPlainText(f"[{channel}] {text}")
            if not text.endswith("\n"):
                self.log_view.insertPlainText("\n")

    def _set_stage(self, stage: str) -> None:
        self.stage_label.setText(f"Current stage: {stage}")

    def _update_elapsed(self) -> None:
        value = self._format_duration(monotonic() - self.started_at)
        self.elapsed_label.setText(f"Elapsed: {value}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.runner.running:
            answer = QMessageBox.question(
                self,
                "Processing is active",
                "Cancel the current run process tree and close the application?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel()
        event.accept()
