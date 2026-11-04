from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QStackedWidget,
    QWidget,
)

from video_heatmap.detector_registry import list_face_detector_specs
from .config import FaceConfig, FlowConfig, Method


FIRST_RELEASE_GUI_METHODS = (Method.BASELINE, Method.FACE, Method.FLOW)


METHOD_DETAILS = {
    Method.BASELINE: {
        "label": "Screen-space cumulative baseline",
        "status": "BASELINE",
        "history": "Cumulative from the selected interval start in fixed screen coordinates.",
        "limitations": [
            "Does not follow a person or object.",
            "Cumulative history is not equivalent to optical-flow recent-window history.",
        ],
        "brush": "Brush Size (screen pixels)",
    },
    Method.FACE: {
        "label": "Face-relative cumulative — detector-only main method",
        "status": "MAIN METHOD",
        "history": "Cumulative canonical memory using unique TXT source-row assignment.",
        "limitations": [
            "Exactly one face and a single conversation partner are assumed.",
            "The tracking lifecycle retains canonical memory, briefly freezes on loss, then hides stale display.",
            "Nearest matching is current-gaze display only; a face box is not landmark or accuracy ground truth.",
        ],
        "brush": "Brush Radius (canonical canvas pixels)",
    },
    Method.FLOW: {
        "label": "Optical-flow recent-window — EXPERIMENTAL",
        "status": "EXPERIMENTAL",
        "history": "Recent window of still-propagatable pixel trajectories; not cumulative memory.",
        "limitations": [
            "Automatic candidates are observations, not hard cuts; candidate-only is the default.",
            "window_expired is normal; flow failure reasons remain separate.",
            "Does not prove object attachment or long-term identity.",
        ],
        "brush": "Brush Size (screen pixels)",
    },
}


class _ParameterWheelMixin:
    """Send parameter control wheel input to the containing scroll area.

    Keyboard input and explicit control actions retain their normal Qt behavior;
    only accidental wheel changes are redirected.
    """

    def wheelEvent(self, event: object) -> None:
        scroll_area = self.parentWidget()
        while scroll_area is not None and not isinstance(scroll_area, QScrollArea):
            scroll_area = scroll_area.parentWidget()
        if scroll_area is None:
            event.ignore()  # type: ignore[union-attr]
            return
        delta = event.angleDelta().y()  # type: ignore[union-attr]
        if delta:
            bar = scroll_area.verticalScrollBar()
            steps = round(delta / 120)
            bar.setValue(bar.value() - steps * bar.singleStep() * 3)
        event.accept()  # type: ignore[union-attr]


class ParameterSpinBox(_ParameterWheelMixin, QSpinBox):
    """A parameter spin box whose wheel scrolls its surrounding panel."""


class ParameterDoubleSpinBox(_ParameterWheelMixin, QDoubleSpinBox):
    """A parameter double spin box whose wheel scrolls its surrounding panel."""


class ParameterComboBox(_ParameterWheelMixin, QComboBox):
    """A parameter selector whose wheel scrolls its surrounding panel."""


class PathPicker(QWidget):
    changed = Signal()

    def __init__(self, caption: str, file_filter: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.caption = caption
        self.file_filter = file_filter
        self.edit = QLineEdit()
        self.button = QPushButton("Browse…")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)
        self.button.clicked.connect(self._browse)
        self.edit.textChanged.connect(self.changed)

    def _browse(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, self.caption, self.edit.text(), self.file_filter)
        if selected:
            self.edit.setText(selected)

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None


class FacePanel(QWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)
        self.provider = ParameterComboBox()
        for spec in list_face_detector_specs():
            self.provider.addItem(spec.display_name, spec.detector_id)
        self.model = PathPicker("Select face detector model", "Model files (*.onnx *.pb);;All files (*)")
        self.detector_config = PathPicker("Select detector config", "OpenCV configs (*.pbtxt);;All files (*)")
        self.config_label = QLabel("Detector config")
        self.confidence = ParameterDoubleSpinBox()
        self.confidence.setRange(0, 1)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.8)
        self.freeze = ParameterDoubleSpinBox()
        self.freeze.setRange(0, 60)
        self.freeze.setValue(0.5)
        self.freeze.setSuffix(" s")
        note = QLabel("Exactly one face; single-conversation-partner identity policy. Detector-only tracking lifecycle.")
        note.setWordWrap(True)
        form.addRow("Detector provider", self.provider)
        form.addRow("Detector model", self.model)
        form.addRow(self.config_label, self.detector_config)
        form.addRow("Detector confidence", self.confidence)
        form.addRow("Maximum freeze", self.freeze)
        form.addRow(note)
        self.provider.currentIndexChanged.connect(self._provider_changed)
        self.model.changed.connect(self.changed)
        self.detector_config.changed.connect(self.changed)
        self.confidence.valueChanged.connect(self.changed)
        self.freeze.valueChanged.connect(self.changed)
        self._provider_changed()

    def _provider_changed(self) -> None:
        dnn_ssd = self.provider.currentData() == "opencv-dnn-ssd"
        self.config_label.setVisible(dnn_ssd)
        self.detector_config.setVisible(dnn_ssd)
        self.confidence.setValue(0.7 if dnn_ssd else 0.8)
        self.changed.emit()

    def config(self) -> FaceConfig:
        detector_id = str(self.provider.currentData())
        return FaceConfig(
            self.model.path() or Path(""),
            self.confidence.value(),
            self.freeze.value(),
            detector_id,
            self.detector_config.path() if detector_id == "opencv-dnn-ssd" else None,
        )


class FlowPanel(QWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        form = QFormLayout(self)
        self.window_seconds = ParameterDoubleSpinBox()
        self.window_seconds.setRange(0.1, 3600)
        self.window_seconds.setValue(5)
        self.window_seconds.setSuffix(" s")
        self.event_policy = ParameterComboBox()
        self.event_policy.addItems(["candidate-only", "flow-loss"])
        self.recovery_frames = ParameterSpinBox()
        self.recovery_frames.setRange(1, 1000)
        self.recovery_frames.setValue(3)
        self.auto_event = QCheckBox("Automatic visual-discontinuity candidate observation enabled")
        self.auto_event.setChecked(True)
        self.manual_boundaries = PathPicker("Select manual boundaries", "CSV/JSON files (*.csv *.json);;All files (*)")
        self.event_annotations = PathPicker("Select reviewed event annotations", "CSV/JSON files (*.csv *.json);;All files (*)")
        note = QLabel("EXPERIMENTAL recent-window flow. Automatic candidates are observations, not hard cuts or resets by default.")
        note.setWordWrap(True)
        form.addRow("Window", self.window_seconds)
        form.addRow("Event policy", self.event_policy)
        form.addRow("Recovery stable frames", self.recovery_frames)
        form.addRow(self.auto_event)
        form.addRow("Manual boundaries (optional)", self.manual_boundaries)
        form.addRow("Event annotations (optional)", self.event_annotations)
        form.addRow(note)
        for signal in (
            self.window_seconds.valueChanged,
            self.event_policy.currentIndexChanged,
            self.recovery_frames.valueChanged,
            self.auto_event.toggled,
            self.manual_boundaries.changed,
            self.event_annotations.changed,
        ):
            signal.connect(self.changed)

    def config(self) -> FlowConfig:
        return FlowConfig(
            self.window_seconds.value(),
            self.event_policy.currentText(),
            self.recovery_frames.value(),
            self.auto_event.isChecked(),
            self.manual_boundaries.path(),
            self.event_annotations.path(),
        )


class MethodPanels(QStackedWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.empty = QWidget()
        self.face = FacePanel()
        self.flow = FlowPanel()
        self._indices = {
            Method.BASELINE: self.addWidget(self.empty),
            Method.FACE: self.addWidget(self.face),
            Method.FLOW: self.addWidget(self.flow),
        }
        self.face.changed.connect(self.changed)
        self.flow.changed.connect(self.changed)

    def set_method(self, method: Method) -> None:
        self.setCurrentIndex(self._indices[method])
