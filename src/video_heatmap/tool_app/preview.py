from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Any

from PySide6.QtCore import QSignalBlocker, QUrl, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QVideoFrame, QVideoSink
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .artifacts import ARTIFACT_SCHEMA_VERSION, RUN_SCHEMA_VERSION, write_json_atomic


SCREENSHOT_SCHEMA_VERSION = "video_heatmap_screenshot.v1"


@dataclass(frozen=True)
class PreviewArtifact:
    id: str
    kind: str
    role: str
    label: str
    path: Path
    media_type: str
    exportable: bool


def _resolved_run_path(manifest: dict[str, Any], item: dict[str, Any]) -> Path:
    if item.get("path_base") != "run_directory":
        raise ValueError(f"Artifact {item.get('id', '<unknown>')} has an unsupported path base.")
    base_text = manifest.get("path_base")
    if not isinstance(base_text, str) or not base_text:
        raise ValueError("The artifact manifest does not define its run directory.")
    relative = Path(str(item.get("path", "")))
    if not str(relative) or relative.is_absolute():
        raise ValueError(f"Artifact {item.get('id', '<unknown>')} has an invalid relative path.")
    base = Path(base_text).resolve()
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Artifact {item.get('id', '<unknown>')} escapes its run directory.") from exc
    return resolved


def preview_artifacts(summary: dict[str, Any]) -> list[PreviewArtifact]:
    """Return only manifest-declared media from a completed, exportable run."""

    if summary.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("Preview requires video_heatmap_tool_run.v2.")
    if summary.get("status") != "completed" or summary.get("exportable") is not True:
        raise ValueError("Preview is available only for a completed, exportable run.")
    manifest = summary.get("artifacts")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Preview requires a validated video_heatmap_artifacts.v1 manifest.")
    items = manifest.get("artifacts")
    if not isinstance(items, list):
        raise ValueError("The artifact manifest does not contain an artifact list.")

    result: list[PreviewArtifact] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("previewable") is not True:
            continue
        artifact_id = str(item.get("id", ""))
        if not artifact_id or artifact_id in seen:
            raise ValueError("Preview artifact IDs must be non-empty and unique.")
        media_type = str(item.get("media_type", ""))
        kind = str(item.get("kind", ""))
        if kind not in {"image", "video"} or not (
            media_type.startswith("image/") or media_type.startswith("video/")
        ):
            raise ValueError(f"Preview artifact {artifact_id} is not supported image/video media.")
        path = _resolved_run_path(manifest, item)
        if item.get("exists") is not True or not path.is_file():
            raise ValueError(f"Preview artifact is missing: {path}")
        seen.add(artifact_id)
        result.append(PreviewArtifact(
            id=artifact_id,
            kind=kind,
            role=str(item.get("role", "")),
            label=str(item.get("label", artifact_id)),
            path=path,
            media_type=media_type,
            exportable=item.get("exportable") is True,
        ))
    return result


def save_screenshot(
    image: QImage,
    destination: Path,
    *,
    summary: dict[str, Any],
    artifact: PreviewArtifact,
    media_time_ms: int,
) -> tuple[Path, Path]:
    """Save a preview frame plus an identity sidecar without changing its source."""

    available = {item.id: item for item in preview_artifacts(summary)}
    if (
        artifact.id not in available
        or available[artifact.id].path != artifact.path
        or not available[artifact.id].exportable
    ):
        raise ValueError("Screenshot source is not a previewable artifact in this run manifest.")
    if image.isNull():
        raise ValueError("No image or decoded video frame is available to save.")
    destination = destination.resolve()
    if destination.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        destination = destination.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        if not image.save(str(temporary)):
            raise OSError(f"Qt could not encode screenshot: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    sidecar = destination.with_suffix(destination.suffix + ".json")
    write_json_atomic(sidecar, {
        "schema_version": SCREENSHOT_SCHEMA_VERSION,
        "source_run_id": summary.get("run_id"),
        "source_method": summary.get("method"),
        "source_artifact": {
            "id": artifact.id,
            "role": artifact.role,
            "label": artifact.label,
            "path": str(artifact.path),
            "media_type": artifact.media_type,
        },
        "media_time_ms": max(0, int(media_time_ms)),
        "saved_path": str(destination),
        "saved_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    })
    return destination, sidecar


class _FitImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("No completed artifact loaded")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setStyleSheet("background: #181818; color: #cccccc;")
        self._image = QImage()

    @property
    def image(self) -> QImage:
        return self._image.copy()

    def set_image(self, image: QImage) -> None:
        self._image = image.copy()
        self._update_pixmap()

    def clear_image(self) -> None:
        self._image = QImage()
        self.clear()
        self.setText("No completed artifact loaded")

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._image.isNull():
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(pixmap)


class PreviewWidget(QWidget):
    frame_availability_changed = Signal(bool)
    artifact_availability_changed = Signal(bool)
    error_occurred = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.summary: dict[str, Any] | None = None
        self.artifacts: list[PreviewArtifact] = []
        self.current_artifact: PreviewArtifact | None = None
        self._current_frame = QImage()

        layout = QVBoxLayout(self)
        self.output_to_preview_label = QLabel("Output to preview")
        self.output_to_preview_label.setStyleSheet("font-weight: bold;")
        self.artifact_combo = QComboBox()
        self.preview_empty_state = QLabel(
            "Outputs appear here after successful generation. Only outputs from a completed, "
            "exportable run can be selected for preview."
        )
        self.preview_empty_state.setWordWrap(True)
        layout.addWidget(self.output_to_preview_label)
        layout.addWidget(self.artifact_combo)
        layout.addWidget(self.preview_empty_state)
        self.image_label = _FitImageLabel()
        layout.addWidget(self.image_label, 1)

        self.player = QMediaPlayer(self)
        self.video_sink = QVideoSink(self)
        self.player.setVideoOutput(self.video_sink)
        self.video_sink.videoFrameChanged.connect(self._video_frame_changed)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.time_label = QLabel("00:00 / 00:00")
        self.speed_combo = QComboBox()
        for label, rate in (("0.5x", 0.5), ("1.0x", 1.0), ("1.5x", 1.5), ("2.0x", 2.0)):
            self.speed_combo.addItem(label, rate)
        self.speed_combo.setCurrentIndex(1)
        controls.addWidget(self.play_button)
        controls.addWidget(self.seek_slider, 1)
        controls.addWidget(self.time_label)
        controls.addWidget(QLabel("Preview Speed"))
        controls.addWidget(self.speed_combo)
        layout.addLayout(controls)

        self.artifact_combo.currentIndexChanged.connect(self.load_current)
        self.play_button.clicked.connect(self.toggle_playback)
        self.seek_slider.sliderMoved.connect(self.player.setPosition)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackStateChanged.connect(self._playback_state_changed)
        self.player.errorOccurred.connect(self._media_error)
        self._set_video_controls(False)

    @property
    def current_frame(self) -> QImage:
        return self._current_frame.copy()

    @property
    def media_time_ms(self) -> int:
        return int(self.player.position()) if self.current_artifact and self.current_artifact.kind == "video" else 0

    def set_summary(self, summary: dict[str, Any]) -> None:
        artifacts = preview_artifacts(summary)
        self.clear()
        self.summary = summary
        self.artifacts = artifacts
        with QSignalBlocker(self.artifact_combo):
            for item in artifacts:
                self.artifact_combo.addItem(item.label, item.id)
        available = bool(artifacts)
        self.preview_empty_state.setVisible(not available)
        self.artifact_availability_changed.emit(available)
        if available:
            self.artifact_combo.setCurrentIndex(0)
            self.load_current()

    def clear(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        self.summary = None
        self.artifacts = []
        self.current_artifact = None
        self.artifact_combo.clear()
        self.preview_empty_state.setVisible(True)
        self._set_frame(QImage())
        self.image_label.clear_image()
        self.seek_slider.setRange(0, 0)
        self.time_label.setText("00:00 / 00:00")
        self._set_video_controls(False)
        self.artifact_availability_changed.emit(False)

    def load_current(self) -> None:
        index = self.artifact_combo.currentIndex()
        if index < 0 or index >= len(self.artifacts):
            return
        self.player.stop()
        self.current_artifact = self.artifacts[index]
        item = self.current_artifact
        if item.kind == "image":
            self.player.setSource(QUrl())
            image = QImage(str(item.path))
            if image.isNull():
                self._set_frame(QImage())
                self.image_label.clear_image()
                message = f"Qt could not decode preview image: {item.path}"
                self.image_label.setText(message)
                self.error_occurred.emit(message)
                return
            self._set_frame(image)
            self.image_label.set_image(image)
            self.seek_slider.setRange(0, 0)
            self.time_label.setText("Image")
            self._set_video_controls(False)
        else:
            self._set_frame(QImage())
            self.image_label.clear_image()
            self.image_label.setText("Loading video preview…")
            self._set_video_controls(True)
            self.player.setSource(QUrl.fromLocalFile(str(item.path)))
            self.player.setPlaybackRate(float(self.speed_combo.currentData()))

    def toggle_playback(self) -> None:
        if not self.current_artifact or self.current_artifact.kind != "video":
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _set_video_controls(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.seek_slider.setEnabled(enabled)
        self.speed_combo.setEnabled(enabled)

    def _set_frame(self, image: QImage) -> None:
        self._current_frame = image.copy()
        self.frame_availability_changed.emit(not image.isNull())

    def _video_frame_changed(self, frame: QVideoFrame) -> None:
        image = frame.toImage()
        if image.isNull():
            return
        self._set_frame(image)
        self.image_label.set_image(image)

    def _speed_changed(self) -> None:
        rate = self.speed_combo.currentData()
        if rate is not None:
            self.player.setPlaybackRate(float(rate))

    def _position_changed(self, position: int) -> None:
        if not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(position)
        self.time_label.setText(f"{self._format_time(position)} / {self._format_time(self.player.duration())}")

    def _duration_changed(self, duration: int) -> None:
        self.seek_slider.setRange(0, max(0, duration))
        self._position_changed(self.player.position())

    def _playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.play_button.setText("Pause" if state == QMediaPlayer.PlaybackState.PlayingState else "Play")

    def _media_error(self, _error: QMediaPlayer.Error, error_string: str) -> None:
        if not error_string:
            return
        message = f"Video preview error: {error_string}"
        if self.current_artifact is not None:
            message += f" ({self.current_artifact.path})"
        self.image_label.setText(message)
        self.error_occurred.emit(message)

    @staticmethod
    def _format_time(milliseconds: int) -> str:
        total_seconds = max(0, int(milliseconds)) // 1000
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
