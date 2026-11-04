from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrackingPhase(str, Enum):
    TRACKING = "tracking"
    SHORT_LOSS_FREEZE = "short_loss_freeze"
    TARGET_NOT_VISIBLE = "target_not_visible"
    REACQUIRED = "reacquired"


@dataclass(frozen=True)
class TrackingDecision:
    phase: TrackingPhase
    reason: str
    accept_region_gaze: bool
    render_current_overlay: bool
    render_frozen_frame: bool
    reacquired: bool = False


@dataclass
class LossEpisode:
    start_frame: int
    sample_count_before_loss: int
    last_lost_frame: int
    freeze_visible_frames: int = 0
    hidden_retained_frames: int = 0
    began_after_acquisition: bool = True
    recovery_frame: int | None = None
    reacquisition_frame: int | None = None
    sample_count_at_reacquisition: int | None = None
    sample_count_after_reacquisition: int | None = None

    @property
    def lost_frames(self) -> int:
        return self.last_lost_frame - self.start_frame + 1

    def as_dict(self, fps: float) -> dict[str, int | float | None]:
        return {
            "start_frame": self.start_frame,
            "last_lost_frame": self.last_lost_frame,
            "lost_frames": self.lost_frames,
            "loss_duration_seconds": self.lost_frames / fps,
            "freeze_visible_frames": self.freeze_visible_frames,
            "freeze_visible_seconds": self.freeze_visible_frames / fps,
            "hidden_retained_frames": self.hidden_retained_frames,
            "hidden_retained_seconds": self.hidden_retained_frames / fps,
            "began_after_acquisition": self.began_after_acquisition,
            "recovery_frame": self.recovery_frame,
            "reacquisition_frame": self.reacquisition_frame,
            "sample_count_before_loss": self.sample_count_before_loss,
            "sample_count_at_reacquisition": self.sample_count_at_reacquisition,
            "sample_count_after_reacquisition": self.sample_count_after_reacquisition,
        }


class TrackingLifecycle:
    """Detector-only target visibility and canonical-memory lifecycle.

    The first version assumes one main conversation partner. Exactly one face is
    required for identity-confident rendering and accumulation. Zero faces means
    detector loss; multiple faces mean identity uncertainty. Neither case accepts
    region-relative gaze.
    """

    IDENTITY_POLICY = "single-conversation-partner; exactly-one-face-required"

    def __init__(self, *, max_freeze_frames: int) -> None:
        if max_freeze_frames < 0:
            raise ValueError("max_freeze_frames must not be negative")
        self.max_freeze_frames = int(max_freeze_frames)
        self.phase = TrackingPhase.TARGET_NOT_VISIBLE
        self.has_acquired_target = False
        self.reacquisition_count = 0
        self.loss_episodes: list[LossEpisode] = []
        self._active_loss: LossEpisode | None = None
        self._pending_reacquisition: LossEpisode | None = None
        self._last_frame_index: int | None = None
        self._freeze_available = False
        self.total_lost_frames = 0
        self.freeze_visible_frames = 0
        self.hidden_retained_frames = 0
        self.identity_uncertain_frames = 0
        self.hard_cut_frames = 0

    def _check_frame_index(self, frame_index: int) -> None:
        if frame_index < 0:
            raise ValueError("frame_index must not be negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase strictly")
        self._last_frame_index = frame_index

    def step(
        self,
        *,
        frame_index: int,
        detection_count: int,
        canonical_sample_count: int,
        hard_cut: bool = False,
    ) -> TrackingDecision:
        self._check_frame_index(frame_index)
        if detection_count < 0:
            raise ValueError("detection_count must not be negative")
        if canonical_sample_count < 0:
            raise ValueError("canonical_sample_count must not be negative")

        if hard_cut:
            self.hard_cut_frames += 1
            self._freeze_available = False

        identity_confident = detection_count == 1
        if identity_confident:
            self._freeze_available = True
            if not self.has_acquired_target:
                if self._active_loss is not None:
                    episode = self._active_loss
                    episode.recovery_frame = frame_index
                    episode.sample_count_at_reacquisition = canonical_sample_count
                    self.loss_episodes.append(episode)
                    self._active_loss = None
                    self._pending_reacquisition = episode
                self.has_acquired_target = True
                self.phase = TrackingPhase.TRACKING
                reason = "initial_target_acquired"
                reacquired = False
            elif self._active_loss is not None:
                episode = self._active_loss
                episode.recovery_frame = frame_index
                episode.reacquisition_frame = frame_index
                episode.sample_count_at_reacquisition = canonical_sample_count
                self.loss_episodes.append(episode)
                self._active_loss = None
                self._pending_reacquisition = episode
                self.reacquisition_count += 1
                self.phase = TrackingPhase.REACQUIRED
                reason = "same_partner_assumption_reacquired"
                reacquired = True
            else:
                self.phase = TrackingPhase.TRACKING
                reason = "single_face_detected"
                reacquired = False
            return TrackingDecision(
                phase=self.phase,
                reason=reason,
                accept_region_gaze=True,
                render_current_overlay=True,
                render_frozen_frame=False,
                reacquired=reacquired,
            )

        if detection_count > 1:
            self.identity_uncertain_frames += 1
            reason = "identity_uncertain_multiple_faces"
        elif hard_cut:
            reason = "confirmed_hard_cut_transient_state_cleared"
        else:
            reason = "detector_missing"

        if self._active_loss is None:
            self._active_loss = LossEpisode(
                start_frame=frame_index,
                last_lost_frame=frame_index,
                sample_count_before_loss=canonical_sample_count,
                began_after_acquisition=self.has_acquired_target,
            )
        elif self._active_loss is not None:
            self._active_loss.last_lost_frame = frame_index

        self.total_lost_frames += 1
        lost_frames = self._active_loss.lost_frames
        can_freeze = (
            self._freeze_available
            and self._active_loss.began_after_acquisition
            and lost_frames <= self.max_freeze_frames
        )
        if can_freeze:
            self.phase = TrackingPhase.SHORT_LOSS_FREEZE
            self.freeze_visible_frames += 1
            self._active_loss.freeze_visible_frames += 1
        else:
            self.phase = TrackingPhase.TARGET_NOT_VISIBLE
            self.hidden_retained_frames += 1
            self._active_loss.hidden_retained_frames += 1

        return TrackingDecision(
            phase=self.phase,
            reason=reason,
            accept_region_gaze=False,
            render_current_overlay=False,
            render_frozen_frame=can_freeze,
        )

    def record_post_frame_sample_count(self, canonical_sample_count: int) -> None:
        if canonical_sample_count < 0:
            raise ValueError("canonical_sample_count must not be negative")
        if self._pending_reacquisition is not None:
            self._pending_reacquisition.sample_count_after_reacquisition = (
                canonical_sample_count
            )
            self._pending_reacquisition = None

    def summary(self, *, fps: float) -> dict[str, object]:
        if fps <= 0:
            raise ValueError("fps must be positive")
        episodes = [episode.as_dict(fps) for episode in self.loss_episodes]
        if self._active_loss is not None:
            episodes.append(self._active_loss.as_dict(fps))
        return {
            "identity_policy": self.IDENTITY_POLICY,
            "max_freeze_frames": self.max_freeze_frames,
            "max_freeze_seconds": self.max_freeze_frames / fps,
            "current_phase": self.phase.value,
            "loss_episode_count": len(episodes),
            "reacquisition_count": self.reacquisition_count,
            "total_lost_frames": self.total_lost_frames,
            "total_loss_duration_seconds": self.total_lost_frames / fps,
            "freeze_visible_frames": self.freeze_visible_frames,
            "freeze_visible_seconds": self.freeze_visible_frames / fps,
            "hidden_retained_frames": self.hidden_retained_frames,
            "hidden_retained_seconds": self.hidden_retained_frames / fps,
            "identity_uncertain_frames": self.identity_uncertain_frames,
            "hard_cut_frames": self.hard_cut_frames,
            "episodes": episodes,
        }
