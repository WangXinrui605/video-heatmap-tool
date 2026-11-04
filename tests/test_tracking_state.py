from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_heatmap.tracking_state import TrackingLifecycle, TrackingPhase


class TrackingLifecycleTests(unittest.TestCase):
    def test_valid_lost_valid_freezes_then_reacquires(self) -> None:
        lifecycle = TrackingLifecycle(max_freeze_frames=2)
        first = lifecycle.step(frame_index=0, detection_count=1, canonical_sample_count=3)
        frozen = lifecycle.step(frame_index=1, detection_count=0, canonical_sample_count=4)
        reacquired = lifecycle.step(
            frame_index=2, detection_count=1, canonical_sample_count=4
        )
        lifecycle.record_post_frame_sample_count(5)

        self.assertEqual(first.phase, TrackingPhase.TRACKING)
        self.assertTrue(frozen.render_frozen_frame)
        self.assertFalse(frozen.accept_region_gaze)
        self.assertEqual(reacquired.phase, TrackingPhase.REACQUIRED)
        self.assertTrue(reacquired.accept_region_gaze)
        summary = lifecycle.summary(fps=10.0)
        self.assertEqual(summary["reacquisition_count"], 1)
        episode = summary["episodes"][0]
        self.assertEqual(episode["sample_count_before_loss"], 4)
        self.assertEqual(episode["sample_count_at_reacquisition"], 4)
        self.assertEqual(episode["sample_count_after_reacquisition"], 5)

    def test_beginning_with_loss_never_freezes_without_valid_frame(self) -> None:
        lifecycle = TrackingLifecycle(max_freeze_frames=3)
        decision = lifecycle.step(
            frame_index=0, detection_count=0, canonical_sample_count=0
        )
        self.assertEqual(decision.phase, TrackingPhase.TARGET_NOT_VISIBLE)
        self.assertFalse(decision.render_frozen_frame)
        summary = lifecycle.summary(fps=30.0)
        self.assertEqual(summary["loss_episode_count"], 1)
        self.assertFalse(summary["episodes"][0]["began_after_acquisition"])

    def test_permanent_loss_hides_after_grace_and_retains_samples(self) -> None:
        lifecycle = TrackingLifecycle(max_freeze_frames=2)
        lifecycle.step(frame_index=0, detection_count=1, canonical_sample_count=7)
        decisions = [
            lifecycle.step(frame_index=frame, detection_count=0, canonical_sample_count=7)
            for frame in range(1, 5)
        ]
        self.assertEqual(
            [decision.phase for decision in decisions],
            [
                TrackingPhase.SHORT_LOSS_FREEZE,
                TrackingPhase.SHORT_LOSS_FREEZE,
                TrackingPhase.TARGET_NOT_VISIBLE,
                TrackingPhase.TARGET_NOT_VISIBLE,
            ],
        )
        summary = lifecycle.summary(fps=20.0)
        self.assertEqual(summary["freeze_visible_frames"], 2)
        self.assertEqual(summary["hidden_retained_frames"], 2)
        self.assertEqual(summary["episodes"][0]["sample_count_before_loss"], 7)

    def test_multiple_short_losses_are_separate_episodes(self) -> None:
        lifecycle = TrackingLifecycle(max_freeze_frames=1)
        counts = [1, 0, 1, 1, 0, 1]
        for frame, count in enumerate(counts):
            lifecycle.step(
                frame_index=frame, detection_count=count, canonical_sample_count=2
            )
            lifecycle.record_post_frame_sample_count(2)
        summary = lifecycle.summary(fps=10.0)
        self.assertEqual(summary["loss_episode_count"], 2)
        self.assertEqual(summary["reacquisition_count"], 2)

    def test_long_unrelated_scene_does_not_render_but_reuses_memory(self) -> None:
        lifecycle = TrackingLifecycle(max_freeze_frames=1)
        lifecycle.step(frame_index=0, detection_count=1, canonical_sample_count=9)
        lifecycle.step(frame_index=1, detection_count=0, canonical_sample_count=9)
        hidden = lifecycle.step(
            frame_index=2, detection_count=0, canonical_sample_count=9
        )
        reacquired = lifecycle.step(
            frame_index=3, detection_count=1, canonical_sample_count=9
        )
        self.assertFalse(hidden.render_current_overlay)
        self.assertFalse(hidden.render_frozen_frame)
        self.assertTrue(reacquired.render_current_overlay)
        self.assertEqual(
            lifecycle.summary(fps=10.0)["episodes"][0][
                "sample_count_at_reacquisition"
            ],
            9,
        )

    def test_hard_cut_clears_freeze_not_canonical_memory(self) -> None:
        lifecycle = TrackingLifecycle(max_freeze_frames=5)
        lifecycle.step(frame_index=0, detection_count=1, canonical_sample_count=11)
        cut = lifecycle.step(
            frame_index=1,
            detection_count=0,
            canonical_sample_count=11,
            hard_cut=True,
        )
        self.assertEqual(cut.phase, TrackingPhase.TARGET_NOT_VISIBLE)
        self.assertFalse(cut.render_frozen_frame)
        self.assertEqual(
            lifecycle.summary(fps=10.0)["episodes"][0][
                "sample_count_before_loss"
            ],
            11,
        )

    def test_multiple_faces_are_identity_uncertain_and_do_not_merge(self) -> None:
        lifecycle = TrackingLifecycle(max_freeze_frames=0)
        lifecycle.step(frame_index=0, detection_count=1, canonical_sample_count=4)
        uncertain = lifecycle.step(
            frame_index=1, detection_count=2, canonical_sample_count=4
        )
        self.assertEqual(uncertain.reason, "identity_uncertain_multiple_faces")
        self.assertFalse(uncertain.accept_region_gaze)
        self.assertFalse(uncertain.render_current_overlay)
        self.assertEqual(lifecycle.summary(fps=10.0)["identity_uncertain_frames"], 1)


if __name__ == "__main__":
    unittest.main()
