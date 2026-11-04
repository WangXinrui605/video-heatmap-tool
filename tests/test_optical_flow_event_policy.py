import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from video_heatmap.optical_flow import (
    AUTOMATIC_EVENT_LABEL,
    EventPolicyState,
    FlowPair,
    advance_event_policy,
    expired_trajectory_mask,
    is_visual_discontinuity_candidate,
    next_technical_segment_id,
    read_legacy_event_summary,
)


class OpticalFlowEventPolicyTests(unittest.TestCase):
    @staticmethod
    def pair(disturbance: float, appearance: float) -> FlowPair:
        flow = np.zeros((2, 2, 2), dtype=np.float32)
        return FlowPair(flow, flow, 1.0, 1.0, disturbance, appearance)

    def test_normal_continuous_flow_is_not_candidate(self) -> None:
        self.assertFalse(is_visual_discontinuity_candidate(self.pair(0.01, 0.1), 0.03, 0.15))

    def test_rapid_continuous_motion_is_only_an_automatic_candidate(self) -> None:
        self.assertTrue(is_visual_discontinuity_candidate(self.pair(0.04, 0.2), 0.03, 0.15))
        transition = advance_event_policy(
            EventPolicyState(), candidate=True, confirmed_hard_cut=False,
            event_policy="candidate-only", recovery_stable_frames=3,
        )
        self.assertEqual(transition.action, "no_reset")
        self.assertFalse(transition.state.loss_active)

    def test_manual_hard_cut_has_distinct_reset_action(self) -> None:
        transition = advance_event_policy(
            EventPolicyState(), candidate=False, confirmed_hard_cut=True,
            event_policy="candidate-only", recovery_stable_frames=3,
        )
        self.assertEqual(transition.action, "confirmed_hard_cut_reset")
        self.assertTrue(transition.confirmed_hard_cut)

    def test_candidate_does_not_split_but_confirmed_cut_splits_technical_segment(self) -> None:
        self.assertEqual(next_technical_segment_id(4, False), 4)
        self.assertEqual(next_technical_segment_id(4, True), 5)

    def test_flow_loss_merges_consecutive_candidates_and_recovers_on_nth_stable(self) -> None:
        state = EventPolicyState()
        entered = advance_event_policy(
            state, candidate=True, confirmed_hard_cut=False,
            event_policy="flow-loss", recovery_stable_frames=3,
        )
        self.assertTrue(entered.entered_loss)
        repeated = advance_event_policy(
            entered.state, candidate=True, confirmed_hard_cut=False,
            event_policy="flow-loss", recovery_stable_frames=3,
        )
        self.assertEqual(repeated.state.loss_episode_id, entered.state.loss_episode_id)
        state = repeated.state
        for index in range(1, 4):
            transition = advance_event_policy(
                state, candidate=False, confirmed_hard_cut=False,
                event_policy="flow-loss", recovery_stable_frames=3,
            )
            state = transition.state
            self.assertEqual(transition.recovered, index == 3)
        self.assertEqual(state.flow_interval_id, 1)

    def test_p5_frames_115_116_form_two_events_one_episode(self) -> None:
        state = EventPolicyState()
        event_frames = []
        recovery = None
        episode_ids = []
        for frame in range(115, 121):
            candidate = frame in {115, 116}
            transition = advance_event_policy(
                state, candidate=candidate, confirmed_hard_cut=False,
                event_policy="flow-loss", recovery_stable_frames=3,
            )
            state = transition.state
            if candidate:
                event_frames.append(frame)
                episode_ids.append(state.loss_episode_id)
            if transition.recovered:
                recovery = frame
        self.assertEqual(event_frames, [115, 116])
        self.assertEqual(len(set(episode_ids)), 1)
        self.assertEqual(recovery, 119)

    def test_candidate_with_no_active_trajectories_still_enters_episode(self) -> None:
        transition = advance_event_policy(
            EventPolicyState(), candidate=True, confirmed_hard_cut=False,
            event_policy="flow-loss", recovery_stable_frames=1,
        )
        self.assertTrue(transition.entered_loss)
        self.assertEqual(transition.state.loss_episode_id, 0)

    def test_window_expiry_is_distinct_from_failure(self) -> None:
        mask = expired_trajectory_mask(np.array([1.0, 3.1]), frame_time=4.0, window_seconds=2.0)
        np.testing.assert_array_equal(mask, np.array([True, False]))

    def test_legacy_boundary_is_not_confirmed_cut(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({"boundaries": [{"frame": 39}]}), encoding="utf-8")
            loaded = read_legacy_event_summary(path)
        self.assertEqual(loaded["event_counts"]["confirmed_hard_cut"], 0)
        self.assertEqual(loaded["event_counts"][AUTOMATIC_EVENT_LABEL], 0)
        self.assertEqual(loaded["legacy_unreviewed_boundary_reset"], [{"frame": 39}])


if __name__ == "__main__":
    unittest.main()
