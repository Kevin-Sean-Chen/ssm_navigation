"""Tests for strain kinematics and signal-response measures."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gap_crossing import strain_kinematics as kinematics


def make_track(signal, track_id="2026/a::track0", vx=-5.0, theta=180.0, jumps=None):
    """Return a processed track with constant motion."""
    signal = np.asarray(signal, dtype=float)
    frame_count = len(signal)
    time_s = np.arange(frame_count) / kinematics.FRAME_RATE_HZ
    x = 100 + vx * time_s
    return {
        "track_id": track_id,
        "source_file": track_id.split("::")[0],
        "xy": np.column_stack([x, np.zeros(frame_count)]),
        "velocity": np.column_stack([np.full(frame_count, vx), np.zeros(frame_count)]),
        "time_s": time_s,
        "signal": signal,
        "speed_smooth": np.full(frame_count, abs(vx)),
        "theta_smooth": np.full(frame_count, theta),
        "dtheta_smooth": np.full(frame_count, 10.0),
        "jumps": np.zeros(frame_count, dtype=bool) if jumps is None else jumps,
    }


class FrameTests(unittest.TestCase):
    """Check state labels and derived quantities."""

    def test_states_follow_signal_exit_time(self):
        """Frames after an exit are post-signal until POST_SIGNAL_S passes."""
        in_signal = np.array([0, 1, 1, 0, 0, 0, 0], dtype=bool)
        time_s = np.arange(7, dtype=float)

        states = kinematics.label_states(in_signal, time_s, post_signal_s=2.0)

        np.testing.assert_array_equal(states, [2, 0, 0, 1, 1, 1, 2])

    def test_upwind_motion_has_heading_zero_and_alignment_one(self):
        """A fly moving toward -x with theta 180 is aligned upwind."""
        frames = kinematics.describe_track(make_track(np.zeros(10), vx=-5.0, theta=180.0))

        np.testing.assert_allclose(frames["upwind_velocity"], 5.0)
        np.testing.assert_allclose(frames["heading_from_upwind"], 0.0)
        np.testing.assert_allclose(frames["upwind_alignment"], 1.0)

    def test_jump_frames_are_masked(self):
        """Quantities at jump frames are NaN; signal state is kept."""
        jumps = np.zeros(5, dtype=bool)
        jumps[2] = True
        frames = kinematics.describe_track(make_track([0, 1, 1, 0, 0], jumps=jumps))

        self.assertTrue(np.isnan(frames["speed"][2]))
        self.assertTrue(frames["in_signal"][2])
        self.assertFalse(frames["valid"][2])


class BoutTests(unittest.TestCase):
    """Check bouts, intervals, and censoring."""

    def test_bouts_and_intervals_mark_censoring(self):
        """Bouts at track edges and the final interval are censored."""
        signal = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
        frames = kinematics.describe_track(make_track(signal))

        bouts, intervals = kinematics.make_track_bouts(frames, "track")

        self.assertEqual([bout["left_censored"] for bout in bouts], [True, False])
        self.assertEqual([bout["right_censored"] for bout in bouts], [False, False])
        np.testing.assert_allclose(
            [bout["duration_s"] for bout in bouts], np.array([2, 3]) / kinematics.FRAME_RATE_HZ
        )
        np.testing.assert_allclose(
            [interval["duration_s"] for interval in intervals],
            np.array([3, 2]) / kinematics.FRAME_RATE_HZ,
        )
        self.assertEqual([interval["censored"] for interval in intervals], [False, True])

    def test_bout_upwind_displacement_is_positive_for_upwind_motion(self):
        """Moving toward -x during a bout gives positive upwind displacement."""
        frames = kinematics.describe_track(make_track([0, 1, 1, 1, 0], vx=-6.0))

        bouts, _ = kinematics.make_track_bouts(frames, "track")

        self.assertGreater(bouts[0]["upwind_displacement_mm"], 0)


class JumpTests(unittest.TestCase):
    """Check signal bouts and events around tracking jumps."""

    def test_signal_state_carries_through_a_jump(self):
        """A jump inside a bout does not split it into an exit and an entry."""
        signal = np.array([0, 1, 1, np.nan, np.nan, 1, 0, 0], dtype=float)
        jumps = np.isnan(signal)
        frames = kinematics.describe_track(make_track(signal, jumps=jumps))

        np.testing.assert_array_equal(frames["in_signal"], [0, 1, 1, 1, 1, 1, 0, 0])
        self.assertTrue(np.isnan(frames["speed"][3]))
        self.assertTrue(np.isnan(frames["signal_present"][3]))
        self.assertEqual(frames["signal_present"][5], 1.0)

    def test_bout_and_interval_edges_at_jumps_are_censored(self):
        """Bout edges next to a jump are censored; intervals stop at the jump."""
        signal = np.array([0, 1, 1, 0, 0, np.nan, 0, 1, 0, 0], dtype=float)
        jumps = np.isnan(signal)
        frames = kinematics.describe_track(make_track(signal, jumps=jumps))

        bouts, intervals = kinematics.make_track_bouts(frames, "track")

        self.assertEqual([bout["left_censored"] for bout in bouts], [False, False])
        self.assertEqual(
            [(interval["duration_s"] * kinematics.FRAME_RATE_HZ, interval["censored"]) for interval in intervals],
            [(2.0, True), (2.0, True)],
        )

    def test_profile_events_skip_jumps(self):
        """Onsets with a jump in the blank period and offsets at jumps are dropped."""
        blank = int(kinematics.MIN_PRIOR_BLANK_S * kinematics.FRAME_RATE_HZ)
        in_signal = np.r_[np.zeros(blank), np.ones(5), np.zeros(blank), np.ones(5), np.zeros(5)].astype(bool)
        jumps = np.zeros(len(in_signal), dtype=bool)
        jumps[blank + 5 + 3] = True
        jumps[blank + 4] = True

        events = kinematics.find_profile_events(in_signal, jumps)

        np.testing.assert_array_equal(events["onset"], [blank])
        np.testing.assert_array_equal(events["offset"], [2 * blank + 10])


class DurationBinTests(unittest.TestCase):
    """Check frame-aligned duration bins."""

    def test_every_bin_holds_a_whole_frame_duration(self):
        """No log bin is narrower than the frame step."""
        bins = kinematics.make_frame_duration_bins(60)
        frames = np.arange(1, 60 * kinematics.FRAME_RATE_HZ + 1) / kinematics.FRAME_RATE_HZ
        counts, _ = np.histogram(frames, bins=bins)

        self.assertTrue((counts > 0).all())
        self.assertAlmostEqual(bins[0], 0.5 / kinematics.FRAME_RATE_HZ)


class ProfileTests(unittest.TestCase):
    """Check event-triggered windows."""

    def test_onsets_need_a_blank_period(self):
        """An onset right after a short gap is not used; offsets inside the track are."""
        blank = int(kinematics.MIN_PRIOR_BLANK_S * kinematics.FRAME_RATE_HZ)
        in_signal = np.r_[np.zeros(blank), np.ones(5), np.zeros(3), np.ones(5), np.zeros(5)].astype(bool)

        events = kinematics.find_profile_events(in_signal)

        np.testing.assert_array_equal(events["onset"], [blank])
        np.testing.assert_array_equal(events["offset"], [blank + 5, blank + 13])

    def test_windows_are_nan_outside_the_track(self):
        """Window samples before the start or after the end are NaN."""
        windows = kinematics.gather_windows(np.arange(5.0), np.array([1]), np.array([-2, -1, 0, 4]))

        np.testing.assert_array_equal(windows, [[np.nan, 0.0, 1.0, np.nan]])


class SessionTests(unittest.TestCase):
    """Check one full session analysis."""

    def tearDown(self):
        plt.close("all")

    def test_session_tables_and_plots(self):
        """A synthetic two-strain run produces every table and figure."""
        rate = kinematics.FRAME_RATE_HZ
        pattern = np.r_[np.zeros(2 * rate), np.ones(rate // 2), np.zeros(3 * rate)]
        signal = np.tile(pattern, 8)
        tracks = []
        rows = []
        for label, vx, vial in [("empty", -5.0, 0), ("fc2", -2.0, 1)]:
            for index in range(3):
                track_id = f"2026/{label}{index}::track0"
                tracks.append({**make_track(signal, track_id, vx=vx), "dataset_label": label})
                rows.append({
                    "source_file": f"2026/{label}{index}", "dataset_label": label,
                    "recording_year": 2026, "recording_month": 8, "recording_day": 19 + index,
                    "recording_experimenter": "kevin", "recording_vial": vial,
                })
        recordings = pd.DataFrame(rows)

        with patch.object(kinematics, "MIN_STATE_FRAMES", 10):
            tables = kinematics.analyze_sessions(tracks, recordings)
            rng = np.random.default_rng(0)
            with (
                patch.object(kinematics, "BOOTSTRAP_SAMPLES", 200),
                patch.object(kinematics, "PROFILE_BOOTSTRAP_SAMPLES", 50),
                patch.object(kinematics, "PERMUTATION_COUNT", 200),
            ):
                summaries = kinematics.summarize(tables, rng)
                comparisons = kinematics.compare_strains(tables, "empty", rng)

        state = tables["state_metrics"]
        upwind = state.loc[(state["metric"] == "upwind_velocity_mean") & (state["state"] == "in_signal")]
        self.assertEqual(upwind.groupby("dataset_label")["value"].mean().to_dict(), {"empty": 5.0, "fc2": 2.0})
        self.assertEqual(tables["bout_metrics"]["session_id"].nunique(), 6)
        self.assertFalse(tables["profiles"].empty)
        upwind_comparison = comparisons.loc[
            (comparisons["metric"] == "upwind_velocity_mean") & (comparisons["state"] == "in_signal")
        ]
        self.assertAlmostEqual(upwind_comparison["difference"].iloc[0], -3.0)

        order = ["empty", "fc2"]
        colors = {"empty": "C0", "fc2": "C1"}
        figure_count_before = len(plt.get_fignums())
        kinematics.plot_state_distributions(summaries["state_histograms"], order, colors)
        kinematics.plot_state_metrics(state, summaries["state_metrics"], order, colors)
        kinematics.plot_bout_metrics(
            tables["bout_metrics"], summaries["bout_metrics"],
            summaries["duration_histograms"], order, colors,
        )
        for event in kinematics.EVENT_TYPES:
            kinematics.plot_profiles(summaries["profiles"], event, order, colors)
        self.assertEqual(len(plt.get_fignums()) - figure_count_before, 5)


if __name__ == "__main__":
    unittest.main()
