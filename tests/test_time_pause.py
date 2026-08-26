import collections
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from main import OverlayWindow


class _FakeButton:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _FakeLine:
    def __init__(self, text=""):
        self._full_text = text

    def set_text(self, text):
        self._full_text = text

    def update_data(self, *args):
        self._full_text = args


class TimePauseTests(unittest.TestCase):
    def make_window_stub(self):
        return SimpleNamespace(
            time_paused=False,
            time_pause_started=0.0,
            _resume_rebaseline_pending=False,
            time_pause_button=_FakeButton(),
            active_time_line=_FakeLine("持續練等：00:20"),
            active_session_start=100.0,
            last_gain_time=110.0,
            last_history_update=112.0,
            exp_history=collections.deque([(105.0, 1000.0), (112.0, 1100.0)]),
            recompute_visibility=lambda: None,
        )

    def test_pause_and_resume_exclude_paused_wall_clock_time(self):
        window = self.make_window_stub()

        with patch("main.time.time", return_value=120.0):
            OverlayWindow.toggle_time_pause(window)

        self.assertTrue(window.time_paused)
        self.assertEqual(window.time_pause_button.text, "時間繼續")
        self.assertIn("已停止", window.active_time_line._full_text)

        with patch("main.time.time", return_value=150.0):
            OverlayWindow.toggle_time_pause(window)

        self.assertFalse(window.time_paused)
        self.assertEqual(window.time_pause_button.text, "時間停止")
        self.assertEqual(window.active_session_start, 130.0)
        self.assertEqual(window.last_gain_time, 140.0)
        self.assertEqual(window.last_history_update, 142.0)
        self.assertEqual(list(window.exp_history), [(135.0, 1000.0), (142.0, 1100.0)])
        self.assertTrue(window._resume_rebaseline_pending)

    def test_update_stats_is_frozen_while_paused(self):
        window = SimpleNamespace(
            last_data=None,
            base_exp_for_level=[0.0, 0.0],
            time_paused=True,
            verified_abs_exp=5.0,
            exp_history=collections.deque([(10.0, 5.0)]),
        )

        OverlayWindow.update_stats(window, 1, 10.0, 66.0)

        self.assertEqual(window.verified_abs_exp, 5.0)
        self.assertEqual(list(window.exp_history), [(10.0, 5.0)])


if __name__ == "__main__":
    unittest.main()
