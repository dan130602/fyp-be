"""Boundary invariants for the inference-only refinement."""
import unittest
import numpy as np
from run_asr import RATE, guarded_bounds


class BoundaryTests(unittest.TestCase):
    def check_coverage(self, audio, maximum):
        bounds = list(guarded_bounds(audio, maximum))
        cursor = 0
        for start, end, reason in bounds:
            self.assertEqual(start, cursor)
            self.assertGreater(end, start)
            self.assertLessEqual(end-start, maximum*RATE)
            cursor = end
        self.assertEqual(cursor, len(audio))
        return bounds

    def test_sustained_pause_is_selected(self):
        audio = np.full(65*RATE+173, 0.1, dtype=np.float32)
        audio[25*RATE:26*RATE] = 0
        bounds = self.check_coverage(audio, 30)
        self.assertTrue(25*RATE < bounds[0][1] < 26*RATE)
        self.assertEqual(bounds[0][2], "sustained_pause")

    def test_continuous_speech_is_flagged_not_dropped(self):
        bounds = self.check_coverage(np.full(65*RATE+173, 0.1, dtype=np.float32), 24)
        self.assertEqual(bounds[0][1], 24*RATE)
        self.assertEqual(bounds[0][2], "hard_limit_no_safe_pause")

    def test_short_silence_and_non_frame_aligned_ends(self):
        for length in (1, RATE-1, 30*RATE+17, 61*RATE+173):
            self.check_coverage(np.zeros(length, dtype=np.float32), 30)

    def test_brief_dip_is_not_accepted(self):
        audio = np.full(40*RATE, 0.1, dtype=np.float32)
        audio[29*RATE:29*RATE+RATE//10] = 0
        self.assertEqual(self.check_coverage(audio, 30)[0][2], "hard_limit_no_safe_pause")


if __name__ == "__main__":
    unittest.main()
