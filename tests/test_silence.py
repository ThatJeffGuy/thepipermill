"""Silence trimming: edges and long inner pauses shrink, the speech itself is never touched (synthetic audio, no models)."""
import unittest
import numpy as np
from voicestudio import silence

SR = 22050


def tone(seconds, f=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype("float32")      # a voiced-like burst


def quiet(seconds): return np.random.default_rng(1).normal(0, 2e-4, int(seconds * SR)).astype("float32")


def speech_s(y):
    s, _ = silence._silent(y, SR); return (~s).sum() * silence.HOP_S


class Trim(unittest.TestCase):
    def test_edges_are_trimmed_and_speech_kept(self):
        y = np.concatenate([quiet(1.0), tone(2.0), quiet(1.0)])
        out, sec = silence.trim_clip(y, SR, lead=0.15, tail=0.25)
        self.assertAlmostEqual(len(out) / SR, 2.0 + 0.15 + 0.25, delta=0.06); self.assertAlmostEqual(speech_s(out), speech_s(y), delta=0.03)

    def test_short_edges_are_left_alone(self):
        y = np.concatenate([quiet(0.1), tone(2.0), quiet(0.2)])
        out, sec = silence.trim_clip(y, SR); self.assertLess(sec, 0.02)

    def test_long_inner_pause_is_shortened(self):
        y = np.concatenate([quiet(0.2), tone(1.5), quiet(1.6), tone(1.5), quiet(0.3)])
        out, _ = silence.trim_clip(y, SR, max_pause=0.6)
        self.assertLess(len(out) / SR, len(y) / SR - 0.9); self.assertAlmostEqual(speech_s(out), speech_s(y), delta=0.03)

    def test_short_inner_pause_is_kept(self):
        y = np.concatenate([quiet(0.2), tone(1.5), quiet(0.4), tone(1.5), quiet(0.3)])
        a, _ = silence.trim_clip(y, SR); b, _ = silence.trim_clip(y, SR, max_pause=0.6); self.assertEqual(len(a), len(b))

    def test_a_clip_that_would_become_too_short_is_kept(self):
        y = np.concatenate([quiet(0.5), tone(0.5), quiet(0.5)])
        out, sec = silence.trim_clip(y, SR); self.assertEqual(sec, 0.0)


if __name__ == "__main__": unittest.main()
