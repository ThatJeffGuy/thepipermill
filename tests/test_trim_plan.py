"""The cut planner: which stretches of a clip survive after the user deletes words from its text (no audio, no GPU)."""
import unittest
from voicestudio import trim, verify

ORIG = "Hello there my friend. What do you want? I want the money. Never mind that. Goodbye now."


def words(text, step=0.4, gap=0.1):
    out, t = [], 0.2
    for w in text.split(): out.append((w, round(t, 2), round(t + step, 2))); t += step + gap
    return out, round(t + 0.2, 2)


W, D = words(ORIG)


def plan(new): return trim.plan(ORIG, new, W, D)


class Plan(unittest.TestCase):
    def test_middle_sentence_removed(self):
        st, runs, n = plan("Hello there my friend. I want the money. Never mind that. Goodbye now.")
        self.assertEqual(st, "ok"); self.assertEqual(n, 4); self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0][2], "Hello there my friend."); self.assertEqual(runs[1][2], "I want the money. Never mind that. Goodbye now.")
        self.assertLess(runs[0][1], runs[1][0])                      # the removed words' audio is outside both pieces

    def test_several_stretches_start_middle_end(self):
        st, runs, _ = plan("my friend. I want the money. Goodbye")
        self.assertEqual([r[2] for r in runs], ["my friend.", "I want the money.", "Goodbye"])

    def test_back_and_forth_keeps_every_other_line(self):
        st, runs, _ = plan("Hello there my friend. I want the money. Goodbye now.")
        self.assertEqual(len(runs), 3)

    def test_single_deleted_word_splits(self):
        st, runs, n = plan("Hello there friend. What do you want? I want the money. Never mind that. Goodbye now.")
        self.assertEqual((st, n, len(runs)), ("ok", 1, 2))

    def test_spelling_fix_is_not_a_cut(self):
        self.assertEqual(plan("Hello there my friend. What do you want? I want the money. Never mind that. Good bye now.")[0], "nocut")

    def test_added_words_are_not_a_cut(self):
        self.assertEqual(plan(ORIG + " sir")[0], "nocut")

    def test_everything_deleted(self):
        self.assertEqual(plan("")[0], "removed")

    def test_correction_next_to_a_deletion(self):
        st, runs, n = plan("Hello there, my fiend. I want the monkey. Never mind that. Goodbye now.")
        self.assertEqual((st, n, len(runs)), ("ok", 4, 2))
        self.assertIn("fiend", runs[0][2]); self.assertIn("monkey", runs[1][2])      # the corrected spelling is kept

    def test_text_that_does_not_fit_the_audio(self):
        self.assertEqual(trim.plan("completely different words here", "different", W, D)[0], "nomatch")


class Wer(unittest.TestCase):
    def test_same_words(self): self.assertEqual(verify.wer("It is fine.", "it is fine"), 0)
    def test_contractions_count_as_the_same(self): self.assertEqual(verify.wer("we're going to go", "we're gonna go"), 0)
    def test_one_in_four(self): self.assertAlmostEqual(verify.wer("a b c d", "a b x d"), 0.25)


if __name__ == "__main__": unittest.main()
