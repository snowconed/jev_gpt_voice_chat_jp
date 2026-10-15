"""Offline checks for complete short stop/denial rules, not semantic inference."""
import unittest

from interruption_intent import interruption_kind


class InterruptionIntentTests(unittest.TestCase):
    def test_explicit_short_pause_requests(self):
        for text in ("待って", "ちょっと待って", "少し待ってください", "待って下さい。",
                     "ちょっと待ってください！", "少々待ってください。", "ストップ",
                     "止めて", "やめて", "止めてください", "やめてください"):
            with self.subTest(text=text):
                self.assertEqual(interruption_kind(text), "pause")

    def test_bare_denials_wait_for_correction(self):
        for text in ("違う", "違います", "いや", "いやいや", "いいえ", "いえ",
                     "そうじゃない", "そうではない", "そうじゃないです", "そうではありません",
                     "違うよ", "違うんです", "違いますよ"):
            with self.subTest(text=text):
                self.assertEqual(interruption_kind(text), "correction")

    def test_short_combinations_require_a_full_match(self):
        self.assertEqual(interruption_kind("いや、違う、待って"), "pause")
        self.assertEqual(interruption_kind("待って、いや、違います"), "pause")
        self.assertEqual(interruption_kind("いや、違う"), "correction")
        self.assertEqual(interruption_kind("いいえ、違います。"), "correction")
        self.assertEqual(interruption_kind("待って待って"), "pause")
        self.assertEqual(interruption_kind("違う違う"), "correction")
        self.assertIsNone(interruption_kind("いや、違う、待って、明日です。"))

    def test_substantive_corrections_are_never_held_as_short_denials(self):
        for text in ("待って、明日の話です", "いや、赤がいいです", "待っている人がいます",
                     "違う映画を見ました", "いいえ、違います。正しくは木曜日です。",
                     "待って、それは違います", "いや、違うんですけど、赤です", "そうじゃない映画です",
                     "止めてもいいですか", "やめてほしくないです", "違いません", "待ってます"):
            with self.subTest(text=text):
                self.assertIsNone(interruption_kind(text))

    def test_minimal_english_rules_do_not_match_longer_words(self):
        self.assertEqual(interruption_kind(" STOP! "), "pause")
        self.assertEqual(interruption_kind("Wait, wait."), "pause")
        self.assertEqual(interruption_kind("ＮＯ！"), "correction")
        self.assertEqual(interruption_kind("No, wait!"), "pause")
        for text in ("nobody", "not now", "waiting", "stopper", "No, red please.",
                     "Wait, tomorrow.", "Do not stop", "Please wait"):
            with self.subTest(text=text):
                self.assertIsNone(interruption_kind(text))

    def test_whitespace_and_punctuation_normalization_preserves_input(self):
        text = '  「いや」\n 違います！\t少し待ってください。 '
        original = text
        self.assertEqual(interruption_kind(text), "pause")
        self.assertEqual(text, original)

    def test_acknowledgments_uncertain_sounds_and_incomplete_prefixes_are_not_controls(self):
        for text in ("はい", "うん", "ええ", "なるほど", "そうですね", "うーん", "ううん",
                     "待", "待っ", "ちょっと", "少し", "違", "違い", "そうじゃ",
                     "はい、待って", "", " \t\n", "。！？", None, 1):
            with self.subTest(text=text):
                self.assertIsNone(interruption_kind(text))

    def test_many_repeated_denials_do_not_require_recursive_matching(self):
        self.assertEqual(interruption_kind("いや、" * 600 + "違う。"), "correction")
        self.assertIsNone(interruption_kind("いや、" * 600 + "明日です。"))


if __name__ == "__main__":
    unittest.main()
