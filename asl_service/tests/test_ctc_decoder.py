import unittest

from asl_service.ctc_decoder import greedy_ctc_decode, tokens_with_timestamps


class CtcDecoderTests(unittest.TestCase):
    def test_collapses_repeats_and_requires_blank_between_duplicate_words(self):
        labels = ["ME", "GO"]
        probabilities = [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.05, 0.05, 0.9],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.05, 0.05, 0.9],
            [0.1, 0.85, 0.05],
        ]
        decoded = greedy_ctc_decode(probabilities, labels)
        self.assertEqual([token.label for token in decoded], ["ME", "GO", "GO"])
        timed = tokens_with_timestamps(decoded, len(probabilities), 7000)
        self.assertEqual(timed[0]["startMs"], 0)
        self.assertEqual(timed[-1]["endMs"], 7000)


if __name__ == "__main__":
    unittest.main()
