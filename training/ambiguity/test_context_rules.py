"""python3 -m unittest discover -s training/ambiguity -p 'test_*.py'"""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_rules import (EXACT_NEXT, EXACT_PREV, RuleSet, contexts, encode,  # noqa: E402
                           guard_matches, one_pos_fix)

SKY = {"surface": "하늘을", "source": [["하늘", "NNG"], ["을", "JKO"]], "index": 0,
       "pos": "NNP", "guards": [[EXACT_PREV, "나는"]], "support": 9}


class ContextRulesTest(unittest.TestCase):
    def test_guards_match_rust_semantics(self):
        cases = [((0, "", None, "새"), True), ((0, "", "나는", None), False),
                 ((13, "", "나는", None), True), ((2, "나는", "그나는", None), True),
                 ((3, "나는", "나는", None), False), ((3, "는", "는", None), False),
                 ((4, "번", None, "번째"), True), ((8, "째다", None, "번째다"), True),
                 ((10, "그", "그나는", None), True), ((EXACT_PREV, "나는", "그나는", None), False),
                 ((EXACT_NEXT, "새", None, "새"), True), ((EXACT_NEXT, "새", None, None), False)]
        for args, want in cases:
            self.assertEqual(guard_matches(*args), want, args)

    def test_mined_contexts_satisfy_their_guards(self):
        for prev, nxt in [(None, "번째"), ("그나는", None), ("는", "새")]:
            for kind, value, _ in contexts(prev, nxt):
                self.assertTrue(guard_matches(kind, value, prev, nxt), (kind, value))

    def test_one_pos_fix_is_unique_and_skips_symbols(self):
        source = (("하늘", "NNG"), ("을", "JKO"))
        self.assertEqual(one_pos_fix(source, [["하늘", "NNP"], ["을", "JKO"]]), (0, "NNP"))
        self.assertIsNone(one_pos_fix(source, [["하늘", "NNG"], ["을", "JKO"]]))
        self.assertIsNone(one_pos_fix((("!", "SF"),), [["!", "SW"]]))

    def test_apply_needs_exact_source_span_and_guard(self):
        rules = RuleSet([SKY])
        tokens = [["나", "NP", 0, 2], ["는", "JX", 0, 2], ["하늘", "NNG", 3, 6], ["을", "JKO", 3, 6]]
        out, edits = rules.apply("나는 하늘을", tokens)
        self.assertEqual(out[2][1], "NNP")
        self.assertEqual(edits, [(2, "하늘", "NNG", "NNP")])
        self.assertEqual(rules.apply("너는 하늘을", tokens)[1], [])
        self.assertEqual(rules.apply("나는 하늘을", tokens, {"하늘"})[1], [])
        crossing = [t[:] for t in tokens]
        crossing[3][3] = 7
        self.assertEqual(rules.apply("나는 하늘을", crossing)[1], [])

    def test_encode_layout(self):
        blob = encode([SKY])
        self.assertEqual(blob[:5], b"GCPR\x01")
        self.assertEqual(struct.unpack_from("<I", blob, 5)[0], 1)


if __name__ == "__main__":
    unittest.main()
