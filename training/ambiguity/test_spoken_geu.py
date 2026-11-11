import unittest

from context_rules import encode
from spoken_geu import decode, greedy_guards


class DecodeTest(unittest.TestCase):
    def test_round_trip(self):
        rules = [{"surface": "그", "source": [["그", "MM"]], "index": 0, "pos": "IC",
                  "guards": [[13, ""], [4, "뭐"], [15, "이제"]]},
                 {"surface": "제가", "source": [["저", "NP"], ["가", "JKS"]], "index": 1, "pos": "JKC",
                  "guards": [[14, "아니"]]}]
        self.assertEqual(decode(encode(rules)), rules)


class GreedyTest(unittest.TestCase):
    def test_uses_marginal_precision_after_earlier_guards(self):
        a, b, c = (4, "뭐"), (4, "사"), (13, "")
        occ = [(True, "d1", "SX", frozenset({a})), (True, "d2", "SX", frozenset({a, c})),
               (True, "d3", "SX", frozenset({c})), (False, "d4", "SX", frozenset({b})),
               (False, "d5", "SX", frozenset({b, c}))]
        guards, support, wrong = greedy_guards(occ, precision=0.6, good_min=1, docs_min=1, cap=8)
        # c alone is 2/3 precise, but after a it would add one right and one wrong occurrence.
        self.assertEqual(guards, [a])
        self.assertEqual((support, wrong), (2, 0))


if __name__ == "__main__":
    unittest.main()
