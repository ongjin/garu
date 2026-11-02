"""python3 -m unittest discover -s training/gate -p 'test_*.py'"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate  # noqa: E402
from score import counts  # noqa: E402
from sets import NIKL_2025, SX_FILE, load  # noqa: E402

SMALL = {**gate.GATE, "replicates": 2_000}


def rows(name, n_docs, per_doc, stratum_of, tp_of):
    out = []
    for d in range(n_docs):
        for k in range(per_doc):
            out.append({"id": f"{name}:{d}:{k}", "doc": f"{name}:{d}",
                        "stratum": stratum_of(d), "tp": tp_of(d, k), "pred": 10, "gold": 10})
    return out


def dataset(tp_of=lambda name, s, d, k: 9):
    out = {}
    for name, strata in (("gold", ["뉴스", "문학"]), ("sx25", ["SX"]), ("nx25", ["경제", "IT/과학"])):
        per_doc = 1 if name == "gold" else 5
        out[name] = rows(name, 200, per_doc, lambda d, st=strata: st[d % len(st)],
                         lambda d, k, nm=name, st=strata: tp_of(nm, st[d % len(st)], d, k))
    return out


class GateTest(unittest.TestCase):
    def test_identical_is_zero_and_fails(self):
        base = dataset()
        report = gate.judge(base, dataset(), SMALL)
        self.assertFalse(report["pass"])
        self.assertFalse(report["regressions"])
        for s in report["subsets"].values():
            self.assertEqual((s["delta_pp"], s["lo_pp"], s["hi_pp"]), (0.0, 0.0, 0.0))

    def test_consistent_gain_passes(self):
        rnd = random.Random(1)
        gains = {(n, d, k): rnd.random() < 0.3 for n in gate.SETS for d in range(200) for k in range(5)}
        cand = dataset(lambda n, s, d, k: 9 + gains[(n, d, k)])
        report = gate.judge(dataset(), cand, SMALL)
        self.assertTrue(report["benefit"])
        self.assertTrue(report["pass"])

    def test_stratum_regression_blocks(self):
        rnd = random.Random(2)
        flips = {(n, d, k): rnd.random() < 0.3 for n in gate.SETS for d in range(200) for k in range(5)}

        def tp(n, s, d, k):
            if s == "문학":
                return 9 - flips[(n, d, k)]
            return 9 + flips[(n, d, k)]

        report = gate.judge(dataset(), dataset(tp), SMALL)
        self.assertTrue(report["benefit"])
        self.assertEqual(report["regressions"], ["gold/문학"])
        self.assertFalse(report["pass"])

    def test_single_unit_change_is_not_significant(self):
        def tp(n, s, d, k):
            return 8 if (n, d, k) == ("gold", 1, 0) else 9

        report = gate.judge(dataset(), dataset(tp), SMALL)
        self.assertFalse(report["regressions"])

    def test_counts_use_ep_norm_multiset(self):
        pred = [["나", "NP"], ["는", "JX"], ["가", "VV"], ["았", "EP"], ["다", "EF"]]
        gold = [["나", "NP"], ["는", "JX"], ["가", "VV"], ["았", "EP"], ["다", "EF"], [".", "SF"]]
        self.assertEqual(counts(pred, gold), (5, 5, 6))


@unittest.skipUnless((NIKL_2025 / SX_FILE).exists(), "NIKL 2025 SX 없음")
class SetsTest(unittest.TestCase):
    def test_sx25_matches_heldout_loader(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import eval_nikl2025_guueh as legacy
        from reverse_2025_to_2021 import reverse_convert

        old = legacy.load_sx()
        new = load("sx25")
        self.assertEqual(len(old), len(new))
        for (text, g25), rec in zip(old, new):
            self.assertEqual(text, rec["text"])
            self.assertEqual(reverse_convert(g25), rec["gold"])


if __name__ == "__main__":
    unittest.main()
