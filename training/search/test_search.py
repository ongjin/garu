"""python3 -m unittest discover -s training/search -p 'test_*.py'"""
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export  # noqa: E402
import router  # noqa: E402


class RouterTest(unittest.TestCase):
    def test_score_matches_rust_fixture(self):
        # crates/garu-core/src/search.rs router_matches_python_hashing와 같은 값
        weights = np.array([i - 8 for i in range(16)], dtype=np.int8)
        self.assertEqual(router.features("가나", 4), [0, 1, 2, 6, 7, 8, 9, 12, 13])
        self.assertEqual(router.score("가나", weights, 0.5, -1.0, 4), -8.0)
        self.assertEqual(router.score("  가나 ", weights, 0.5, -1.0, 4), -8.0)

    def test_strip_follows_rust_whitespace(self):
        self.assertEqual(router.features("\x1c가", 4), router.features("\x1c가", 4))
        self.assertNotEqual(router.features("\x1c가", 4), router.features("가", 4))
        self.assertEqual(router.features("　가 ", 4), router.features("가", 4))


class ExportTest(unittest.TestCase):
    def test_encode_layout_and_quantization(self):
        blob = export.encode([(70_000, -0.5), (3, 1.0), (5, 1e-9)], 0.5)
        magic, dim, scale, count, qscale = struct.unpack_from("<5sBfIf", blob)
        self.assertEqual((magic, dim, scale, count), (b"GSCW\x01", 20, 0.5, 2))
        self.assertAlmostEqual(qscale, 1.0 / 32767, places=9)
        self.assertEqual(blob[18:], bytes([3, 0xED, 0xA2, 0x04])  # 3, 69_997 LEB128
                         + struct.pack("<hh", 32767, round(-0.5 * 32767)))


if __name__ == "__main__":
    unittest.main()
