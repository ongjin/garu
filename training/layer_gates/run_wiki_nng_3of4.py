"""Wiki-NNG 3/4 완화: 위키 도메인 NNG, 3/4 합의, freq≥10.

Wiki-NNG-strict 통과 후에만 실행 (단계적 완화).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from filter import LayerConfig
from runner import run_layer, BASELINE_F1


CONFIG = LayerConfig(
    name="Wiki-NNG-3of4",
    required_pos={"NNG"},
    min_votes=3,
    min_freq=10,
    allowed_domains={"wiki"},
    min_surface_len=2,
    surface_stoplist=set(),
)


if __name__ == "__main__":
    result = run_layer(CONFIG, BASELINE_F1, tolerance_per_domain=0.10)
    print()
    print("=== Wiki-NNG-3of4 result ===")
    print(f"candidates: {result['candidates']}")
    print(f"added: {result['added']}")
    print(f"decision: {result['decision']}")
    if result['f1']:
        for k, v in result['f1'].items():
            bk = result['baseline'].get(k, 0)
            print(f"  {k}: {v:.4f} (baseline {bk:.4f}, delta={v-bk:+.4f})")
