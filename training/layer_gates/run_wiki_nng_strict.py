"""Wiki-NNG strict: 위키 도메인 NNG, 4/4 합의, freq≥10.

스펙 L6 (어절 캐시)와 별도 — Wiki sample 50K 추가로 가능해진 신규 layer.
NNG에 한정 (Wiki NNP 재현 위험 회피, memory: project_kiwi_actually_wins).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from filter import LayerConfig
from runner import run_layer, BASELINE_F1


CONFIG = LayerConfig(
    name="Wiki-NNG-strict",
    required_pos={"NNG"},
    min_votes=4,
    min_freq=10,
    allowed_domains={"wiki"},
    min_surface_len=2,
    surface_stoplist=set(),
)


if __name__ == "__main__":
    result = run_layer(CONFIG, BASELINE_F1, tolerance_per_domain=0.10)
    print()
    print("=== Wiki-NNG-strict result ===")
    print(f"candidates: {result['candidates']}")
    print(f"added: {result['added']}")
    print(f"decision: {result['decision']}")
    if result['f1']:
        for k, v in result['f1'].items():
            bk = result['baseline'].get(k, 0)
            print(f"  {k}: {v:.4f} (baseline {bk:.4f}, delta={v-bk:+.4f})")
