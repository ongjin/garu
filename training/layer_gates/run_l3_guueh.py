"""L3-구어 (blog 데이터로 proxy).

블로그 도메인의 단일 NNG/MAG 후보, 4/4 합의, freq>=10.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from filter import LayerConfig
from runner import run_layer, BASELINE_F1


CONFIG = LayerConfig(
    name="L3-Guueh",
    required_pos={"NNG", "MAG"},
    min_votes=4,
    min_freq=10,
    allowed_domains={"blog"},
    min_surface_len=2,
    surface_stoplist={
        "와", "헐", "ㅎ", "ㅋ", "ㅠ", "ㅜ",
        "정말", "진짜", "완전", "너무", "엄청",
    },
)


if __name__ == "__main__":
    result = run_layer(CONFIG, BASELINE_F1, tolerance_per_domain=0.10)
    print()
    print("=== L3-Guueh result ===")
    print(f"candidates: {result['candidates']}")
    print(f"added: {result['added']}")
    print(f"decision: {result['decision']}")
    if result['f1']:
        for k, v in result['f1'].items():
            bk = result['baseline'].get(k, 0)
            print(f"  {k}: {v:.4f} (baseline {bk:.4f}, delta={v-bk:+.4f})")
