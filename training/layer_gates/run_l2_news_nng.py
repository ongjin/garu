"""L2-News-NNG: 뉴스 도메인 단일 NNG 후보, 3/4 합의, freq>=10."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from filter import LayerConfig
from runner import run_layer, BASELINE_F1


CONFIG = LayerConfig(
    name="L2-News-NNG",
    required_pos={"NNG"},
    min_votes=3,
    min_freq=10,
    allowed_domains={"news"},
    min_surface_len=2,
    surface_stoplist={
        "기자", "오늘", "어제", "내일", "올해", "작년", "내년",
        "그것", "이것", "저것", "이거", "저거", "그거", "여기", "저기", "거기",
    },
)


if __name__ == "__main__":
    result = run_layer(CONFIG, BASELINE_F1, tolerance_per_domain=0.10)
    print()
    print("=== L2-News-NNG result ===")
    print(f"candidates: {result['candidates']}")
    print(f"added: {result['added']}")
    print(f"decision: {result['decision']}")
    if result['f1']:
        for k, v in result['f1'].items():
            bk = result['baseline'].get(k, 0)
            print(f"  {k}: {v:.4f} (baseline {bk:.4f}, delta={v-bk:+.4f})")
