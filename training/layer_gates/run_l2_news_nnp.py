"""L2-News-NNP: 뉴스 도메인 단일 NNP, 4/4 strict, freq>=3.

NNP는 Wiki NNP 재현 위험 — strict 게이트 + stoplist로 차단.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from filter import LayerConfig
from runner import run_layer, BASELINE_F1


CONFIG = LayerConfig(
    name="L2-News-NNP",
    required_pos={"NNP"},
    min_votes=4,
    min_freq=3,
    allowed_domains={"news"},
    min_surface_len=2,
    max_surface_len=15,
    surface_stoplist={
        "오늘", "어제", "내일", "올해", "작년", "기자", "씨", "님", "분",
        "그", "이", "저", "그것", "이것", "저것",
        "1세", "2세", "3세", "전투",
    },
)


if __name__ == "__main__":
    result = run_layer(CONFIG, BASELINE_F1, tolerance_per_domain=0.10)
    print()
    print("=== L2-News-NNP result ===")
    print(f"candidates: {result['candidates']}")
    print(f"added: {result['added']}")
    print(f"decision: {result['decision']}")
    if result['f1']:
        for k, v in result['f1'].items():
            bk = result['baseline'].get(k, 0)
            print(f"  {k}: {v:.4f} (baseline {bk:.4f}, delta={v-bk:+.4f})")
