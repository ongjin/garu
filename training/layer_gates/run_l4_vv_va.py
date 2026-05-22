"""L4-VV/VA: 동사/형용사, 4/4 strict, freq>=10. 가장 위험한 layer."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from filter import LayerConfig
from runner import run_layer, BASELINE_F1


CONFIG = LayerConfig(
    name="L4-VV-VA",
    required_pos={"VV", "VA"},
    min_votes=4,
    min_freq=10,
    min_surface_len=1,
    surface_stoplist=set(),
)


if __name__ == "__main__":
    result = run_layer(CONFIG, BASELINE_F1, tolerance_per_domain=0.10)
    print()
    print("=== L4-VV/VA result ===")
    print(f"candidates: {result['candidates']}")
    print(f"added: {result['added']}")
    print(f"decision: {result['decision']}")
    if result['f1']:
        for k, v in result['f1'].items():
            bk = result['baseline'].get(k, 0)
            print(f"  {k}: {v:.4f} (baseline {bk:.4f}, delta={v-bk:+.4f})")
