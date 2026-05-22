"""Plan C: layer-by-layer F1 gate framework.

서브모듈:
- filter: candidates_pool.jsonl → layer 기준 통과 후보 추출
- merge: 후보 리스트를 content_dict.txt에 머지 (backup + diff)
- runner: 한 layer 전체 파이프라인 (filter → merge → build → F1 측정 → decide)

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 3.
"""
