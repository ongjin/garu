"""Garu 사전 확장용 5분석기 ensemble 모듈.

Plan A에서 wrappers, pos_normalize, surface_normalize, agreement 4개 sub-module 추가.
Plan B 이후 layer_gates/, dict_diff/ 모듈이 본 모듈을 의존.

Garu는 dedup 용도이며 합의 분모(4명)에는 포함되지 않는다.
스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 2.
"""
