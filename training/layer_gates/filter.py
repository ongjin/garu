"""candidates_pool → layer 기준 통과 후보 추출.

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 3.
"""
from dataclasses import dataclass, field


@dataclass
class LayerConfig:
    name: str
    required_pos: set[str]
    min_votes: int
    min_freq: int
    allowed_domains: set[str] | None = None
    min_surface_len: int = 1
    max_surface_len: int = 999
    surface_stoplist: set[str] = field(default_factory=set)


def _votes(rec: dict) -> int:
    return sum(rec["votes"].values())


def filter_candidates(records: list[dict], cfg: LayerConfig) -> list[dict]:
    """LayerConfig 기준 통과 후보만 반환."""
    out = []
    for r in records:
        if r["normalized_pos"] not in cfg.required_pos:
            continue
        if r["in_garu_dict"]:
            continue
        if _votes(r) < cfg.min_votes:
            continue
        if r["frequency"] < cfg.min_freq:
            continue
        if cfg.allowed_domains is not None:
            if not (set(r["source_domains"]) & cfg.allowed_domains):
                continue
        slen = len(r["surface"])
        if slen < cfg.min_surface_len or slen > cfg.max_surface_len:
            continue
        if r["surface"] in cfg.surface_stoplist:
            continue
        out.append(r)
    return out
