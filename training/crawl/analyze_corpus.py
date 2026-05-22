"""크롤된 코퍼스를 5분석기로 일괄 분석 → candidates_pool.jsonl 생성.

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 2.6.

candidates_pool record 형식:
    {"surface": str, "normalized_pos": str,
     "votes": {"kiwi": 0|1, "mecab": 0|1, "kkma": 0|1, "komoran": 0|1},
     "in_garu_dict": bool, "frequency": int, "source_domains": [str]}

frequency = 코퍼스 내 (surface, pos)가 단일 형태소로 인식된 문장 수.
"""
import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ensemble"))
from wrappers import (
    KiwiWrapper, MecabWrapper, KkmaWrapper, KomoranWrapper, GaruWrapper,
)
from agreement import compute_votes


def _iter_sentences(input_paths: list[str]):
    for p in input_paths:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def _process_chunk(args):
    """Worker: 하나의 청크를 처리해서 local pool(직렬화 가능 dict) 반환."""
    chunk, domain = args
    kiwi = KiwiWrapper(); mecab = MecabWrapper()
    kkma = KkmaWrapper(); komoran = KomoranWrapper()
    garu = GaruWrapper()

    pool: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "votes_sum": {"kiwi": 0, "mecab": 0, "kkma": 0, "komoran": 0},
        "in_garu_dict": False,
        "frequency": 0,
    })

    for text in chunk:
        analyses = {
            "kiwi":    kiwi.analyze(text),
            "mecab":   mecab.analyze(text),
            "kkma":    kkma.analyze(text),
            "komoran": komoran.analyze(text),
        }
        try:
            garu_tokens = garu.analyze(text)
        except Exception:
            garu_tokens = []
        votes = compute_votes(analyses, garu_analysis=garu_tokens)
        for key, info in votes.items():
            entry = pool[key]
            entry["frequency"] += 1
            for a, v in info["votes"].items():
                entry["votes_sum"][a] += v
            if info["in_garu_dict"]:
                entry["in_garu_dict"] = True

    return {f"{k[0]}\t{k[1]}": v for k, v in pool.items()}


def _chunks(items, n_chunks):
    chunk_size = max(1, (len(items) + n_chunks - 1) // n_chunks)
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def build_candidates_pool(
    input_paths: list[str], output_path: str, domain: str,
    n_workers: int = 4,
) -> int:
    """입력 코퍼스 파일들을 분석 → 누적 candidates_pool 작성. 반환: 최종 풀 행 수."""

    all_texts = [r["text"] for r in _iter_sentences(input_paths)]
    print(f"Loaded {len(all_texts)} sentences, dispatching to {n_workers} workers")

    chunks = list(_chunks(all_texts, n_workers))
    pool: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "votes_sum": {"kiwi": 0, "mecab": 0, "kkma": 0, "komoran": 0},
        "in_garu_dict": False,
        "frequency": 0,
        "source_domains": set(),
    })

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, local in enumerate(ex.map(_process_chunk, [(c, domain) for c in chunks])):
            print(f"  worker {i+1}/{len(chunks)} returned ({len(local)} keys)")
            for key_str, lentry in local.items():
                surface, pos = key_str.split("\t", 1)
                entry = pool[(surface, pos)]
                entry["frequency"] += lentry["frequency"]
                for a, v in lentry["votes_sum"].items():
                    entry["votes_sum"][a] += v
                if lentry["in_garu_dict"]:
                    entry["in_garu_dict"] = True
                entry["source_domains"].add(domain)

    out_records = []
    for (surface, pos), entry in pool.items():
        votes_bool = {a: (1 if entry["votes_sum"][a] > 0 else 0)
                      for a in ("kiwi", "mecab", "kkma", "komoran")}
        out_records.append({
            "surface": surface,
            "normalized_pos": pos,
            "votes": votes_bool,
            "in_garu_dict": entry["in_garu_dict"],
            "frequency": entry["frequency"],
            "source_domains": sorted(entry["source_domains"]),
        })

    if os.path.exists(output_path):
        existing: dict[tuple[str, str], dict] = {}
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing[(r["surface"], r["normalized_pos"])] = r
        for rec in out_records:
            key = (rec["surface"], rec["normalized_pos"])
            if key in existing:
                old = existing[key]
                rec["frequency"] += old["frequency"]
                rec["source_domains"] = sorted(set(old["source_domains"]) | set(rec["source_domains"]))
                rec["in_garu_dict"] = rec["in_garu_dict"] or old["in_garu_dict"]
                for a in rec["votes"]:
                    rec["votes"][a] = max(rec["votes"][a], old["votes"].get(a, 0))
            existing[key] = rec
        out_records = list(existing.values())

    with open(output_path, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(out_records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--domain", required=True,
                    choices=["news", "specialist", "blog", "test", "wiki"])
    ap.add_argument("--output", default="training/codebook_data/candidates_pool.jsonl")
    args = ap.parse_args()
    n = build_candidates_pool(args.input, args.output, args.domain)
    print(f"candidates_pool now has {n} unique (surface, pos) entries")


if __name__ == "__main__":
    main()
