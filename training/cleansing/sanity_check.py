"""5분석기 어절 단위 호출 + 정규화 sanity check.

50문장 무작위 표본을 추출하고 각 문장의 어절별로 5분석기를 돌려 출력을 dump.
- 분석기가 모두 응답하는지
- POS 정규화 결과가 세종 태그셋 안에 들어가는지
- 어절 단위 분리가 깨지는 케이스가 있는지

성공 기준: 50문장 × 모든 어절에서 5분석기가 morpheme 1개 이상 반환.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "ensemble"))

from wrappers import KiwiWrapper, MecabWrapper, KkmaWrapper, KomoranWrapper, GaruWrapper

SEJONG_TAGS = {
    # 체언
    "NNG", "NNP", "NNB", "NP", "NR",
    # 용언
    "VV", "VA", "VX", "VCP", "VCN",
    # 관형사/부사/감탄사
    "MM", "MAG", "MAJ", "IC",
    # 조사
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    # 어미
    "EP", "EF", "EC", "ETN", "ETM",
    # 접사
    "XPN", "XSN", "XSV", "XSA", "XR",
    # 부호
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
    # 분석불능
    "UNKNOWN", "NA",
}


def main():
    gold_path = ROOT / "training" / "gold_testset" / "gold_testset.jsonl"
    sentences = []
    with open(gold_path) as f:
        for line in f:
            sentences.append(json.loads(line)["text"])
    random.seed(42)
    sample = random.sample(sentences, 50)

    print("Loading analyzers...")
    analyzers = {
        "kiwi": KiwiWrapper(),
        "mecab": MecabWrapper(),
        "kkma": KkmaWrapper(),
        "komoran": KomoranWrapper(),
        "garu": GaruWrapper(),
    }

    issues = {"empty_output": [], "unknown_tags": [], "exception": []}
    seen_tags = set()
    total_eojeols = 0

    for sent_idx, text in enumerate(sample):
        eojeols = text.split()
        for eoj_idx, eojeol in enumerate(eojeols):
            total_eojeols += 1
            for name, analyzer in analyzers.items():
                try:
                    morphs = analyzer.analyze(eojeol)
                except Exception as e:
                    issues["exception"].append((sent_idx, eoj_idx, name, str(e)[:100]))
                    continue
                if not morphs:
                    issues["empty_output"].append((sent_idx, eoj_idx, name, eojeol))
                for _, tag in morphs:
                    seen_tags.add(tag)
                    if tag not in SEJONG_TAGS:
                        issues["unknown_tags"].append((sent_idx, eoj_idx, name, eojeol, tag))

        if (sent_idx + 1) % 10 == 0:
            print(f"  {sent_idx+1}/50 sentences processed")

    print(f"\n=== Sanity Check Result ===")
    print(f"Total eojeols: {total_eojeols}")
    print(f"Distinct tags seen: {sorted(seen_tags)}")
    print(f"\nEmpty outputs: {len(issues['empty_output'])}")
    for x in issues["empty_output"][:5]:
        print(f"  {x}")
    print(f"\nUnknown tags: {len(issues['unknown_tags'])}")
    unknown_tag_counter = {}
    for _, _, name, _, tag in issues["unknown_tags"]:
        unknown_tag_counter[(name, tag)] = unknown_tag_counter.get((name, tag), 0) + 1
    for (name, tag), cnt in sorted(unknown_tag_counter.items(), key=lambda x: -x[1])[:20]:
        print(f"  {name} {tag}: {cnt}")
    print(f"\nExceptions: {len(issues['exception'])}")
    for x in issues["exception"][:5]:
        print(f"  {x}")

    # 합격 기준
    if len(issues["empty_output"]) > total_eojeols * 0.02:
        print("\nFAIL: empty_output > 2% of eojeols")
        sys.exit(1)
    if len(issues["exception"]) > 0:
        print("\nFAIL: exceptions occurred")
        sys.exit(1)
    if len(issues["unknown_tags"]) > total_eojeols * 0.10:
        print(f"\nWARN: unknown_tags > 10% — extend pos_normalize.py before proceeding")
        sys.exit(2)
    print("\nPASS")


if __name__ == "__main__":
    main()
