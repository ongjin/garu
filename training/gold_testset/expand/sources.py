"""도메인별 raw 문장 추출 entry point."""
import argparse
from pathlib import Path

from sources_nikl import load_nikl_raw
from sources_wiki import load_kowiki_raw
from sources_web import load_web_raw

WEB_PATHS = {
    "SNS": Path(__file__).parent / "_candidates" / "SNS_web_raw.txt",
    "의료": Path(__file__).parent / "_candidates" / "의료_web_raw.txt",
}

DOMAIN_MAP = {
    "뉴스": [("nikl_nxmp", lambda n, s: load_nikl_raw("NXMP", n, s))],
    "구어": [("nikl_sxmp", lambda n, s: load_nikl_raw("SXMP", n, s))],
    "일상": [("kowiki", lambda n, s: load_kowiki_raw(n, s))],
    "문학": [("kowiki", lambda n, s: load_kowiki_raw(n, s))],
    "기술": [("kowiki", lambda n, s: load_kowiki_raw(n, s))],
    "SNS": [("web", lambda n, s: load_web_raw(WEB_PATHS["SNS"], n, s))],
    "의료": [("web", lambda n, s: load_web_raw(WEB_PATHS["의료"], n, s))],
}


def extract_for_domain(domain: str, n: int, seed: int = 42) -> list[str]:
    if domain not in DOMAIN_MAP:
        raise ValueError(f"Unknown domain: {domain}")
    sources = DOMAIN_MAP[domain]
    per_source = n // len(sources)
    remainder = n % len(sources)
    out = []
    for i, (_, loader) in enumerate(sources):
        k = per_source + (1 if i < remainder else 0)
        out.extend(loader(k, seed + i))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sents = extract_for_domain(args.domain, args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for s in sents:
            f.write(s + "\n")
    print(f"Wrote {len(sents)} to {args.out}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        out = extract_for_domain("뉴스", n=10, seed=42)
        assert len(out) == 10
        print(f"뉴스[0]: {out[0]}")
        out2 = extract_for_domain("구어", n=10, seed=42)
        assert len(out2) == 10
        print(f"구어[0]: {out2[0]}")
        print("OK")
