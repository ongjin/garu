"""구어 담화표지 `그`의 Section 17 규칙 (연구 이력 #83).

NIKL 2021 SX는 담화표지 `그`를 대개 `그~`로 전사해 Garu는 그 어절만 IC로 낸다. 2025 전사에는
늘임표가 없어 맨 `그`가 늘 MM이 된다. 늘임표를 지운 2021 자료에서 (그, [그/MM])→IC 규칙의
이웃 어절 조건을 탐욕 선택하고, 기존 Section 17 규칙 뒤에 붙인다.

    python3 training/ambiguity/prepare.py --out LAB/orig --model js/models/base.gmdl
    python3 training/ambiguity/spoken_geu.py strip LAB/orig LAB/strip
    # LAB/strip/{train,dev}.txt를 analyze_batch --json-spans로 분석해 {split}.base.jsonl 생성
    python3 training/ambiguity/spoken_geu.py mine LAB/strip
    python3 training/ambiguity/spoken_geu.py append LAB/strip/rule_그.json js/models/base.gmdl OUT.gmdl [--blob B]

선택 규칙: 정밀도·최소 지지·문서 수·조건 상한 격자에서 dev NX F1이 기준보다 낮지 않은 설정 중
dev SX 이득이 가장 큰 것(동률이면 조건이 적은 것). 고정 골드·2025 자료는 쓰지 않는다.
"""
import itertools
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gate")]
from context_cache import POS, sections  # noqa: E402
from context_rules import MAGIC, contexts, encode, eojeols, normalized, read_rows, score  # noqa: E402

SURFACE, SOURCE, TARGET = "그", (("그", "MM"),), "IC"
TILDE = "~"
GRID = {"precision": (0.5, 0.55, 0.6, 0.7, 0.8), "good": (3, 8), "docs": (2, 3), "cap": (32, 128, 512)}


def strip(src, dst):
    """SX 문장의 늘임표를 지운 사본. 어절을 공백 하나로 다시 잇고 정답 형태소의 `~`도 지운다."""
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        counts = Counter()
        with open(src / f"{split}.jsonl", encoding="utf-8") as f, \
                open(dst / f"{split}.jsonl", "w", encoding="utf-8") as out, \
                open(dst / f"{split}.txt", "w", encoding="utf-8") as text_out:
            for line in f:
                row = json.loads(line)
                if row["src"] == "SX" and TILDE in row["text"]:
                    words = []
                    for form, _, _, morphs in row["words"]:
                        new_form = form.replace(TILDE, "")
                        new_morphs = [[m[0].replace(TILDE, ""), m[1]] for m in morphs if m[0].replace(TILDE, "")]
                        if new_form and new_morphs:
                            words.append([new_form, new_morphs])
                    if not words:
                        counts["dropped"] += 1
                        continue
                    position, spans = 0, []
                    for form, morphs in words:
                        spans.append([form, position, position + len(form), morphs])
                        position += len(form) + 1
                    row["text"] = " ".join(w[0] for w in words)
                    row["words"] = spans
                    row["morphemes"] = [m for w in words for m in w[1]]
                    counts["stripped"] += 1
                counts[row["src"]] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                text_out.write(row["text"] + "\n")
        print(split, dict(counts))


def occurrences(data):
    out = []
    for row, tokens in read_rows(data, "train"):
        for (surface, source, prev, nxt, *_), word in zip(eojeols(row["text"], tokens), row["words"]):
            if surface != SURFACE or source != SOURCE or not word[3]:
                continue
            good = normalized(word[3]) == normalized([(SURFACE, TARGET)])
            guards = frozenset((kind, value) for kind, value, _ in contexts(prev, nxt))
            out.append((good, row["doc"], row["src"], guards))
    return out


def greedy_guards(occ, precision, good_min, docs_min, cap):
    """학습 발생에서 새로 덮는 정답-오답이 가장 큰 조건부터 고른다."""
    stats = defaultdict(lambda: [0, 0, set()])
    index = defaultdict(list)
    for i, (good, doc, _, guards) in enumerate(occ):
        for guard in guards:
            stats[guard][0 if good else 1] += 1
            if good:
                stats[guard][2].add(doc)
            index[guard].append(i)
    # 정렬해 두어 동점일 때도 실행(해시 시드)과 무관하게 같은 조건을 고른다.
    pool = sorted(g for g, (a, b, d) in stats.items()
                  if a >= good_min and len(d) >= docs_min and a / (a + b) >= precision)
    covered, chosen = [False] * len(occ), []
    while len(chosen) < cap:
        best = None
        for guard in pool:
            if guard in chosen:
                continue
            new = [i for i in index[guard] if not covered[i]]
            good = sum(occ[i][0] for i in new)
            if not new or good < good_min or good / len(new) < precision:
                continue
            if best is None or good - (len(new) - good) > best[0]:
                best = (good - (len(new) - good), guard, new)
        if best is None or best[0] <= 0:
            break
        chosen.append(best[1])
        for i in best[2]:
            covered[i] = True
    support = sum(occ[i][0] for i in range(len(occ)) if covered[i])
    return chosen, support, sum(covered) - support


def mine(data):
    occ = occurrences(data)
    print("train occurrences", len(occ), dict(Counter((o[2], o[0]) for o in occ)))
    results = []
    for precision, good_min, docs_min, cap in itertools.product(*GRID.values()):
        guards, support, wrong = greedy_guards(occ, precision, good_min, docs_min, cap)
        if not guards:
            continue
        rule = {"surface": SURFACE, "source": [list(m) for m in SOURCE], "index": 0, "pos": TARGET,
                "guards": [[kind, value] for kind, value in guards], "support": support}
        f1, changes = score(data, "dev", [rule])
        delta = {s: (f1[f"{s}/cand"] - f1[f"{s}/base"]) * 100 for s in ("all", "NX", "SX")}
        results.append({"config": [precision, good_min, docs_min, cap], "guards": len(guards),
                        "train_covered": [support, wrong], "dev_delta_pp": delta, "changes": changes, "rule": rule})
        print(results[-1]["config"], len(guards), {k: round(v, 4) for k, v in delta.items()}, changes, flush=True)
    eligible = [r for r in results if r["dev_delta_pp"]["NX"] >= 0]
    best = max(eligible, key=lambda r: (round(r["dev_delta_pp"]["SX"], 6), -r["guards"]))
    print("selected", best["config"], best["guards"], best["dev_delta_pp"])
    (data / f"rule_{SURFACE}.json").write_text(json.dumps(
        {"selected": best, "all": [{k: v for k, v in r.items() if k != "rule"} for r in results]},
        ensure_ascii=False, indent=1))


def decode(blob):
    """Section 17 blob → 규칙 목록 (context_rules.encode의 역)."""
    assert blob[:len(MAGIC)] == MAGIC
    offset = len(MAGIC)

    def take(fmt):
        nonlocal offset
        values = struct.unpack_from(fmt, blob, offset)
        offset += struct.calcsize(fmt)
        return values

    def string():
        nonlocal offset
        (length,) = take("<H")
        value = blob[offset:offset + length].decode("utf-8")
        offset += length
        return value

    rules = []
    for _ in range(take("<I")[0]):
        surface = string()
        source = [[string(), POS[take("<B")[0]]] for _ in range(take("<B")[0])]
        index, pos, count = take("<BBH")
        guards = [[take("<B")[0], string()] for _ in range(count)]
        rules.append({"surface": surface, "source": source, "index": index, "pos": POS[pos], "guards": guards})
    assert offset == len(blob)
    return rules


def append(rule_path, in_model, out_model, blob_path=None):
    import brotli
    parts = sections(Path(in_model).read_bytes())
    rules = decode(parts[17])
    assert encode(rules) == parts[17], "Section 17 재인코딩 불일치"
    rule = json.loads(Path(rule_path).read_text())["selected"]["rule"]
    key = (rule["surface"], [list(m) for m in rule["source"]])
    assert all((r["surface"], r["source"]) != key for r in rules), "같은 원래 열의 규칙이 이미 있다"
    parts[17] = encode(rules + [rule])
    raw = b"GMDL" + struct.pack("<I", 3) + b"".join(
        struct.pack("<BI", kind, len(data)) + data for kind, data in parts.items())
    Path(out_model).write_bytes(brotli.compress(raw, quality=11))
    if blob_path:
        Path(blob_path).write_bytes(parts[17])
    print(f"section17: {len(rules)}+1 rules, {len(parts[17]):,}B raw, model {Path(out_model).stat().st_size:,}B")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "strip":
        strip(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "mine":
        mine(Path(sys.argv[2]))
    elif mode == "append":
        blob = sys.argv[sys.argv.index("--blob") + 1] if "--blob" in sys.argv else None
        append(sys.argv[2], sys.argv[3], sys.argv[4], blob)
    else:
        sys.exit(f"unknown mode: {mode}")
