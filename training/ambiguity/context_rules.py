"""어절 전체 출력열 기준 한 POS 교정 규칙 (Section 17 후보).

Section 16은 출력이 Section 13 캐시와 같을 때만 고친다. 여기서는 production 최종
출력(형태·POS·span)이 학습 규칙의 원래 열과 정확히 같으면 캐시 여부와 상관없이 POS
하나를 바꾼다. 형태·토큰 수·순서·span은 유지한다.

    python3 training/ambiguity/context_rules.py mine DATA     # DATA/{train,dev}.jsonl + *.base.jsonl
    python3 training/ambiguity/context_rules.py select DATA   # dev 32설정 비교 → DATA/selected.json
    python3 training/ambiguity/context_rules.py export DATA/selected.json IN.gmdl OUT.gmdl [--blob B]

DATA는 prepare.py 산출물(train/dev)과 analyze_batch --json-spans 출력(<split>.base.jsonl).
고정 골드·2025 자료의 정답은 채굴·선택에 쓰지 않는다.
"""
import itertools
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "gold_testset/expand"), str(HERE.parent / "gate")]
from ep_norm import normalize_ep_morphemes  # noqa: E402

GUARDS_PER_RULE = 4
MIN_GOOD = 8          # 설정 격자의 최솟값. 채굴 단계에서 이보다 적은 후보는 버린다.
MIN_DOCS = 3
EDGE_BEFORE, EDGE_AFTER = 0, 13
EXACT_PREV, EXACT_NEXT = 14, 15
GRID = {"good": (8, 16), "docs": (3, 5), "precision": (0.98, 1.0), "ngram": (1, 2),
        "cap": (128, 512)}


def normalized(morphs):
    return tuple(tuple(m) for m in normalize_ep_morphemes([list(m[:2]) for m in morphs]))


def contexts(previous, following):
    """(kind, value, n). n=0은 n-gram이 아닌 조건(문장 경계·어절 전체)."""
    out = []
    if previous is None:
        out.append((EDGE_BEFORE, "", 0))
    else:
        out.append((EXACT_PREV, previous, 0))
        for n in range(1, 4):
            if len(previous) >= n:
                out.append((n, previous[-n:], n))
                out.append((9 + n, previous[:n], n))
    if following is None:
        out.append((EDGE_AFTER, "", 0))
    else:
        out.append((EXACT_NEXT, following, 0))
        for n in range(1, 4):
            if len(following) >= n:
                out.append((3 + n, following[:n], n))
                out.append((6 + n, following[-n:], n))
    return out


def guard_matches(kind, value, previous, following):
    n = len(value)
    if kind == EDGE_BEFORE:
        return previous is None
    if kind == EDGE_AFTER:
        return following is None
    if kind == EXACT_PREV:
        return previous == value
    if kind == EXACT_NEXT:
        return following == value
    if kind in (1, 2, 3, 10, 11, 12):
        word, want = previous, kind if kind <= 3 else kind - 9
        suffix = kind <= 3
    else:
        word, want = following, kind - 6 if kind >= 7 else kind - 3
        suffix = kind >= 7
    if word is None or n != want or len(word) < want:
        return False
    return (word[-want:] if suffix else word[:want]) == value


def eojeols(text, tokens):
    """공백 어절마다 (surface, source, prev, next, start, end).

    source는 모든 토큰 span이 어절 span과 같을 때의 (형태, POS) 열, 아니면 None."""
    words, spans, pos = text.split(), [], 0
    for w in words:
        start = text.index(w, pos)
        spans.append((start, start + len(w)))
        pos = start + len(w)
    out = []
    for i, (w, (b, e)) in enumerate(zip(words, spans)):
        local = [t for t in tokens if t[2] < e and b < t[3]]
        aligned = bool(local) and all(t[2] == b and t[3] == e for t in local)
        out.append((w, tuple((t[0], t[1]) for t in local) if aligned else None,
                    words[i - 1] if i else None, words[i + 1] if i + 1 < len(words) else None,
                    b, e))
    return out


def one_pos_fix(source, gold):
    """source에서 POS 하나만 바꿔 정답(ep_norm)과 같아지는 유일한 (index, pos)."""
    target = normalized(gold)
    if normalized(source) == target:
        return None
    gold_pos = {p for _, p in gold}
    hits = []
    for i, (form, old) in enumerate(source):
        for new in gold_pos:
            if new == old or new.startswith("S") or old.startswith("S"):
                continue
            trial = list(source)
            trial[i] = (form, new)
            if normalized(trial) == target:
                hits.append((i, new))
    return hits[0] if len(hits) == 1 else None


def corrected(source, fix):
    i, new = fix
    out = list(source)
    out[i] = (out[i][0], new)
    return tuple(out)


def read_rows(data, split):
    with open(data / f"{split}.jsonl", encoding="utf-8") as g, \
            open(data / f"{split}.base.jsonl", encoding="utf-8") as p:
        for gl, pl in itertools.zip_longest(g, p):
            assert gl is not None and pl is not None, "행 수 불일치"
            yield json.loads(gl), json.loads(pl)


def mine(data):
    """학습 문장에서 후보와 문맥 통계를 모은다."""
    support, support_docs = defaultdict(Counter), defaultdict(set)
    counts = Counter()
    for row, tokens in read_rows(data, "train"):
        for (surface, source, *_), word in zip(eojeols(row["text"], tokens), row["words"]):
            if source is None or not word[3]:
                continue
            counts["aligned_eojeols"] += 1
            fix = one_pos_fix(source, word[3])
            if fix is None:
                continue
            counts["one_pos_occurrences"] += 1
            support[surface, source][fix] += 1
            support_docs[surface, source, fix].add(row["doc"])

    targets = {}
    for key, fixes in support.items():
        fix, n = max(fixes.items(), key=lambda kv: (kv[1], kv[0]))
        if n >= MIN_GOOD and len(support_docs[(*key, fix)]) >= MIN_DOCS:
            targets[key] = fix
    counts["candidates"] = len(targets)

    stats = defaultdict(lambda: [0, 0, set()])
    ngram_of = {}
    occurrences = Counter()
    for row, tokens in read_rows(data, "train"):
        for (surface, source, prev, nxt, *_), word in zip(eojeols(row["text"], tokens), row["words"]):
            key = (surface, source)
            if source is None or key not in targets or not word[3]:
                continue
            good = normalized(corrected(source, targets[key])) == normalized(word[3])
            occurrences[key, good] += 1
            for kind, value, n in contexts(prev, nxt):
                ngram_of[kind] = n
                s = stats[key, kind, value]
                s[0 if good else 1] += 1
                if good:
                    s[2].add(row["doc"])
    counts["negative_occurrences"] = sum(v for (_, good), v in occurrences.items() if not good)
    stats = {k: (g, b, len(d), ngram_of[k[1]]) for k, (g, b, d) in stats.items()}
    mined = {"targets": targets, "stats": stats,
             "positives": {k: occurrences[k, True] for k in targets}, "counts": dict(counts)}
    with open(data / "mined.pkl", "wb") as f:
        pickle.dump(mined, f)
    print(json.dumps(dict(counts), ensure_ascii=False))
    return mined


def build_rules(mined, config):
    by_key = defaultdict(list)
    for (key, kind, value), (good, bad, docs, n) in mined["stats"].items():
        if good < config["good"] or docs < config["docs"]:
            continue
        if good / (good + bad) < config["precision"]:
            continue
        if n and n < config["ngram"]:
            continue
        by_key[key].append((good, bad, kind, value))
    rules = []
    for key, guards in by_key.items():
        surface, source = key
        index, pos = mined["targets"][key]
        chosen = sorted(guards, key=lambda g: (-g[0], g[1], g[2], g[3]))[:GUARDS_PER_RULE]
        rules.append({"surface": surface, "source": [list(m) for m in source],
                      "index": index, "pos": pos,
                      "guards": [[kind, value] for _, _, kind, value in chosen],
                      "support": mined["positives"][key]})
    rules.sort(key=lambda r: (-r["support"], r["surface"], r["source"]))
    return rules[:config["cap"]]


class RuleSet:
    def __init__(self, rules):
        self.by_key = {(r["surface"], tuple(map(tuple, r["source"]))): r for r in rules}

    def apply(self, text, tokens, user_words=frozenset()):
        """tokens: [form, pos, start, end] 목록. 바뀐 사본과 편집 목록을 돌려준다."""
        out = [list(t) for t in tokens]
        edits = []
        for surface, source, prev, nxt, b, e in eojeols(text, tokens):
            rule = self.by_key.get((surface, source)) if source else None
            if rule is None:
                continue
            if not any(guard_matches(k, v, prev, nxt) for k, v in rule["guards"]):
                continue
            if surface in user_words or any(f in user_words for f, _ in source):
                continue
            local = [i for i, t in enumerate(out) if t[2] == b and t[3] == e]
            at = local[rule["index"]]
            edits.append((at, out[at][0], out[at][1], rule["pos"]))
            out[at][1] = rule["pos"]
        return out, edits


def score(data, split, rules):
    from score import counts as sent_counts
    ruleset = RuleSet(rules)
    agg = defaultdict(lambda: [0, 0, 0])
    changes = Counter()
    for row, tokens in read_rows(data, split):
        new, edits = ruleset.apply(row["text"], tokens)
        before = sent_counts(tokens, row["morphemes"])
        after = sent_counts(new, row["morphemes"]) if edits else before
        for key in ("all", row["src"]):
            for i in range(3):
                agg[key, "base"][i] += before[i]
                agg[key, "cand"][i] += after[i]
        if edits:
            changes["changed"] += 1
            changes["pos_edits"] += len(edits)
            fb = 2 * before[0] / (before[1] + before[2])
            fa = 2 * after[0] / (after[1] + after[2])
            changes["win" if fa > fb + 1e-12 else "loss" if fa < fb - 1e-12 else "tie"] += 1
    f1 = {f"{k}/{v}": 2 * c[0] / (c[1] + c[2]) for (k, v), c in agg.items()}
    return f1, dict(changes)


def select(data):
    import brotli
    with open(data / "mined.pkl", "rb") as f:
        mined = pickle.load(f)
    results = []
    for values in itertools.product(*GRID.values()):
        config = dict(zip(GRID, values))
        rules = build_rules(mined, config)
        f1, changes = score(data, "dev", rules)
        size = len(brotli.compress(json.dumps(rules, ensure_ascii=False).encode(), quality=11))
        ok = all(f1[f"{s}/cand"] >= f1[f"{s}/base"] for s in ("NX", "SX"))
        results.append({"config": config, "rules": len(rules),
                        "guards": sum(len(r["guards"]) for r in rules), "json_brotli": size,
                        "dev": f1, "changes": changes, "nonnegative": ok})
        print(config, len(rules), f"{f1['all/cand'] * 100:.4f}", changes, ok, flush=True)
    eligible = [r for r in results if r["nonnegative"]]
    best = max(eligible, key=lambda r: (r["dev"]["all/cand"], -r["json_brotli"], -r["rules"]))
    rules = build_rules(mined, best["config"])
    (data / "selected.json").write_text(json.dumps(
        {"selection": best, "candidates": results, "rules": rules}, ensure_ascii=False, indent=1))
    print("selected", best["config"], best["rules"], f"{best['dev']['all/cand'] * 100:.4f}")


MAGIC = b"GCPR\x01"


def encode(rules):
    """Section 17 blob. crates/garu-core/src/codebook.rs ContextRules::from_bytes와 같은 포맷."""
    import struct
    from context_cache import POS

    def string(value):
        data = value.encode("utf-8")
        return struct.pack("<H", len(data)) + data

    out = bytearray(MAGIC + struct.pack("<I", len(rules)))
    for rule in rules:
        out += string(rule["surface"]) + struct.pack("<B", len(rule["source"]))
        for form, pos in rule["source"]:
            out += string(form) + struct.pack("<B", POS.index(pos))
        out += struct.pack("<BBH", rule["index"], POS.index(rule["pos"]), len(rule["guards"]))
        for kind, value in rule["guards"]:
            out += struct.pack("<B", kind) + string(value)
    return bytes(out)


def export(selected, in_model, out_model, blob_path=None):
    import struct
    import brotli
    from context_cache import sections

    blob = encode(json.loads(Path(selected).read_text())["rules"])
    parts = sections(Path(in_model).read_bytes())
    parts[17] = blob
    raw = b"GMDL" + struct.pack("<I", 3) + b"".join(
        struct.pack("<BI", kind, len(data)) + data for kind, data in parts.items())
    Path(out_model).write_bytes(brotli.compress(raw, quality=11))
    if blob_path:
        Path(blob_path).write_bytes(blob)
    print(f"section17: {len(blob):,}B raw, model {Path(out_model).stat().st_size:,}B brotli")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "export":
        blob = sys.argv[sys.argv.index("--blob") + 1] if "--blob" in sys.argv else None
        export(sys.argv[2], sys.argv[3], sys.argv[4], blob)
    else:
        {"mine": mine, "select": select}[mode](Path(sys.argv[2]))
