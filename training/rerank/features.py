"""재순위 perceptron feature 추출 + FNV-1a feature hashing.

Rust 이식 대상: feature 문자열 생성 규칙과 해시가 바이트 단위로 동일해야 함.
- feature 문자열은 UTF-8 인코딩 후 FNV-1a 64bit → bucket = hash & (DIM-1)
- tokens는 Rust analyze_topn_for_rerank의 출력 [form, pos, start, end] 리스트
  (재순위 입력 후처리 완료, 최종 context POS 교정과 표시용 자모 정규화 전).
  같은 어절의 토큰은 start가 같으므로, start가 바뀌는 곳이 공백이다.
"""

DIM = 1 << 20  # 1M buckets
# Section 14 ver: 3은 어절 경계 feature(sb:/spw:/sw:)를 포함한다 (rerank.rs FeatureSet::Spacing).
SECTION_VERSION = 3

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def fnv1a(s):
    h = FNV_OFFSET
    for b in s.encode("utf-8"):
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def extract(tokens, score_delta, rank):
    """한 후보의 sparse feature dict {name: value}.

    tokens: [[form, pos, start, end], ...]  score_delta: cand.score - min_score (>=0)
    rank: 1-based viterbi 순위
    """
    feats = {
        "bias": 1.0,
        "score_delta": score_delta,
        f"rank:{rank}": 1.0,
        "ntok": float(len(tokens)),
    }

    def add(name, v=1.0):
        feats[name] = feats.get(name, 0.0) + v

    prev2, prev, prev_form, prev_start = "<s2>", "<s>", "<s>", None
    for form, pos, start, *_ in tokens:
        add(f"u:{pos}")
        add(f"b:{prev}|{pos}")
        add(f"t:{prev2}|{prev}|{pos}")
        add(f"w:{form}/{pos}")
        add(f"pw:{prev_form}|{pos}")
        if prev_start is not None and start != prev_start:
            add(f"sb:{prev}|{pos}")
            add(f"spw:{prev_form}|{pos}")
            add(f"sw:{form}/{pos}")
        prev2, prev, prev_form, prev_start = prev, pos, form, start
    add(f"b:{prev}|</s>")
    add(f"t:{prev2}|{prev}|</s>")
    add(f"lw:{prev_form}/{prev}|</s>")

    # 문장 끝 신호: 마지막 비-기호 태그 (EC↔EF 최빈 플립 타깃)
    last_c = "<none>"
    for _, pos, *_ in reversed(tokens):
        if not pos.startswith("S"):
            last_c = pos
            break
    add(f"endc:{last_c}")
    return feats


_bucket_cache = {}


def _bucket(name):
    b = _bucket_cache.get(name)
    if b is None:
        b = fnv1a(name) & (DIM - 1)
        _bucket_cache[name] = b
    return b


def hash_feats(feats):
    """dict → ([bucket_id...], [value...]) 정렬된 병렬 리스트."""
    acc = {}
    for name, v in feats.items():
        b = _bucket(name)
        acc[b] = acc.get(b, 0.0) + v
    ids = sorted(acc)
    return ids, [acc[i] for i in ids]
