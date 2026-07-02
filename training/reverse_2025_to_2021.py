"""NIKL MP 2025 골드를 2021/Sejong(=Garu/v15k) 분절 컨벤션으로 역변환한다.

`eval_nikl_mp.py`의 `normalize_for_2025`는 채점용 *대칭* 정규화(gold·pred 양쪽을 2025의
거친 입도로 올림)다. 여기서는 그 역방향 — 2025 골드를 2021 입도로 *내려서* held-out 구어
평가나 도구 용도로 쓸 수 있게 한다. 검증: `normalize_for_2025(reverse(g)) ==
normalize_for_2025(g)`가 2025 SX 16,397/16,407 문장(99.94%)에서 성립 — 즉 채점 정규화기의
등가류 안에서 역함수다. 예외 10건은 아래 의도적 제외인 '동사 걸친 _ 복합어'뿐.
(eval_nikl2025_guueh.py --check-invertibility 로 재확인)

적용 규칙 (모두 2021 코퍼스 통계 기반, extract_nikl2025_reverse_whitelist.py):
  R0  '_' 결합 복합명사 un-join: 사업_분야/NNG → 사업/NNG + 분야/NNG (명사 POS만).
  R1  명사+하/되/시키/받 병합용언 분리: 생각하/VV → 생각/NNG + 하/XSV,
      공개되/VV → 공개/NNG + 되/XSV. 어휘화 용언(대하/위하/좋아하…)은 keep_merged로 유지.
      분리 판정은 어기가 content_dict 명사이거나 split_nouns에 있을 때.
  R3  간접인용 clitic 재병합: 가/VV+ㄴ다/EF+고/JKQ → 가/VV+ㄴ다고/EC (며/EC·는/ETM 동형),
      2025 인용격조사 라/이라/라고/JKQ → Garu의 라고/라며/라는(EC/ETM) canonical.

의도적 제외 (역변환 노이즈가 높아 net-negative로 확인됨):
  * 적/성/화 등 파생접미사(XSN) 분리 — 목적→목+적 류 오분리 다발. 채점용 대칭 정규화
    (normalize_for_2025 rule 2)에만 유효하고, 훈련/역변환 주입 시 Garu 대비 F1 하락
    (2025 SX 실측 0.882→0.881). 여기서는 병합형 NNG 그대로 둔다.
  * 동사 걸친 '_' 복합어 un-join — 음주_운전하/VV 류. un-join하면 앞부분에 VV 오태깅
    (음주/VV). 비명사 POS의 '_' 토큰은 분해하지 않고 '_'만 제거해 단일 토큰으로 둔다.

Usage (모듈): from reverse_2025_to_2021 import reverse_convert
      (CLI):   python3 training/reverse_2025_to_2021.py   # 내장 예제 실행
"""
import json
from pathlib import Path

_DIR = Path(__file__).parent
_WL = json.load(open(_DIR / "nikl2025_reverse_whitelist.json"))
KEEP_MERGED = set(_WL["keep_merged"])
SPLIT_NOUNS = set(_WL["split_nouns"])

# content_dict의 명사 표면 (분리 판정용). 없으면 split_nouns만으로 동작.
NOUNS = set(SPLIT_NOUNS)
_cd = _DIR / "codebook_data" / "content_dict.txt"
if _cd.exists():
    with open(_cd) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[1] in ("NNG", "NNP", "NNB", "NR"):
                NOUNS.add(p[0])

VERBALIZERS = ["시키", "당하", "받", "되", "하"]      # longest-first
NOMINAL_UNJOIN = {"NNG", "NNP", "NNB", "NR", "SL", "SH", "SN"}


def reverse_convert(morphs):
    """[[form,pos],...] (2025 gold) → [[form,pos],...] (2021 convention)."""
    # R0: '_' un-join — 명사 POS만 분해, 동사 걸친 복합어는 '_'만 제거(제외).
    exp = []
    for f, p in morphs:
        if "_" not in f:
            exp.append([f, p]); continue
        parts = [x for x in f.split("_") if x]
        if p in NOMINAL_UNJOIN and len(parts) >= 2:
            exp.extend([part, p] for part in parts)
        else:
            exp.append([f.replace("_", ""), p])
    morphs = exp

    out = []
    i, n = 0, len(morphs)
    while i < n:
        f, p = morphs[i]
        nf, npos = morphs[i + 1] if i + 1 < n else (None, None)

        # R1: 병합용언 → 명사 + XSV/XSA (하/되/시키/당하/받)
        if p in ("VV", "VA") and len(f) >= 2 and any(f.endswith(s) for s in VERBALIZERS):
            handled = False
            for suf in VERBALIZERS:
                if f.endswith(suf) and len(f) > len(suf):
                    base = f[:-len(suf)]
                    if f in KEEP_MERGED:              # 어휘화 → 유지
                        out.append([f, p]); handled = True; break
                    if base in NOUNS:                  # 생산적 → 분리
                        out.append([base, "NNG"])
                        out.append([suf, "XSA" if p == "VA" else "XSV"])
                        handled = True; break
            if handled:
                i += 1; continue
            out.append([f, p]); i += 1; continue      # 자·복합용언 → 유지

        # R3: 간접인용 clitic 재병합
        if p == "EF" and (nf, npos) in (("고", "JKQ"), ("며", "EC"), ("는", "ETM")):
            out.append([f + nf, "ETM" if npos == "ETM" else "EC"]); i += 2; continue
        if p == "JKQ" and f in ("라고", "이라고"):
            out.append(["라고", "EC"]); i += 1; continue
        if p == "JKQ" and f in ("라", "이라"):
            if (nf, npos) == ("며", "EC"):
                out.append(["라며", "EC"]); i += 2; continue
            if (nf, npos) == ("는", "ETM"):
                out.append(["라는", "ETM"]); i += 2; continue
            if (nf, npos) == ("고", "EC"):
                out.append(["라고", "EC"]); i += 2; continue

        out.append([f, p]); i += 1
    return out


if __name__ == "__main__":
    examples = [
        [["생각하", "VV"], ["았", "EP"], ["다", "EF"]],
        [["공개되", "VV"], ["었", "EP"], ["고", "EC"]],
        [["대하", "VV"], ["ㄴ", "ETM"]],
        [["만들", "VV"], ["ㄴ", "ETM"]],
        [["사업_분야", "NNG"], ["를", "JKO"]],
        [["음주_운전하", "VV"], ["면", "EC"]],   # 동사 걸친 복합어 → un-join 제외
        [["사회적", "NNG"], ["이", "VCP"]],       # XSN 분리 제외 → 병합 유지
        [["가", "VV"], ["ㄴ다", "EF"], ["고", "JKQ"]],
    ]
    for ex in examples:
        o = reverse_convert(ex)
        print(" ".join(f"{a}/{b}" for a, b in ex), " => ", " ".join(f"{a}/{b}" for a, b in o))
    print(f"keep_merged={len(KEEP_MERGED)} split_nouns={len(SPLIT_NOUNS)} nouns={len(NOUNS)}")
