"""normalize_for_2025 단위테스트.

NIKL 형태분석 2025의 거친 분절 컨벤션을 gold·pred 대칭 정규화로 Garu/2021
입도에 맞춘다. 각 케이스는 gold(2025) 표면과 Garu 표면이 정규화 후 동일해지는지
확인 — 즉 normalize(gold) == normalize(garu) 여야 채점이 convention-neutral.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from eval_nikl_mp import normalize_for_2025 as N


# ── 0) '_' 결합 복합명사 un-join ──────────────────────────────────────────
def test_unjoin_compound():
    assert N([("사업_분야", "NNG")]) == [("사업", "NNG"), ("분야", "NNG")]


# ── 1) 명사/어근 + XSV/XSA → 용언 ─────────────────────────────────────────
def test_xsv_merge():
    assert N([("방문", "NNG"), ("하", "XSV")]) == [("방문하", "VV")]

def test_xsa_merge():
    assert N([("건강", "NNG"), ("하", "XSA")]) == [("건강하", "VA")]


# ── 2) 명사/어근 + 적/XSN → X적/NNG (③) ──────────────────────────────────
def test_jeok_merge_nng():
    # Garu: 사회/NNG + 적/XSN → gold 2025: 사회적/NNG
    assert N([("사회", "NNG"), ("적", "XSN")]) == [("사회적", "NNG")]

def test_jeok_merge_symmetric_gold_noop():
    # gold 2025는 이미 병합형(사회적/NNG) — 룰 미발동, 그대로.
    assert N([("사회적", "NNG")]) == [("사회적", "NNG")]

def test_jeok_only_xsn_suffix():
    # 다른 XSN 접미사(들)는 병합 안 함 — 적/XSN에만 적용.
    assert N([("사람", "NNG"), ("들", "XSN")]) == [("사람", "NNG"), ("들", "XSN")]

def test_jeok_lexical_noun_untouched():
    # 목적/NNG 같은 어휘명사(적/XSN 분리 아님)는 손대지 않음.
    assert N([("목적", "NNG"), ("이", "JKS")]) == [("목적", "NNG"), ("이", "JKS")]


# ── 3) EF + 인용 clitic → 병합 (④A 다고/다며/다는) ────────────────────────
def test_indirect_go_merge():
    # gold: 가/VV + ㄴ다/EF + 고/JKQ  →  Garu: 가/VV + ㄴ다고/EC
    gold = [("가", "VV"), ("ㄴ다", "EF"), ("고", "JKQ")]
    garu = [("가", "VV"), ("ㄴ다고", "EC")]
    assert N(gold) == N(garu) == [("가", "VV"), ("ㄴ다고", "EC")]

def test_indirect_go_past():
    gold = [("가", "VV"), ("았", "EP"), ("다", "EF"), ("고", "JKQ")]
    garu = [("가", "VV"), ("았", "EP"), ("다고", "EC")]
    assert N(gold) == N(garu)

def test_imperative_ra_go_is_ef():
    # 오라고 지배형: 오/VV + 라/EF + 고/JKQ  →  Garu 오/VV + 라고/EC
    gold = [("오", "VV"), ("라", "EF"), ("고", "JKQ")]
    garu = [("오", "VV"), ("라고", "EC")]
    assert N(gold) == N(garu) == [("오", "VV"), ("라고", "EC")]

def test_indirect_mye_merge():
    # gold: 가/VV + 겠/EP + 다/EF + 며/EC  →  Garu: 다며/EC
    gold = [("가", "VV"), ("겠", "EP"), ("다", "EF"), ("며", "EC")]
    garu = [("가", "VV"), ("겠", "EP"), ("다며", "EC")]
    assert N(gold) == N(garu)

def test_indirect_neun_merge():
    # gold: 가/VV + ㄴ다/EF + 는/ETM  →  Garu: 가/VV + ㄴ다는/ETM
    gold = [("가", "VV"), ("ㄴ다", "EF"), ("는", "ETM")]
    garu = [("가", "VV"), ("ㄴ다는", "ETM")]
    assert N(gold) == N(garu) == [("가", "VV"), ("ㄴ다는", "ETM")]

def test_plain_connective_go_untouched():
    # 평서 연결 고/EC(먹고)는 EF 뒤가 아니므로 병합 안 함.
    assert N([("먹", "VV"), ("고", "EC")]) == [("먹", "VV"), ("고", "EC")]

def test_adnominal_neun_untouched():
    # 일반 관형 는/ETM(하는)은 동사어간 뒤 — EF 뒤가 아니라 그대로.
    assert N([("하", "VV"), ("는", "ETM")]) == [("하", "VV"), ("는", "ETM")]


# ── 4) 계사 인용 (④B): 간접 복사문은 이/VCP 유지, 직접인용 JKQ만 canonical ──
def test_indirect_copula_go_keeps_vcp():
    # 예능이라고: 2025 gold 이/VCP + 라/EF + 고/JKQ  →  Garu 이/VCP + 라고/EC.
    # 3번(EF+고/JKQ 병합)이 처리, 계사 이/VCP는 양쪽에 그대로 남아 매치.
    gold = [("예능", "NNG"), ("이", "VCP"), ("라", "EF"), ("고", "JKQ")]
    garu = [("예능", "NNG"), ("이", "VCP"), ("라고", "EC")]
    assert N(gold) == N(garu) == [("예능", "NNG"), ("이", "VCP"), ("라고", "EC")]

def test_indirect_copula_neun_keeps_vcp():
    # 단점이라는: gold 이/VCP + 라/EF + 는/ETM  →  Garu 이/VCP + 라는/ETM.
    gold = [("단점", "NNG"), ("이", "VCP"), ("라", "EF"), ("는", "ETM")]
    garu = [("단점", "NNG"), ("이", "VCP"), ("라는", "ETM")]
    assert N(gold) == N(garu) == [("단점", "NNG"), ("이", "VCP"), ("라는", "ETM")]

def test_indirect_copula_mye_keeps_vcp():
    gold = [("우려", "NNG"), ("이", "VCP"), ("라", "EF"), ("며", "EC")]
    garu = [("우려", "NNG"), ("이", "VCP"), ("라며", "EC")]
    assert N(gold) == N(garu) == [("우려", "NNG"), ("이", "VCP"), ("라며", "EC")]

def test_direct_quote_rago_symmetric():
    # 직접인용 "…"라고: 양쪽 다 라고/JKQ → 동일 canonical (EC)로 상호 매치.
    gold = [("다", "EF"), ("”", "SS"), ("라고", "JKQ")]
    garu = [("다", "EF"), ("”", "SS"), ("라고", "JKQ")]
    # 다/EF는 SS 앞이라 3번 규칙(EF+clitic 인접) 미발동, 4b가 라고만 canonical화.
    assert N(gold) == N(garu) == [("다", "EF"), ("”", "SS"), ("라고", "EC")]

def test_direct_quote_ramye():
    # 직접인용 "…"라며: Garu 라며/EC(단일) ↔ gold 라/JKQ + 며/EC → 둘 다 라며/EC.
    garu = [("다", "EF"), ("”", "SS"), ("라며", "EC")]
    gold = [("다", "EF"), ("”", "SS"), ("라", "JKQ"), ("며", "EC")]
    assert N(garu) == N(gold)


# ── 손대지 말 것: 진짜 차이는 마스킹 금지 ─────────────────────────────────
def test_copula_non_quote_untouched():
    # 이/VCP + 다/EF(평서 계사문 "사과다")는 인용 아님 — 그대로.
    assert N([("사과", "NNG"), ("이", "VCP"), ("다", "EF")]) == \
        [("사과", "NNG"), ("이", "VCP"), ("다", "EF")]

def test_copula_rado_untouched():
    # 이/VCP + 라도/EC("학생이라도")는 인용 라고/라며/라는 아님 — 그대로.
    assert N([("학생", "NNG"), ("이", "VCP"), ("라도", "EC")]) == \
        [("학생", "NNG"), ("이", "VCP"), ("라도", "EC")]


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"OK ({len(TESTS)} tests)")
