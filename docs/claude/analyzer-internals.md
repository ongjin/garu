> **언제 읽나**: "[분석 오류]" GitHub 이슈를 디버깅하거나, `codebook.rs`(4,300줄)에 lattice 전략·후처리 규칙을 추가/수정할 때. 어떤 경로가 특정 분석 결과를 만드는지, 어디를 고쳐야 하는지의 지도.

# 분석기 내부 구조 (codebook.rs)

## 파이프라인

```
build_lattice(text)          # 아크 생성 (사전 + 코드북 + 재구성 전략 + 오타 아크)
  → 어절 캐시 아크 주입        # analyze()/analyze_topn() 안에서 (-2.0 비용)
  → viterbi() / viterbi_nbest # 문장 수준 Trigram 디코딩 (+ decode 내부 모음조화 후처리)
  → fix_* 후처리 체인          # POS/분절 교정 (analyze와 analyze_topn이 체인이 다름!)
  → 출력
```

## 분석 경로 = 아크가 어디서 오나

한 어절의 분석 후보(아크)는 4곳에서 나온다. 디버깅 1순위는 **"이 결과가 어느 출처에서 왔나"** 확정:

1. **내용어 사전 (FST, Section 6)** — 어간/명사. `드시/VV`(freq 428), `가시/NNG` 같은 어휘화 항목 포함. `content_dict.lookup()`.
2. **접미사 코드북 (Section 7, 31K 패턴)** — 활용형 통째를 **canonical 형태소**로 저장. 예: `무거워`→`[무겁/VA, 어/EF]`, `잡아`→`[잡/VV, 어/EF]`. ⚠️ 어미가 `어`로 저장되고 **후처리 모음조화가 `어→아`로 realize** (잡아). 즉 `[어간, 어]`는 "틀린 게 아니라 pre-harmony"다.
3. **스마트 어절 캐시 (Section 13, 큐레이션)** — Viterbi 오답 어절만. 매우 낮은 비용(-2.0)으로 주입되지만 **trigram이 뒤집을 수 있다**. (예: `드신다` 캐시는 `들+시+ㄴ다`인데 `진지를 드신다` 문맥에선 trigram이 `드시/VV`를 밀어 뒤집힘 — 이게 issue #4 표면 원인.) `eojeol_cache.bin`은 in-place 패칭(전체 리빌드는 -2pp 회귀).
4. **lattice 재구성 전략** (아래) — 사전/코드북/캐시에 없는 활용형을 합성.

## lattice 재구성 전략 (build_lattice 내부)

코드 주석의 라벨 그대로:

| 전략 | 하는 일 | 예 |
|------|---------|-----|
| **A (Strategy A)** | 내용어 + 접미사 코드북 매칭 (기본 경로) | |
| **A2b** | 모음축약 복원 (명령형 어미) | 건너라→건너+어라 |
| **A2c** | 첫 접미사 음절 종성분리 — 높임 `시`+ETM `ㄴ` 병합 해소 | 깨신→깨+시+ㄴ (개음절 어간만) |
| **E** | 모음축약 분해 (ㅘ/ㅝ/ㅙ → 어간모음+어미) | 이리와→오+아, 줘→주+어. ce_jong 0/20(ㅆ→었/았) |
| **A3** | 종성분리 — 어간 끝음절에 어미 자음 병합 | 고친다→고치+ㄴ다. ㅆ는 았/었 fallback |
| **A3v** | 모음 un-contraction — 축약 음절의 어간모음 복원 후 앞 음절에 병합해 다음절 VV/VA 조회 | 과거: 권했다→권하+았, 비쳤다→비치+었 (했→하+았, ㅕㅆ→ㅣ+었). 현재/연결: 권해서→권하+아서, 비쳐서→비치+어서 (해→하+아, ㅕ→ㅣ+어, **prefix_len≥2만** — 해/펴 단음절 노이즈 방지). VV/VA/VX 사전 게이트. ⚠️A3는 했→해/렸→려로 모음을 안 풀어 다음절 어간을 못 찾음 — A3v가 보완 |
| **A4** | ㄹ불규칙 탈락 복원 | 무니→물+니, 사세요→살+세요 |
| **D** | 오타 교정(자모 치환) — **OOV 위치에서만** | 햇다→했다 |
| **B / C** | 순수 기능 접미사 standalone / 내용어+기능 contracted | |
| — | `입니다/입니까` VCP+EF 아크 강제 주입 | 진입/NNG에 안 지도록 |

⚠️ ㅂ불규칙 현재형(아름다워)은 **전용 재구성 코드가 없다** — 접미사 코드북이 `[아름답/VA, 어/EF]`로 통째 저장한 걸 쓴다. 과거형(아름다웠다/추웠다)·불규칙 존댓말(고우시다/걸으신다/저으신다)은 `build_codebook_model.py`의 `augment_irregular_conjugations`(ㅂ `었` 분기 + ㅅ불규칙 분기 `IRREG_SIOT_STEMS` + ㅡ탈락 `았/었` ㅆ-병합 버그수정)와 `augment_irregular_honorific`(ㅂ/ㄷ/ㅅ 어간 + `으시/EP` 어절 주입)로 코드북에 넣어 복원됨. 회귀 가드 `training/test_irregular_restore.py`.

## 디코더: viterbi() vs viterbi_nbest()

- `analyze()` → **`viterbi()`** (single-best). trigram + 내부 형태소 전이 + **EOS 비용** 포함.
- `analyze_topn()` → **`viterbi_nbest()`** (top-k). dump_topk 예제가 이걸 씀.
- ⚠️ **두 디코더가 1위를 다르게 낼 수 있다** (비용 모델 미세 차이, 특히 EOS 처리). issue #4 `드신다`에서 nbest #1은 `들+시+ㄴ다`인데 single-best는 다르게 골랐음. dump_topk 결과 ≠ analyze() 결과일 수 있으니 주의.

## 후처리

**(1) decode 함수 내부** (viterbi/viterbi_nbest 양쪽에 중복 존재):
- `었→았` 모음조화 (앞 어간 끝모음 ㅏ/ㅗ)
- `어→아` 모음조화 — **canonical `어`를 양성어간 뒤에서 `아`로 realize하는 일반 메커니즘**. ㅂ불규칙은 `is_pieup_irregular_keeps_eo`로 가드(표면형 검사: 정규 `잡아`⊃`잡`, 불규칙 `아름다워`⊅`아름답`). 곱/돕만 예외.
- EP 뒤 `아→어` 역정규화, `merge_ec_ef_vowel`, `fix_xsv_xsa`

**(2) `fix_*` 체인** — ⚠️ **`analyze()`(24개)와 `analyze_topn()`(34개)가 다르다.** analyze_topn에만 있는 것: `fix_oneora, fix_haeyo_endings, fix_yo_jx_merge, fix_extra_endings, fix_mag_copula_ya, fix_myeoch_si_ya, fix_sn_counter_copula, fix_han_standalone_mm, fix_lge_endings, fix_iri_mag`. **새 fix_ 규칙 추가 시 두 곳 다 등록**할지 판단할 것 (issue #4 `fix_deusi_honorific`, `fix_geuraeseo_maj`, `fix_si_dependent_noun`, `fix_quote_jkq`는 양쪽 등록).

⚠️ **POS 보정의 일부는 codebook이 아니라 `model.rs::analyze_inner`의 R1~R6 override가 최종 결정** (codebook fix_* 이후 실행되므로 codebook에서 고쳐도 덮인다). 시간명사 오늘/어제/지금(조사앞 NNG·그 외 MAG)은 **R1**, 내일→NNG는 R2, 뭐/저기→NP는 R3/R4, NNP→NNG 힌트는 R6. 시간명사·지시대명사 POS는 여기를 고칠 것.

## 디버깅 방법

```bash
# 단일/배치 분석 (analyze 경로 = 실제 출력)
GARU_MODEL=js/models/base.gmdl cargo run -q --release --example analyze_batch <입력.txt> [--json]

# top-k 후보 + 점수 (analyze_topn 경로 = 후보가 lattice에 있는지 확인)
GARU_MODEL=js/models/base.gmdl cargo run -q --release --example dump_topk <입력.txt> [k]

# raw 아크 (출처 추적용) — 정답 분해 아크가 lattice에 있는지/비용이 얼마인지 확인
GARU_MODEL=js/models/base.gmdl cargo run -q --release --example dump_arcs "<문장>"
#   [start..end] cost  형태소... 를 (start,end,cost)순 정렬 출력. cb.dump_arcs()는 캐시 주입 전 lattice.

# 골드 F1 회귀 측정 (norm=헤드라인, --no-norm=raw)
(cd training/gold_testset && python3 eval_f1.py --analyzers garu [--no-norm])
```

디버깅 순서: ① dump_topk로 **정답 후보가 lattice에 있나** 확인 → ② 없으면 재구성 전략/코드북 갭(예: 으시 OOV), 있는데 랭킹이 밀리면 비용/trigram 문제, 1위가 맞는데 출력이 다르면 **후처리 fix_*/모음조화가 망가뜨리는 것** (issue #5가 이 케이스).

## 이번 세션에서 확인된 함정

- 접미사 코드북은 어미를 `어`로 저장 → `[어간, 어]`는 pre-harmony (정상). 모음조화 후처리가 realize.
- ㅂ불규칙 정규/불규칙은 어간 토큰만으론 구분 불가 → **표면형** 필요 (둘 다 코드북이 `[stem, 어]`로 줌).
- 어절 캐시 ≠ 사전 ≠ 코드북이 같은 어절에 서로 다른 분석을 줄 수 있고, 최종 선택은 trigram이 결정.
- `analyze()`와 `analyze_topn()`의 후처리가 달라 결과가 갈릴 수 있음.
