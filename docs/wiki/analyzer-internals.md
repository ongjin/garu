> **언제 읽나**: "[분석 오류]" GitHub 이슈를 디버깅하거나, `codebook.rs`·`model.rs`에 lattice 전략·후처리 규칙을 추가/수정할 때. 어떤 경로가 특정 분석 결과를 만드는지, 어디를 고쳐야 하는지의 지도.

# 분석기 내부 구조 (codebook.rs)

`CodebookAnalyzer`는 탐색 후보를 만들고, 공개 `model::Analyzer`는 그 후보의 선택과 최종 교정을 담당한다. 학습용 후보와 사용자에게 표시하는 결과의 후처리 범위를 구분해야 분석 오류와 재순위 피처 불일치를 추적할 수 있다.

## 파이프라인

```
build_lattice(text)          # 아크 생성 (사전 + 코드북 + 재구성 전략 + 오타 아크)
  → dedup_arcs               # 같은 span·같은 형태소열 중복 아크 제거 (열등한 쪽만)
  → 어절 캐시 아크 주입        # 기본 -1.0, 마지막 어절의 EC 종결 캐시는 -0.5
  → viterbi() / viterbi_nbest # 문장 수준 Trigram 디코딩 (+ decode 내부 모음조화 후처리)
  → fix_* 후처리 체인          # POS/분절 교정 (analyze와 analyze_topn이 체인이 다름!)
  → 후처리 결과 dedup          # analyze_topn만. fix_*가 접어버린 동일 결과 제거
  → CodebookAnalyzer 출력    # 공개 Analyzer의 선택·최종 교정은 아래 참조
```

**중복 제거가 두 군데인 이유** (research-history #44): 아크는 사전·코드북·재구성 전략이 같은 분석을 각각 만들어 실측 9%가 완전 중복이고, `viterbi_nbest` 내부 dedup은 **후처리 이전** 형태를 비교하므로 `fix_nde_merge` 같은 규칙이 서로 다른 경로를 같은 결과로 접으면 중복이 그대로 남는다(같은 토큰열이 rank 1·2·6에 동시 등장하던 상태). 둘 다 N-best 슬롯만 축내고 재순위 feature의 `rank`를 왜곡한다. **후보를 더 만드는 방향은 반대로 회귀**한다 — 역추적 풀을 2배로 늘려 실효 후보 6.76개를 준 실험은 v15k −0.05pp였다(재순위 픽 품질이 병목).

## 분석 경로 = 아크가 어디서 오나

한 어절의 분석 후보(아크)는 다음 경로에서 나온다. 디버깅 1순위는 **"이 결과가 어느 출처에서 왔나"** 확정:

1. **내용어 사전 (FST, Section 6)** — 어간/명사. `드시/VV`(freq 428), `가시/NNG` 같은 어휘화 항목 포함. `content_dict.lookup()`.
2. **접미사 코드북 (Section 7, 31K 패턴)** — 활용형 통째를 **canonical 형태소**로 저장. 예: `무거워`→`[무겁/VA, 어/EF]`, `잡아`→`[잡/VV, 어/EF]`. ⚠️ 어미가 `어`로 저장되고 **후처리 모음조화가 `어→아`로 realize** (잡아). 즉 `[어간, 어]`는 "틀린 게 아니라 pre-harmony"다. ⚠️ **로드 임계 `MIN_SUFFIX_FREQ=75`가 정답 분석을 버릴 수 있다** — 같은 표면에 여러 분석이 있을 때 임계 미만인 쪽은 아크가 아예 안 생긴다(`개가`의 `[개/NNG, 가/JKS]` freq 72가 탈락해 `개가 짖는다`→개/NNB였던 사례, research-history #48). `suffix_codebook.json`에는 있는데 dump_arcs에 없으면 이 임계를 의심하고 `SPOKEN_SUFFIX_BUMPS`로 올릴 것.
3. **스마트 어절 캐시 (Section 13, 큐레이션)** — Viterbi 오답 어절만. 기본 비용 -1.0으로 주입되지만 **trigram이 뒤집을 수 있다**. 마지막 어절의 EC 종결 캐시는 -0.5로 완화해 EF 후보가 경쟁하게 한다. (예: `드신다` 캐시는 `들+시+ㄴ다`인데 `진지를 드신다` 문맥에선 trigram이 `드시/VV`를 밀어 뒤집힘 — 이게 issue #4 표면 원인.) `eojeol_cache.bin`은 in-place 패칭(전체 리빌드는 -2pp 회귀). ⚠️ **반대 함정: 낡은 캐시 항목이 사전 수리를 눌러버린다** — `대박` 캐시가 `[대/XPN,박/NNG]`로 남아 있어 사전 freq 5000 보정(decf9a7) 후에도 단독/문두 어절이 계속 깨졌음(research-history #38, 캐시 in-place 교정으로 수리). **사전 freq를 고쳤는데 특정 어절이 안 낫는다면 캐시에 같은 표면의 항목이 있는지 먼저 확인할 것** (dump_arcs에 없는 형태소가 최종 출력에 있으면 캐시 출처 확정).
4. **lattice 재구성 전략** (아래) — 사전/코드북/캐시에 없는 활용형을 합성.
5. **런타임 사용자 사전** (`add_user_word`, 모델 밖) — `build_lattice` 끝(OOV fallback 직전)에서 표면 매칭으로 통짜 아크 주입. 비용은 `get_word_cost(surface, pos, freq)`라 사전 아크와 같은 척도로 경쟁한다(기본 freq 5000). 어절 경계를 넘지 않으며, 등록이 없으면 주입 루프 자체를 건너뛰어 기존 동작과 byte-identical. **사용자가 "등록했는데 안 먹는다"고 하면 freq를 올려볼 것** — 캐시(기본 -1.0)나 고빈도 2분해가 이길 수 있다. Section 16의 최종 교정은 어절 전체 또는 교정 대상 형태소가 사용자 사전에 등록되어 있으면 건너뛴다.

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

- `CodebookAnalyzer::analyze()` → **`viterbi()`** (single-best). trigram + 내부 형태소 전이 + **EOS 비용** 포함.
- `CodebookAnalyzer::analyze_topn()` → **`viterbi_nbest()`** (top-k). 공개 `Analyzer`와 학습 덤프가 이 후보 경로를 사용한다.
- ⚠️ **두 디코더가 1위를 다르게 낼 수 있다** (비용 모델 미세 차이, 특히 EOS 처리). issue #4 `드신다`에서 nbest #1은 `들+시+ㄴ다`인데 single-best는 다르게 골랐음. dump_topk 결과 ≠ analyze() 결과일 수 있으니 주의.

## 후처리

**(1) decode 함수 내부** (viterbi/viterbi_nbest 양쪽에 중복 존재):
- `었→았` 모음조화 (앞 어간 끝모음 ㅏ/ㅗ)
- `어→아` 모음조화 — **canonical `어`를 양성어간 뒤에서 `아`로 realize하는 일반 메커니즘**. ㅂ불규칙은 `is_pieup_irregular_keeps_eo`로 가드(표면형 검사: 정규 `잡아`⊃`잡`, 불규칙 `아름다워`⊅`아름답`). 곱/돕만 예외.
- EP 뒤 `아→어` 역정규화, `merge_ec_ef_vowel`, `fix_xsv_xsa`

**(2) `fix_*` 체인** — ⚠️ **`analyze()`(29개)와 `analyze_topn()`(39개)가 다르다.** analyze_topn에만 있는 것: `fix_oneora, fix_haeyo_endings, fix_yo_jx_merge, fix_extra_endings, fix_mag_copula_ya, fix_myeoch_si_ya, fix_sn_counter_copula, fix_han_standalone_mm, fix_lge_endings, fix_iri_mag`. **새 fix_ 규칙 추가 시 두 곳 다 등록**할지 판단할 것 (issue #4 `fix_deusi_honorific`, `fix_geuraeseo_maj`, `fix_si_dependent_noun`, `fix_quote_jkq`, `fix_it_copula`와 2026-07 룰 4종 `fix_geureon_mm`/`fix_seyo_merge`/`fix_jeoneun_np`/`fix_present_copula`는 양쪽 등록). **주의: model.rs Analyzer는 `analyze_topn`만 쓴다 — analyze에만 등록하면 실 파이프라인에서 무발동**(fix_it_copula 최초 누락 사례).

⚠️ **POS 보정의 일부는 codebook이 아니라 `model.rs`의 R1~R6 override에 있다** (codebook fix_* 이후 실행되므로 codebook에서 고쳐도 덮인다). 시간명사 오늘/어제/지금(조사앞 NNG·그 외 MAG)은 **R1**, 내일→NNG는 R2, 뭐/저기→NP는 R3/R4, NNP→NNG 힌트는 R6. 시간명사·지시대명사 POS는 여기와 그 뒤의 Section 16 적용 여부를 함께 확인한다.

**재순위 perceptron (GMDL Section 14, 2026-07-13 채택 — 현 배포 모델 포함)**: Section 14가 있으면 후보 폭이 k=5→10으로 확장되고, `analyze_inner`의 최종 선택 블록(viterbi cost − context bonus argmin) 뒤에서 perceptron 점수가 확신 마진(모델에 저장, τ=4)을 넘으면 선택을 교체. feature 입력은 후보 clone에 `apply_rerank_rules`로 model 레벨 보정(adj_root_xsa/protected_aux/R1~R6 순서)을 적용한 것 — 학습 덤프가 쓰는 `Analyzer::analyze_topn_for_rerank`와 같은 함수다. Section 16 교정과 표시용 자모 정규화는 점수 입력에 포함하지 않는다. early-return 경로(의존명사 패턴)는 재순위를 타지 않음. Section 14 없으면 재순위만 무동작(k=5). ⚠️ nbest는 k에 따라 상위권 구성이 달라진다(빔 효과): nbest(5) ≠ nbest(20)[:5] — k=5 재순위가 +0.25pp에 그친 원인. **rank1은 k와 무관하게 동일하지만(9,000문장 100% 일치) rank2 이하는 k에 의존한다** — 그래서 `cand[1].score − cand[0].score` 마진을 작은 k로 미리 구해 큰 k를 건너뛰는 최적화는 성립하지 않는다(research-history #39).

**문맥 POS 교정 (GMDL Section 16)**: `apply_context_cache`는 주변 어절 조건 중 하나가 맞고 선택된 어절의 형태·POS열·문자 span이 원래 Section 13 캐시와 정확히 일치할 때만 POS 하나를 바꾼다. 형태·토큰 수·순서·span·점수와 다른 토큰은 그대로 둔다. 사용자 사전에 어절 전체나 교정 대상 형태소가 있으면 적용하지 않는다. 아크를 추가하거나 재순위를 다시 실행하지 않으며, `analyze_inner`가 조기 반환한 결과도 공개 `analyze`의 마지막 단계에서 검사한다. 규칙 정본과 학습·평가 경계는 [모델 빌드 문서](model-build.md)에 있다.

**구어 탐색 비용 보정 (GMDL Section 18)**: `analyze`·`analyze_topn`이 래티스를 만들고 어절 캐시 아크를 주입한 직후 `apply_search_costs`를 호출한다. 먼저 입력 전체(앞뒤 공백 제외)의 문자 1–3-gram으로 구어 판별 점수를 낸다. 임계값을 넘을 때만 각 아크의 품사열·길이·형태소·어절 경계 feature 가중치 합을 아크 비용에 더한다. 그래서 같은 문장이라도 다른 문장과 이어 붙여 입력하면 판별 결과가 달라질 수 있다. `analyze_topn_for_rerank`도 같은 보정을 거친 후보를 돌려준다. 디버깅할 때 `dump_arcs`의 비용은 보정 전 값이다.

**최종 출력열 POS 교정 (GMDL Section 17)**: Section 16 다음에 `apply_context_rules`가 실행된다. 어절의 모든 토큰 span이 어절 span과 같고 형태·POS열이 규칙의 원래 열과 정확히 같으며, 이웃 어절 조건(앞뒤 어절의 1~3자 접두·접미, 어절 전체, 문장 경계) 중 하나가 맞을 때 POS 하나를 바꾼다. n자 조건은 이웃 어절이 n자 이상일 때만 맞는다. 어절 전체나 원래 열의 어느 형태소라도 사용자 사전에 있으면 건너뛴다. 연쇄 적용은 없다.

| 공개 Analyzer API | 후보·결과 처리 |
|---|---|
| `analyze` | 후보 선택과 기존 보정 → Section 16 → Section 17 → 옵션에 따른 표시용 자모 정규화 |
| `analyze_topn` | 학습용 후보 각각에 Section 16 → Section 17 → 옵션에 따른 표시용 자모 정규화. 재순위 선택은 하지 않음 |
| `analyze_topn_for_rerank` | Codebook N-best → `apply_rerank_rules`. Section 16·17·표시용 정규화 없이 실제 재순위 입력 반환 |

**선택적 경계 복원은 이 경로 밖에 있다.** `garu-ko/text`의 `restoreSpacing`·`splitUnspaced`를 가져와 명시적으로 호출하며, 기존 `analyze`·`tokenize`는 띄어쓰기를 자동 복원하지 않는다. JS 구현은 `js/src/text.ts`·`js/src/boundaries.ts`·`js/src/boundary-data.ts`이고 추가 경계 가중치는 기본 GMDL과 별개다. 학습과 선택 후 검증의 한계는 [경계 모델 README](../../training/boundaries/README.md)를 따른다.

## 디버깅 방법

```bash
# 단일/배치 분석 (analyze 경로 = 실제 출력)
GARU_MODEL=js/models/base.gmdl cargo run -q --release --example analyze_batch <입력.txt> [--json]

# top-k 학습 후보 + 점수 (analyze_topn_for_rerank, Section 16·표시용 자모 정규화 제외)
GARU_MODEL=js/models/base.gmdl cargo run -q --release --example dump_topk <입력.txt> [k]

# raw 아크 (출처 추적용) — 정답 분해 아크가 lattice에 있는지/비용이 얼마인지 확인
GARU_MODEL=js/models/base.gmdl cargo run -q --release --example dump_arcs "<문장>"
#   [start..end] cost  형태소... 를 (start,end,cost)순 정렬 출력. cb.dump_arcs()는 캐시 주입 전 lattice.
#   ⚠️ 그래서 **캐시·사용자 사전 아크는 여기 안 보인다** — dump_arcs에 없는데 최종 출력에 있으면
#   캐시(또는 사용자 사전) 출처로 확정할 것. "아크가 없다"고 사전 결손으로 오판하기 쉽다
#   (이불/NNG가 suffix-충돌 제거로 FST에서 빠졌는데 캐시가 가려주던 사례, research-history #41).

# 골드 F1 회귀 측정 (norm=헤드라인, --no-norm=raw)
(cd training/gold_testset && python3 eval_f1.py --analyzers garu [--no-norm])

# 프로파일링 — release는 strip=true라 그냥 sample하면 주소만 나온다. 심볼 빌드 필수.
CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=1 \
  cargo build --release --example analyze_batch
GARU_MODEL=js/models/base.gmdl ./target/release/examples/analyze_batch <큰입력> >/dev/null &
sample $(pgrep -n analyze_batch) 12 -mayDie -f prof.txt
```

⚠️ **속도 A/B는 단발 `time`으로 판정하지 말 것.** 같은 바이너리 재실행 편차가 3~5%라 실제 +7% 이득이 노이즈에 묻힌다. 두 바이너리를 교차 실행하고 **min**을 비교할 것(research-history #39).

디버깅 순서: ① dump_topk로 **정답이 탐색·기존 후처리를 거친 후보에 남아 있나** 확인 → ② 없으면 raw 아크와 비교해 재구성 전략/코드북 갭(예: 으시 OOV)과 후처리 소실을 구분 → ③ 후보는 맞는데 실제 출력이 다르면 문맥 보너스·재순위 선택과 Section 16의 최종 POS 교정을 확인한다. dump_topk에는 Section 16 교정 전 후보가 나오므로 최종 POS 차이만으로 후보 결손을 단정하지 않는다.

## 이번 세션에서 확인된 함정

- 접미사 코드북은 어미를 `어`로 저장 → `[어간, 어]`는 pre-harmony (정상). 모음조화 후처리가 realize.
- ㅂ불규칙 정규/불규칙은 어간 토큰만으론 구분 불가 → **표면형** 필요 (둘 다 코드북이 `[stem, 어]`로 줌).
- 어절 캐시 ≠ 사전 ≠ 코드북이 같은 어절에 서로 다른 분석을 줄 수 있고, 최종 선택은 trigram이 결정.
- `analyze()`와 `analyze_topn()`의 후처리가 달라 결과가 갈릴 수 있음.
- **표면 환각(정합성 버그)**: 접미사 코드북에 키 표면 ≠ 재구성 표면인 **절단 엔트리**가 있었음 (예: 키 `배`→`[배우/VV,었/EP]`(배웠), `표`→`[표,하,었]`(표했), `시장`→`[시,장악,하,었]`). 어떤 strategy든 이를 `span=키길이` 아크로 주입하면 출력 표면이 입력과 불일치(`2.2배다.`→`배출/하/았/다`). **해결(commit)**: `parse_suffix_codebook`에서 로드 시 `suffix_analysis_stem_fits`로 필터 — stem 접두(첫 어미 전 = 내용+하/되 XSV·계사 이) 표면 길이 > 키 길이인 analysis를 제거. stem은 어미로 축약될 뿐 확장 안 되므로 초과는 불가능. 로드 필터라 A2/A3/C 모든 strategy 일괄 차단(코드북 sourced 절단 아크 11588→590으로 감소, 잔여는 content_dict 단어+접미사 조합이라 별개 mechanism·잠재). dump_arcs로 `[i..i+1]`인데 형태소가 다음절이면 이 클래스.

## 이력

- 2026-10-20 Section 16 최종 POS 교정과 사용자 사전 보호를 분석 지도에 반영. 학습 후보 API와 공개 결과의 후처리 범위를 분리하고, 캐시 기본 비용을 현행 -1.0으로 바로잡음. 선택적 JS 경계 복원은 기존 분석 경로 밖으로 명시.
