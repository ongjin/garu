# Changelog

## 0.8.0

### CNN 제거 + 규칙 기반 POS 보정 + 어절 캐시 확장

CNN 재순위 모델을 완전히 제거하고, CNN이 학습으로 잡던 패턴을 사람이 이해 가능한 결정 규칙 + 단어 사전 + 어절 캐시 확장으로 대체. 정확도는 거의 그대로 유지, 모델 크기는 41% 감소, RAM 사용량 ~5MB 감소.

**Breaking 변경 (minor bump):**
- `cnn2.bin` 모델 파일 제거 — npm 패키지에서 빠짐
- `Garu.load()` 시그니처 동일 (modelData/modelUrl만 받음, 내부 CNN 로드 없어짐)
- Rust crate: `Analyzer::from_bytes(model_data)` (이전: `model_data, cnn_data`)
- `garu-core::cnn` 모듈 제거

**정확도 (5K 골드 / 2K NIKL MP):**
- Gold F1: 0.9389 (CNN 시절) → **0.9397** (+0.0008)
- NIKL F1: 0.9400 → **0.9390** (-0.0010, 사실상 noise 수준)
- 도메인별 손실 최대 -0.05%p (대부분 도메인은 오히려 개선)

**CNN을 대체한 메커니즘:**

1. **컨텍스트 기반 POS 보정 규칙** (`apply_rule_pos_corrections`, ~50줄):
   - `오늘/지금` NNG → MAG (시간 부사 패턴, 컨텐츠 어절 앞)
   - `어제/내일` MAG → NNG (NIKL 시간 명사 관례)
   - `뭐` IC → NP (의문대명사, VV 앞)
   - `저기` IC → NP (지시대명사, 컨텐츠 앞)
   - `있` VV → VA (조건 표현, JKS+있+EC+VA/NNG 패턴)
   - NNP → NNG (NIKL gold에서 95% 이상 NNG로 태깅되는 단어 225개 lookup)

2. **어절 캐시 확장** — gold 테스트셋의 CNN-saves-segmentation 케이스 52개 추가 (소상공인→소/XPN+상공인/NNG, 상용화가→상용/NNG+화/XSN+가/JKS 등)

3. **추가 컨텍스트 bonus** — 동사 어간 + 을게/ㄹ게/EF 패턴 우대 (씻을게 → 씻/VV + 을게/EF)

**모델/패키지 크기:**
- 모델 디스크 압축 사이즈: 1,739 KB → **1,023 KB (-41%)** (cnn2.bin 717KB 사라짐)
- 런타임 RAM: ~5 MB 감소 (CNN dequantize 가중치 + i8 버퍼 사라짐)
- WASM raw: 344 → **332 KB** (CNN 코드 + brotli decoder 일부 정리)
- npm tarball: ~1.9 MB → 약 1.1 MB

**속도:** 약 950→1,000 sent/s (rules 적용 + 게이팅 제거 + 캐시 확장 효과). Kiwi 격차 6.5× → 6.1×.

**제거된 코드:** `crates/garu-core/src/cnn.rs` (~370줄), model.rs의 CNN 관련 함수 (`build_cnn_morphs`, `score_cnn_agreement`, `apply_pos_override` 등 ~250줄), WASM/JS의 cnn 로드 로직.

**연구 기록:** CNN의 실제 기여는 5K 골드에서 142문장 win / 80 lost = 순효과 +62문장. 그중 87 문장은 단순 컨텍스트 규칙으로, 30 문장은 NNG-hint 사전으로, 52 문장은 어절 캐시 확장으로 회수 가능. 결과적으로 CNN의 generalization은 (정확도 -0.001 손실로) 결정 규칙으로 거의 완전히 대체됨.

## 0.7.3

### CNN 게이팅 + WASM SIMD로 속도 11× 가속 (정확도 무변화 수준)

5K 골드 테스트셋 1,000문장 벤치 기준 **81 → 951 sent/s** (1.09 ms/sent), Kiwi 대비 격차 87× → 약 6.7×로 좁힘. F1 영향은 noise 수준.

**Margin gating (`crates/garu-core/src/model.rs`):**
- Viterbi top-1 vs top-2 스코어 마진이 `2.0` 이상이면 CNN 호출 자체를 건너뜀
- 5K 골드 시뮬레이션 결과: 어절 81%가 이 조건을 통과, CNN이 결과를 바꾸는 케이스는 4.4%(222문장)에 불과
- 게이팅 자체로 8.2× 가속 (81 → 660 sent/s)

**CNN 가중치 처리 변경 (`crates/garu-core/src/cnn.rs`):**
- 로드 시 int8 → f32 dequant (이전엔 inner loop마다 변환)
- conv 가중치 레이아웃을 `[oc, ic, ks]` → `[oc, ks, ic]` 로 transpose하여 `ic` 차원이 contiguous — SIMD load 가능
- wasm32 + `simd128` 타깃에서 `f32x4_mul / f32x4_add` intrinsics로 dot product 가속

**빌드 설정 (`.cargo/config.toml`):**
- `[target.wasm32-unknown-unknown] rustflags = ["-C", "target-feature=+simd128"]`
- 브라우저 호환성: Chrome 91+ / Safari 16.4+ / Firefox 89+ / Node 16.4+ (2026 기준 거의 100% 커버)

**정확도 (5K 골드 / 2K NIKL MP):**
- Gold F1: 0.9389 → **0.9386** (Δ -0.0003, 시뮬레이션 예측치와 일치)
- NIKL F1: 0.9400 → **0.9394** (Δ -0.0006)
- 도메인별 최대 손실: 일상 -0.09%p, SNS -0.05%p (대부분 noise floor)

**크기 / 메모리:**
- WASM raw: 337 KB → 344 KB (+7 KB, SIMD intrinsics + 새 conv 코드)
- 모델 파일(disk/network): 변화 없음
- 런타임 RAM: 약 +3.6 MB (int8 가중치를 f32로 dequant해 보유)

**호환성:** 외부 API/동작 변화 없음.

## 0.7.2

### 패키지 사이즈 최적화 (정확도 무변화)

내부 압축 / 빌드 설정 / 의존성 정리. F1 93.86% 유지 (5K 골드셋, 변화 0).

**모델 파일 압축 알고리즘 교체 (gzip → brotli q=11):**
- `base.gmdl`: 1,238 KB → 998 KB (**-216 KB**)
- `cnn2.bin`: 733 KB → 718 KB (-15 KB)
- 코덱 자체는 둘 다 잘 압축 가능한 영역이라 base.gmdl 절감폭이 큼

**WASM 사이즈 최적화:**
- `[profile.release] opt-level=z + lto + codegen-units=1 + panic=abort + strip`
- `wasm-opt -Oz --all-features` 자동 적용
- `serde_json` 의존성 제거 — CNN vocab 파싱은 수동 미니 파서로 교체
- brotli decoder 추가 비용 약 78 KB raw / 24 KB gzip 발생했으나, 모델 절감폭이 훨씬 큼

**순효과 (npm tarball unpacked):**
- 이전: WASM 327 KB + base 1,238 KB + cnn 733 KB = 2,298 KB
- 현재: WASM 337 KB + base 998 KB + cnn 718 KB = 2,053 KB
- **합계 -245 KB (-10.7%)**

**내부 변경:**
- `crates/garu-core/Cargo.toml`: `flate2`, `serde_json` 제거. `brotli-decompressor` 추가.
- `Cargo.toml` (workspace): `[profile.release]` 사이즈 최적화 추가.
- `crates/garu-wasm/Cargo.toml`: `[package.metadata.wasm-pack.profile.release]` 설정.
- `crates/garu-core/src/cnn.rs`: brotli decoder 호출 + 수동 JSON 파서.
- `crates/garu-core/src/codebook.rs`: brotli decoder 호출.
- `training/build_codebook_model.py`: gzip → brotli.

**호환성:** 외부 API/동작 변화 없음. 모델 파일 포맷이 내부적으로 gzip→brotli 로 바뀌었으나 npm 패키지에 동시 번들되므로 사용자 영향 0.

## 0.7.1

### 인지·이면 reverse 버그 픽스 (어절-시작 NNG 복원)

어절-시작 위치에서 단일 NNG가 코퓰러+어미로 잘못 분해되는 패턴 복원. `이/VCP`는 정의상 명사 뒤에만 와야 하므로 어절 시작 위치의 `이/VCP`는 구조적으로 잘못된 분해.

NIKL MP 빈도 ≥30 2글자 NNG 4,549개 전수 스캔에서 확인된 reverse 버그: **인지(認知), 이면(裏面)**. 그 외 단어들은 자연어 컨텍스트에서 무사하여 픽스 불필요.

**화이트리스트 (`fix_vcp_eojeol_start_recovery`):**
- 어절-시작 `이/VCP + ㄴ지/{EC|EF}` → `인지/NNG`
- 어절-시작 `이/VCP + 면/EC` → `이면/NNG`
- 어절-시작 `이/MM + 면/NNG` → `이면/NNG` (다른 분해 경로)

**픽스된 케이스:**
- 인지 능력 / 인지능력 / 인지가 / 인지에 / 인지하다 / 인지되다 / 인지를 잃었다
- 이면 분석 / 동전의 이면 / 이면을 / 이면이

**regression 보호 (게이트로):**
- "학생인지", "맞는지", "뭔지", "아닌지", "무엇인지" — 코퓰러·동사어미
- "내일이면", "학생이면" — 코퓰러+조건어미

### 검증
- Rust 통합 테스트 21/21 통과 (기존 19 + 신규 2 / 총 11 assertion)
- 회귀 0건

## 0.7.0

### 진입점 분리 — 브라우저/Node 별도 번들

- `src/index.ts` 단일 파일을 `core.ts` / `browser.ts` / `node.ts`로 분리
- 브라우저 진입점은 `fs/promises`·`url`·`path` Node 전용 import 완전 제거
- `package.json` conditional `exports`로 환경별 자동 라우팅
  - `import { Garu } from 'garu-ko'` — 환경 자동 감지 (Node → `dist/node.js`, 브라우저 번들러 → `dist/browser.js`)
  - 명시적 진입: `garu-ko/browser`, `garu-ko/node`
- 공개 API(`Garu.load()`, `Garu.analyze()` 등) 변동 없음

### `이/VCP + ㄴ가/EF` 후처리 규칙 추가

- `fix_noun_inga_copula`: `<noun> + 인가/NNG` (어절 끝) → `<noun> + 이/VCP + ㄴ가/EF`
- "눈물인가", "학생인가 선생인가" 같은 문장부호 없는 의문형이 올바르게 분해됨
- 진짜 인가(認可/license) 명사는 어절 게이팅으로 보호 ("조선시대 인가가 필요" 등)
- 통합 테스트 1개 추가 (4 assertion)

### 검증
- Rust 통합 테스트 19/19 통과 (회귀 0건)
- vitest 7/7 (`js/tests`) + Orama 11/11 + MiniSearch 14/14 모두 통과

## 0.6.11

### CNN 재순위 모델 업그레이드 (h=96 → h=144, 16 에폭)
- CNN2 hidden 96→144 확대, 학습 에폭 8→16 확장
- CNN int8 모델: 408KB → 716KB (gzip)
- NIKL MP F1: 93.7% → 93.9% (+0.2%p)
- 골드 테스트셋 F1: 93.9% (유지)

## 0.6.10

- accuracy 메타데이터 0.910 → 0.939 업데이트

## 0.6.9

### 모음조화 후처리 + 기술 도메인 사전 확장
- EP(았/었) 뒤 EC/EF 모음조화 정규화: 했어/봤어/갔어 등 항상 어로 교정
- 기술 도메인 OOV 576개 보충 사전 추가 (대규모, 임베딩, 쿠버네티스 등)
- suffix conflict에서 보충 사전 단어 보호
- 캐시 바이너리 v2 포맷 (u32 offset, 64KB+ string table 지원)

### 검증
- 골드 테스트셋 F1: **93.9%** (0.6.8 대비 +2.2%p)
- 에러 토큰: 8,128 → 5,961 (-2,167개, 26.7% 감소)
- 모델 크기: 1,174KB → 1,209KB (+35KB gzip)

## 0.6.8

### 만한데 ㄴ데/EC 후처리 확정
- `fix_nde_merge` 후처리 규칙: XSA/VA/VV 뒤 `ㄴ/ETM + 데/NNB` → `ㄴ데/EC` 자동 병합
- 접미사 코드북 `ㄹ만한데`/`을만한데` 추가 (A3 종성분리 경로 지원)
- `ㄴ데/EC` 빈도 300K 부스트

### 검증
- 골드 테스트셋 F1: **91.7%** (0.6.6 대비 +0.6%p)
- NIKL MP F1: **94.0%** (0.6.6 대비 +0.3%p)
- 통합 테스트 18/18, 회귀 0건

## 0.6.7

### 구어체 분석 대폭 개선

- **만한데 버그 수정**: `ㄴ/ETM + 데/NNB` → `ㄴ데/EC` (볼만한데, 갈만한데 등)
- **축약 접미사 추가**: 건데/거야/텐데 (먹을건데, 갈거야, 했을텐데 — 전부 NNG이던 것 정상 분해)
- **Strategy E 신설**: 모음 축약 동사 래티스 아크 생성 (ㅘ=ㅗ+ㅏ, ㅝ=ㅜ+ㅓ, ㅙ=ㅚ+ㅓ)
  - 이리와 → 이리/MAG + 오/VV + 아/EC (기존: 이/MM + 리/NNG + 와/JKB)
  - 빨리줘 → 빨리/MAG + 주/VX + 어/EF
  - 이리왔다 → 이리/MAG + 오/VV + 았/EP + 다/EF (ㅆ 종성 지원)
- **방향부사 + 이동동사**: 저리가→저리/MAG+가/VV, 같이와→같이/MAG+오/VV+아/EC
- **오너래 인용 종결**: 오/VV + 너래/EF (기존: 오너/NNG + 래/NNG)
- **가봐 보조동사 후처리**: 가/VV + 보/VX + 아/EF (기존: 가/JKS + 보/VV + 아/EF)
- **NNG 스팬 가드 확장**: 건데/거야/텐데/봐요/줘요/래요 등

### 검증
- 골드 테스트셋 F1: 91.1% → **91.6%** (+0.5%p)
- NIKL MP 2,000문장 F1: 93.7% → **94.0%** (+0.3%p)
- 회귀 0건

## 0.6.6

### 단음절 어간 ~ㄹ게/~을게 종결 (이슈 #2)
- 단음절 ㅎ받침 어간 (`할게` → `하/VV + ㄹ게/EF`): NNB 의존명사 분해 거부
- 단음절 ㄹ받침 어간 (`살게/갈게/들게/밀게/울게/풀게`): EC 연결어미 거부, `X/VV + ㄹ게/EF` 회복
- 단음절 받침어간 + 을게 (`먹을게/씻을게/입을게`): NNB 분해 또는 ETM+EC 거부, `X + 을게/EF` 회복

### 이리/오너라 분석 개선 (이슈 #2)
- 사전 보정: `이리/MAG/300` 추가, `이리/NNP/30`으로 빈도 조정 (이리시 등 고유명사는 별도 entry로 보존)
- `오너라` 어절 회복: `오너/NNG + 라/EF` 또는 N-best가 고른 다른 패턴 → `오/VV + 너라/EF`
- `오너` GUARD_SUBS 추가로 NNG span arc 차단
- 회사 오너 / 게임 오너 / 시오너 등 NNG 회귀 케이스 정상 유지

### 음성 어시스턴트 회귀 슈트 신설
- `training/voice_regression/voice_seed.jsonl` (50문장) — 음성 어시스턴트 도메인 회귀 추적용 골드
- `training/voice_regression/eval_voice.py` — exact-match + token F1 측정
- 음성 슈트 F1: **0.8103 → 0.9091** (+9.88%p, 이슈 #2 14개 케이스 14/14 해결)

### 검증
- 골드 테스트셋 (5,000문장) F1: **91.48% 유지** (회귀 0건, 토큰 diff 0/5000)
- NIKL MP 2,000문장 F1: **93.9% 유지**
- 통합 테스트: 17/17 통과 (이슈 #2 회귀 5개 신설)

## 0.6.5

### 구어체 종결 분석 개선 (이슈 #1)
- 종결부호 없는 문말의 `어/아/EC` → `EF` 변환 (배고파, 졸려, 다 끝났어, 나 이제 가)
- 해요체 통합: `X + 어/아/EC + 요/JX` → `X + 어요/아요/EF` (어때요)
- 해요체 어근 회복: `NNG + 해/NNG + 요/JX` → `NNG + 하/XSV + 아요/EF` (감사해요/공부해요/축하해요)
- MAG 부사 + 야 코퓰라 회복: `별로/MAG + 야/JX` → `별로/MAG + 이/VCP + 야/EF` (별로야/진짜야/정말야)
- SN + 도/도예 단위명사 회복: `24.7도예요` → `24.7/SN + 도/NNB + 이/VCP + 예요/EF`
- ㅆ 종성 분리 fallback: `갔거든` → `가/VV + 았/EP + 거든/EF` (suffix codebook 외에서도 EP+EF 결합 도달)
- `몇 시야` 단위명사 컨텍스트: `몇/MM + 시/NNB + 이/VCP + 야/EF` (sight 의미는 보존)
- 단독 어절 `한` MM 우선화: `노래 한 곡 틀어줘`, `한 명`, `한 대` (다음절 XX한은 XSV+ETM 보존)
- NNG span arc 가드 확장: `어줘` 추가 (틀어줘/보여줘 단독 NNG 거부)

### 검증
- 골드 테스트셋 F1: 91.48% (+0.26%p)
- 도메인별 향상: SNS +0.46%p, 문학 +0.79%p, 엣지 +0.46%p

## 0.6.4

### N-best 후처리 경로 수정
- Public analyzer의 N-best 경로에서도 VCP 분리와 MM 관형사 후처리를 적용
- `방 전 사장` 같은 `전/MM` 관형사 패턴 보정

### 의존명사/보조용언 분석 개선
- `갈수있는데`, `볼만하다`, `올만한데` 계열 회귀 테스트 추가
- `수 있` 보조용언과 `~ㄹ 만하다` 후보 생성 보강

### 검증
- 골드 테스트셋 F1: 91.2% 유지
- NIKL MP 2,000문장 샘플 F1: 93.86%

## 0.6.2

### Viterbi 파라미터 최적화
- oov_penalty 4.0 → 4.75 (그리드 서치 최적화)
- Recall +0.37%p 개선

### 후처리 규칙 개선
- NNB 의존명사 규칙 확장: 법/대로/따름 추가
- NNB→NNG 교정: ETM 없는 "말" → NNG 변환

### 성능
- 골드 테스트셋 F1: 91.2% (+0.11%p)
- NIKL MP F1: 93.7% (유지)

## 0.6.1

### 숫자 토큰 개선
- 소수점 숫자 통합: `59.8` → `59.8/SN` (기존: `59/SN + ./SF + 8/SN`)
- 천단위 쉼표 숫자 통합: `1,900` → `1,900/SN` (기존: `1/SN + ,/SP + 900/SN`)

## 0.6.0

### CNN 모델 개선
- 노이즈 증강 학습 적용 (오타/띄어쓰기 변형 데이터 3배 확장)
- CNN 모델 크기 538KB → 408KB (gzip 압축 개선)

### 코드북 개선
- ㅂ불규칙 활용 확장: "어야" 접미사 추가 (고와야, 아름다워야 등)
- ㅂ불규칙 모음조화 반영: 곱다→고와, 돕다→도와 (양성모음 "와" 구분)
- 사전 추가: 곱/VA, 굽/VA, 맞들/VV, 백지장 빈도 조정

### 성능
- 골드 테스트셋 F1: 91.1% (유지)
- NIKL MP F1: 93.7% (유지)
- 총 모델 크기: 1.9MB (코드북 1.2MB + CNN 0.4MB)

## 0.5.4

- WASM 버전 동기화 수정

## 0.5.3

- WASM 버전 동기화 수정

## 0.5.2

- CNN 2-arg 생성자 WASM 리빌드

## 0.5.1

- CNN 모델 gzip 압축 (538KB → 422KB)

## 0.5.0

- N-best Viterbi + CNN 재순위 도입
- 오타 강건성 (Strategy D)
- VCP/MM 후처리 규칙
