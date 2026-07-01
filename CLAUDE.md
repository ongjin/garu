# Garu (가루) — 프로젝트 컨텍스트

## 프로젝트 개요

브라우저에서 실행되는 초경량 한국어 형태소 분석기. 코드북 + N-best Viterbi + 어절 캐시 + 후처리 규칙로 동작 (CNN 폐기).
- **F1 95.1%** (9,000문장 v15k 골드 테스트셋, ep_norm 정규화) / NIKL MP 93.7%
- **모델 1.0 MB** (brotli q=11 압축, npm 패키지에 포함, CDN 불필요)
- **WASM** — 브라우저에서 실행 (raw 337KB / gzip 155KB, opt-level=z + wasm-opt -Oz + brotli decoder)

## 아키텍처

```
입력 텍스트 → 전체 문장 래티스 구축 (캐시 항목을 저비용 아크로 주입, 오타 교정 아크 생성)
           → 문장 수준 Trigram N-best Viterbi (top-5)
           → 후처리 (VCP 분리, MM 관형사 교정, POS 보정 등 fix_*)
           → 출력
```

### 핵심 Rust 코드
- `crates/garu-core/src/codebook.rs` — 래티스 구축, Viterbi 디코딩, 어절 캐시, 후처리 규칙(`fix_*`)
- `crates/garu-core/src/model.rs` — Analyzer (N-best Viterbi + POS override). POS 보정 규칙은 폐기된 CNN의 골드 행동을 distill한 것 (추론은 안 함)
- `crates/garu-core/src/trie.rs` — FST 사전 (다중 POS: u64에 2개 POS pack)
- `crates/garu-core/src/types.rs` — 42개 세종 POS 태그 enum
- `crates/garu-wasm/src/lib.rs` — WASM 바인딩
- `crates/garu-tools/src/build_dict.rs` — FST 빌더 (다중 POS 지원)

### JS/TS (npm 패키지)
- `js/src/core.ts` — `GaruBase` 클래스 (`analyze` / `tokenize` / `nouns` / `modelInfo`)
- `js/src/{browser,node}.ts` — `Garu extends GaruBase` + static `load()` (browser=WASM fetch / node=fs)
- `js/src/normalize.ts` — `normalizeText` / `splitSentences`
- `js/models/base.gmdl` — 번들된 모델
- `js/pkg/` — wasm-pack 빌드 출력

모델 포맷(GMDL 섹션 구성)·학습 파이프라인은 [docs/claude/model-build.md](docs/claude/model-build.md) 참조.

## 빌드

```bash
# 모델 리빌드
python3 training/build_codebook_model.py

# Rust 테스트
cargo test

# WASM 빌드
wasm-pack build crates/garu-wasm --target web --out-dir ../../js/pkg

# 골드 F1 평가 (garu만, n=9000 v15k, ep_norm)
(cd training/gold_testset && python3 eval_f1.py --analyzers garu)

# 벤치마크 (NIKL MP 데이터: 기본 ~/workspace/data/nikl_mp_2021/. *.json glob)
#   kkma가 JVM 크래시로 스크립트 중단 → --analyzers garu,kiwi 권장
python3 training/eval_nikl_mp.py --n 2000 --analyzers garu,kiwi   # garu F1 0.937
# 다른 코퍼스는 NIKL_MP_DIR로 override. 단 2025판은 분절 컨벤션이 거칠어져
#   (명사+하 병합, _복합어 결합) raw F1 비교불가 → --norm-2025 정규화 필요
#   (XSV/XSA·적/XSN 병합, _un-join, 인용 EF+고·며·는 병합, 직접인용 JKQ canonical
#    까지 구현. norm 후 2021 garu 0.9354 불변 / 2025 garu 0.8741. 단위테스트
#    training/test_nikl_norm_2025.py. 되/하 VX↔VV·구어 그/IC는 진짜 차이라 미정규화)
NIKL_MP_DIR=~/workspace/data/nikl_mp_2025 python3 training/eval_nikl_mp.py --n 2000 --analyzers garu,kiwi --norm-2025

# 단일 문장 분석 (디버깅): GARU_MODEL 지정 + analyze_batch 예제
GARU_MODEL=js/models/base.gmdl cargo run -q --release --example analyze_batch <입력파일>
```

## 규칙

- 커밋 메시지에 AI/Claude 관련 내용 포함 금지
- git email: dydwls140@naver.com
- 설계/계획 문서를 repo에 올리지 않음
- push / 배포 / GitHub Release 생성은 사용자 허락 필수 (커밋은 자유)

## 추가 문서 (docs/claude/)

위의 CLAUDE.md 본문에는 매 세션 필요한 공통 컨텍스트만 둔다. 특정 작업 들어갈 때 아래 문서를 직접 읽어와서 참고할 것.

**문서화 규칙** (다음 세션이 lean한 CLAUDE.md를 유지하도록):
- **자동 갱신 (지시 없이)**: 코드·수치·아키텍처·빌드 절차·규칙을 바꿔서 기존 문서(CLAUDE.md·docs/claude/·README·paper 등)가 부정확해지면, "문서 고쳐줘"라는 별도 지시가 없어도 같은 작업 안에서 해당 문서를 함께 갱신한다. 매 작업 끝에 "이번 변경이 문서화된 사실을 무효화했나?"를 점검할 것. (단, push/배포 규칙은 그대로 — 커밋은 자유.)
- **CLAUDE.md는 lean 유지**. 매 세션 자동 로드되므로 high-signal만 — 개요, 아키텍처, 빌드, 규칙. 그 외는 `docs/claude/`로.
- **분할 기준**: 한 주제로 30줄 넘게 쌓이는데 그 내용이 *특정 작업 시에만* 필요하면 `docs/claude/<topic>.md`로 옮긴다. 본문에서 그 섹션을 제거하고 아래 인덱스에 한 줄만 추가.
- **각 doc 첫 줄은 `> **언제 읽나**: ...` blockquote로 trigger 명시**. 인덱스 hook 보고 doc을 열었을 때 첫 줄만으로 자기 작업에 맞는지 즉시 판단 가능하도록.
- **기존 doc 갱신**: 가능하면 같은 doc 안에서 끝낸다. 인덱스 hook이 더 이상 정확하지 않으면 그 hook도 같이 고친다.

- [docs/claude/release.md](docs/claude/release.md) — **npm 배포 풀세트**. Cargo→wasm-pack→tsc→CHANGELOG→npm version→commit/push→publish→gh release 8단계 + 통합 패키지(orama/minisearch) 동기화. X.X.X 배포하거나 통합 패키지 sync할 때.
- [docs/claude/model-build.md](docs/claude/model-build.md) — **모델/학습 빌드**. GMDL v3 섹션 구성(사전/코드북/트라이그램/캐시), build_codebook_model.py의 캐시 보존 동작, 학습 파이프라인 스크립트. 모델·사전·코드북 리빌드하거나 학습 스크립트 만질 때.
- [docs/claude/analyzer-internals.md](docs/claude/analyzer-internals.md) — **codebook.rs 동작 지도**. 아크 출처(사전/코드북/캐시/재구성 전략 A2b·A2c·E·A3·A4·D·B·C), viterbi vs nbest 불일치, 후처리 fix_* 체인이 analyze와 analyze_topn에서 다름, 디버깅 방법. "[분석 오류]" 이슈 디버깅하거나 후처리 규칙 추가할 때.
- [docs/claude/morphology-conventions.md](docs/claude/morphology-conventions.md) — **분석 정답 기준**. 표준국어대사전 우선(골드/Kiwi 맹신 금지), 높임 -시- 분리 원칙(기본형↔보충법, 드시=들+시), ㅂ불규칙 모음조화, 으시 OOV, ep_norm. 분석 정답이 헷갈리거나 골드 라벨 바꿀 때.
- [docs/claude/research-history.md](docs/claude/research-history.md) — **연구 이력 30항목 + 폐기된 CNN**. 무엇을 왜 채택/폐기했는지, 같은 실패 반복 방지. 과거 맥락이 필요하거나 폐기 접근(CNN·perceptron) 재도입 검토 시.
