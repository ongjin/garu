# Garu (가루) — 프로젝트 컨텍스트

## 프로젝트 개요

브라우저에서 실행되는 초경량 한국어 형태소 분석기. 코드북 + N-best Viterbi + 어절 캐시 + 후처리 규칙 + 재순위 perceptron으로 동작 (CNN 폐기).
- **F1 96.0%** (9,000문장 v15k 골드 테스트셋, ep_norm 정규화) / 2025 구어 held-out 91.2%
- **모델 1.2 MB** (brotli q=11 압축, npm 패키지에 포함, CDN 불필요. 재순위 가중치 Section 14 +200KB 포함)
- **WASM** — 브라우저에서 실행 (raw 412KB / gzip 176KB, opt-level=3 + wasm-opt -O3 + brotli decoder. viterbi_nbest 최적화가 재순위 비용을 상쇄해 Kiwi 대비 격차 2.4×→2.07×)

### Purpose (영문 원문)

Garu is a browser-first, lightweight Korean morphological analyzer. The core research goal is to keep the analyzer practical for client-side use while approaching server-grade analyzers in accuracy.

Primary constraints:

- Runs fully offline in the browser through WASM.
- Ships model files inside the npm package; do not require CDN or server calls.
- Keep inference cheap: prefer lookup tables, Viterbi, compact rules, caches, and small quantized models.
- Treat model size, latency, and F1 as a single tradeoff. Accuracy gains that make browser delivery impractical are not acceptable by default.

## CRITICAL RULES

### 규칙 (한국어)

- 커밋 메시지에 AI/Claude 관련 내용 포함 금지
- git email: dydwls140@naver.com
- 설계/계획 문서를 repo에 올리지 않음
- push / 배포 / GitHub Release 생성은 사용자 허락 필수 (커밋은 자유)
- `star-history` 브랜치는 산출물 — `.github/workflows/star-history.yml` 이 README 의 Star History SVG 를 매주 재생성해 force-push 한다(2026-06-30 GitHub stargazers API 가 소유자 전용이 되어 star-history.com 임베드가 깨진 것의 자가 호스팅 대체). 직접 커밋 금지, 차트 수정은 `.github/scripts/star-history.mjs`.

### Local Rules (영문 원문)

- Do not put AI/Claude-related text in commit messages.
- Use git email `dydwls140@naver.com` when committing.
- Do not add speculative design or planning documents to the repo unless the user explicitly asks for that artifact.
- Keep changes narrowly scoped to the current research question.
- Do not overwrite untracked model artifacts or generated files unless the task requires regenerating them.
- If metrics or file sizes are mentioned in docs, verify whether they reflect the current artifacts before treating them as current facts.

## Source Priority

- `AGENTS.md` is the current operational snapshot for architecture, file map, commands, and local rules.
- `docs/paper.md` is the research narrative: why earlier approaches failed, how each optimization helped, and what open research directions remain.
- Code, model artifacts, and fresh benchmark output override both documents when they disagree.
- If reporting metrics externally, rerun or cite the exact benchmark command and dataset split. The paper and `AGENTS.md` contain different historical metric snapshots.

## 아키텍처

### 파이프라인 (현행 — 재순위 perceptron, CNN 은 크기 대비 이득 부족으로 폐기: +700KB 에 +0.2%p)

```
입력 텍스트 → 전체 문장 래티스 구축 (캐시 항목을 저비용 아크로 주입, 오타 교정 아크 생성)
           → 문장 수준 Trigram N-best Viterbi (top-10)
           → 후처리 (VCP 분리, MM 관형사 교정, POS 보정 등 fix_*)
           → 재순위 perceptron (Section 14, 확신 마진 τ=4 넘을 때만 교체)
           → 출력
```

### 핵심 Rust 코드
- `crates/garu-core/src/codebook.rs` — 래티스 구축, Viterbi 디코딩, 어절 캐시, 후처리 규칙(`fix_*`)
- `crates/garu-core/src/model.rs` — Analyzer (N-best Viterbi + POS override + 재순위 게이트). POS 보정 규칙은 폐기된 CNN의 골드 행동을 distill한 것 (추론은 안 함)
- `crates/garu-core/src/rerank.rs` — 재순위 perceptron (FNV-1a feature hashing — `training/rerank/features.py`와 바이트 단위 동일 필수)
- `crates/garu-core/src/trie.rs` — FST 사전 (다중 POS: u64에 2개 POS pack)
- `crates/garu-core/src/types.rs` — 42개 세종 POS 태그 enum
- `crates/garu-wasm/src/lib.rs` — WASM 바인딩
- `crates/garu-tools/src/build_dict.rs` — FST 빌더 (다중 POS 지원)

### JS/TS (npm 패키지)
- `js/src/core.ts` — `GaruBase` 클래스 (`analyze` / `tokenize` / `nouns` / `modelInfo` / 런타임 사용자 사전 `addUserWord`·`addUserWords`·`clearUserWords`·`userWordCount`)
- `js/src/{browser,node}.ts` — `Garu extends GaruBase` + static `load()` (browser=WASM fetch / node=fs)
- `js/src/normalize.ts` — `normalizeText` / `splitSentences`
- `js/models/base.gmdl` — 번들된 모델
- `js/pkg/` — wasm-pack 빌드 출력

모델 포맷(GMDL 섹션 구성)·학습 파이프라인은 [docs/wiki/model-build.md](docs/wiki/model-build.md) 참조.

## 빌드·명령

### 자주 쓰는 명령 (발췌 — 전문은 [docs/wiki/benchmarks.md](docs/wiki/benchmarks.md))

```bash
# 모델 리빌드
python3 training/build_codebook_model.py

# Rust 테스트
cargo test

# WASM 빌드
wasm-pack build crates/garu-wasm --target web --out-dir ../../js/pkg

# 골드 F1 평가 (garu만, n=9000 v15k, ep_norm)
(cd training/gold_testset && python3 eval_f1.py --analyzers garu)

# 2025 구어 held-out 평가 (SX 16.4K문장, 2025→2021 역변환 골드, garu F1 0.910. 학습 사용 금지)
python3 training/eval_nikl2025_guueh.py

# 단일 문장 분석 (디버깅): GARU_MODEL 지정 + analyze_batch 예제
GARU_MODEL=js/models/base.gmdl cargo run -q --release --example analyze_batch <입력파일>
```

## Research Lessons

Do not restart from approaches already shown to be poor fits unless the goal is explicitly to reproduce or disprove them:

- BiLSTM distillation failed because matrix-heavy inference is not browser-friendly.
- Jamo-level sequence labeling failed because it lengthened sequences and destroyed useful syllable-level signals.
- Ambiguity-table expansion hurt accuracy by adding too many noisy candidates.
- Self-learning frequency loops did not converge reliably.
- Perceptron POS reranking produced only tiny gains because many remaining errors are segmentation errors.
- Decomposition preference penalties increased over-segmentation.
- Cost-scale normalization did not improve over tuned parameters.

Strong historical wins:

- Switching from Kiwi-derived output to NIKL gold-derived codebook patterns was the largest improvement.
- Lowering `morpheme_penalty` made multi-morpheme suffix patterns usable.
- Removing noisy Wikipedia NNP entries improved both size and accuracy.
- Multi-POS FST, sparse trigram quantization, and word-bigram costs gave small but clean gains.
- Smart eojeol cache broke the codebook-only ceiling by caching high-correction-value words, not merely frequent words.
- Contextual postprocessing rules gave free accuracy without growing the model.
- Sentence-level Viterbi allows cached analyses to be overridden by stronger sentence context.
- N-best Viterbi plus the reranking perceptron (Section 14, swap only above confidence margin τ=4) is the main current path for better ambiguity handling. CNN agreement scoring was retired because +0.2%p did not justify +700KB of model size — see docs/wiki/research-history.md.

## Research Priorities

Prefer work that targets known residual error classes:

- Segmentation errors: the dominant remaining error source. Investigate Viterbi top-N candidates, reranking-perceptron features, and candidate generation changes that can actually change boundaries.
- POS ambiguity: continue using sentence context, word-bigram rules, reranker confidence margin, and targeted postprocessing.
- OOV and neologisms: explore syllable-pattern POS inference without adding neural inference cost (no CNN — retired: +0.2%p did not justify +700KB).
- Typos and spacing noise: improve typo arcs, noisy training, and pre/postprocessing while measuring clean-text regressions.
- Domain robustness: compare same-domain, split, and cross-domain NIKL results. Do not optimize only one split.
- Model-size discipline: record size deltas for every model or dictionary change.

## Experiment Protocol

Before changing the analyzer, identify:

- The exact error type being targeted.
- Baseline F1 and model size.
- The expected mechanism of improvement.
- The rollback criterion if accuracy, size, or speed regresses.

After changing the analyzer, check at least the focused failure examples and one aggregate benchmark. For broad model changes, compare:

- Gold testset F1.
- NIKL MP F1 when the dataset is available.
- Model gzip size.
- WASM/package impact if the change touches public delivery.

When analyzing dataset errors, preserve exact sentence references and describe whether the issue is segmentation, POS tagging, OOV handling, spacing, or punctuation classification.

## 추가 문서 (docs/wiki/)

위의 AGENTS.md 본문에는 매 세션 필요한 공통 컨텍스트만 둔다. 특정 작업 들어갈 때 아래 문서를 직접 읽어와서 참고할 것.

- [docs/wiki/release.md](docs/wiki/release.md) — **npm 배포 풀세트**. Cargo→wasm-pack→tsc→CHANGELOG→npm version→commit/push→publish→gh release 8단계 + 통합 패키지(orama/minisearch) 동기화. X.X.X 배포하거나 통합 패키지 sync할 때.
- [docs/wiki/model-build.md](docs/wiki/model-build.md) — **모델/학습 빌드**. GMDL v3 섹션 구성(사전/코드북/트라이그램/캐시), build_codebook_model.py의 캐시 보존 동작, 학습 파이프라인 스크립트. 모델·사전·코드북 리빌드하거나 학습 스크립트 만질 때.
- [docs/wiki/analyzer-internals.md](docs/wiki/analyzer-internals.md) — **codebook.rs 동작 지도**. 아크 출처(사전/코드북/캐시/재구성 전략 A2b·A2c·E·A3·A4·D·B·C), viterbi vs nbest 불일치, 후처리 fix_* 체인이 analyze와 analyze_topn에서 다름, 디버깅 방법. "[분석 오류]" 이슈 디버깅하거나 후처리 규칙 추가할 때.
- [docs/wiki/morphology-conventions.md](docs/wiki/morphology-conventions.md) — **분석 정답 기준**. 표준국어대사전 우선(골드/Kiwi 맹신 금지), 높임 -시- 분리 원칙(기본형↔보충법, 드시=들+시), ㅂ불규칙 모음조화, 으시 OOV, ep_norm. 분석 정답이 헷갈리거나 골드 라벨 바꿀 때.
- [docs/wiki/research-history.md](docs/wiki/research-history.md) — **연구 이력 54항목 + 폐기된 CNN**. 무엇을 왜 채택/폐기했는지, 같은 실패 반복 방지. 과거 맥락이 필요하거나 폐기 접근(CNN·perceptron) 재도입 검토 시.
- [docs/wiki/benchmarks.md](docs/wiki/benchmarks.md) — **빌드·평가 명령 전문**. NIKL MP 2021/2025 벤치마크, `--norm-2025` 정규화 범위와 천장 측정, 골드·구어 held-out 평가. 벤치마크를 돌리거나 2025판 점수를 해석할 때.
