# Garu (가루) — 프로젝트 컨텍스트

## 프로젝트 개요

브라우저에서 실행되는 초경량 한국어 형태소 분석기. 코드북 + N-best Viterbi + 어절 캐시 + 후처리 규칙 + 재순위 perceptron + 구어 입력용 탐색 비용 보정으로 동작 (CNN 폐기).
- **F1 96.176%** (9,000문장 v15k 골드 테스트셋, ep_norm 정규화) / 2025 구어 고정 평가 92.321% / 2025 문어 held-out 86.426% — 2026-11-11 실측, 앞의 둘은 연구 선택에 노출된 고정셋. 0.13.0 배포본
- **기본 모델 1,344,225 B** (brotli q=11, npm 패키지에 포함, CDN 불필요. 재순위 Section14, 문맥 POS 교정 Section16·17, 구어 탐색 비용 Section18 포함)
- **WASM** — raw 447,575 B / `gzip -9` 190,669 B (opt-level=3 + wasm-opt -O3 + brotli decoder). 현재 실행 비용 비교는 연구 이력 #70·#78·#79 참조.
- **선택형 무공백 처리** — `garu-ko/text`의 `restoreSpacing`·`splitUnspaced`, `garu-ko/sentences`의 인용문 보존·Q4 문맥 검증. `splitSentences(text)`는 분석기 없이 실행하는 빠른 기본 호출이며, `(text, garu)`는 기존 형태소 검증 경로다. 줄바꿈은 항상 문장 경계다. 기본 분석 경로에는 자동 적용하지 않음. 속도·정확도 범위는 연구 이력 #71·#73·#74·#83 참조.

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
- Record rejected research findings in `docs/wiki/research-history.md`, then remove its temporary workspace, models, dumps and checkpoints. Keep original corpora, reusable annotated data, and adopted code/models in their source-of-truth locations. The user-requested archive cleanup is recorded in `~/workspace/data/cleanup.json`; `~/workspace/data/README.md` identifies the current corpora, including `korean_pos/`. The old `garu_*` research folders and linked worktrees were removed; their historical paths are not runnable instructions.
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
           → 구어로 판별된 입력이면 아크 비용 보정 (Section18, 문자 n-gram 판별기 + 학습된 선형 비용)
           → 문장 수준 Trigram N-best Viterbi (top-10)
           → 후처리 (VCP 분리, MM 관형사 교정, POS 보정 등 fix_*)
           → 재순위 perceptron (Section 14, 확신 마진 τ=4 넘을 때만 교체)
           → 학습 문맥 조건에 따른 최종 POS 교정 (Section16 캐시 일치 → Section17 최종 출력열 일치, 형태·경계 유지)
           → 출력
```

### 핵심 Rust 코드
- `crates/garu-core/src/codebook.rs` — 래티스 구축, Viterbi 디코딩, 어절 캐시, 후처리 규칙(`fix_*`)
- `crates/garu-core/src/model.rs` — Analyzer (N-best Viterbi + POS override + 재순위 게이트). POS 보정 규칙은 폐기된 CNN의 골드 행동을 distill한 것 (추론은 안 함)
- `crates/garu-core/src/rerank.rs` — 재순위 perceptron (FNV-1a feature hashing — `training/rerank/features.py`와 바이트 단위 동일 필수)
- `crates/garu-core/src/search.rs` — Section 18 아크 feature·탐색 비용·구어 판별기 (판별기 해시는 `training/search/router.py`와 동일 필수, 학습은 `examples/search_train.rs`)
- `crates/garu-core/src/trie.rs` — FST 사전 (다중 POS: u64에 2개 POS pack)
- `crates/garu-core/src/types.rs` — 42개 세종 POS 태그 enum
- `crates/garu-wasm/src/lib.rs` — WASM 바인딩
- `crates/garu-tools/src/build_dict.rs` — FST 빌더 (다중 POS 지원)

### JS/TS (npm 패키지)
- `js/src/core.ts` — `GaruBase` 클래스 (`analyze` / `tokenize` / `nouns` / `modelInfo` / 런타임 사용자 사전 `addUserWord`·`addUserWords`·`clearUserWords`·`userWordCount`)
- `js/src/{browser,node}.ts` — `Garu extends GaruBase` + static `load()` (browser=WASM fetch / node=fs)
- `js/src/normalize.ts` — `normalizeText` / `splitSentences`
- `js/src/text.ts`, `boundaries.ts`, `boundary-data.ts` — 선택형 공백 복원·무공백 문장 분리, int8 문자 경계 모델
- `js/src/sentences.ts`, `sentence-verifier.ts`, `sentence-{fast-,}data.ts` — 선택형 바깥 문장 분리. 기본은 작은 문자 문맥 Q4, 분석기 전달 시 형태소 검증
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

# 2025 구어 held-out 평가 (SX 16.4K문장, 2025→2021 역변환 골드, garu F1 0.923. 학습 사용 금지)
python3 training/eval_nikl2025_guueh.py

# 채택 게이트 v2 (gold·2025 SX·2025 NX paired bootstrap, 규칙은 benchmarks.md)
python3 training/gate/score.py predict --model <cand.gmdl> --out <dir> && python3 training/gate/gate.py <base_dir> <dir>

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
- Score-calibrated MBR on the existing candidates and a document-local noun lexicon inferred from first-pass repetitions were not adopted. MBR regressed fixed F1; the lexicon propagated segmentation/POS mismatches and added two-pass cost. Keep synthetic name demonstrations separate from corpus evidence, and do not repeat these same recipes without a new hypothesis. See research-history #62.
- A separate direct CNN + contextual model was tested without Garu candidates or fallback. Its complete Q4 file fit 3MB and improved fixed spoken F1, but broad gold and nickname regressions prevented a default replacement. It was a Python prototype unsupported by current GMDL/WASM; the discarded artifacts were removed. See research-history #63. The old BIO preparation loses morphemes on contractions, so audit target round-trip before reusing it.
- Search-time linear cost learning and learned morphology edits were tested separately. Search learning improved spoken F1 with a small sidecar and no clear native slowdown in the measured sample, but domain regressions remained. Edit rules also showed reference-convention conflicts. Neither was adopted; experimental artifacts were removed. A fixed top-k oracle does not bound every possible learned search cost; see research-history #64 for the corrected scope and measured reachability limits.
- Adding validated KLUE-DP, UD Korean-Kaist and UD Korean-KSL training sentences to the existing perceptron slightly improved external source evaluations but regressed fixed gold/spoken against the shipped model. That model was rejected and removed; the reusable original and normalized data are in `~/workspace/data/korean_pos/`. The experiment changed Section 14 only. UD lemmas may omit endings despite multi-morpheme XPOS: use an explicit OrigLemma when available and exclude unsupported targets. Native KAIST xsa is an adverb suffix, not Sejong XSA. See research-history #65; this did not test adding the data to the 5M contextual model or rebuilding candidates.

- Bulk MeCab-ko-dic, 한국어기초사전 and 우리말샘 imports were tested. Adding every compatible entry regressed fixed gold/spoken and inflated the model; only 1,449 reviewed NNG entries were adopted in Section 6. Source noun labels and raw occurrence counts are not contextual POS gold. Preserve the 55 surrounding-token guards, per-word provenance and source licenses; see research-history #69 and `js/models/NOTICE.md`.

- Contextual POS alternatives are applied after reranking only when the original cached form/POS/span matches and no user-word override exists. Injecting them as extra lattice arcs regressed surrounding analyses. Dump Section14 training candidates through `analyze_topn_for_rerank`, before this final correction. See research-history #70.
- Unspaced text restoration is an opt-in JS module. Boundary F1 is not morphological F1; preserving characters does not mean all spacing is correct. Conservative mode favors fewer false insertions, and quotations/ambiguous text remain difficult. See research-history #71.
- Internal spoken-boundary relabeling was tested with SF-backed proposals, uncertain-EF masking and terminal-deletion augmentation. It did not establish a gain over a threshold-only control, so the new weights and threshold were rejected. EF alone includes reported speech and postposed clauses; keep reviewed training references separate from held-out references. See research-history #72 and `training/boundaries/review_labels_20261020.json`.
- The optional Q4 sentence verifier combines candidate filtering, matched hard negatives and character/morphology context. Its quote-preserving API reaches95.73/96.30% boundary F1 on inspected2025NX whitespace/terminal-deletion variants; quote protection alone accounts for most of the gain. Original2021 references sometimes split within quotes and regress under this policy. Do not call this universal95% accuracy or a pure neural gain. PyTorch2.8 MPS locally ignores embedding padding_idx in backward: retain the explicit padding mask and export guard. See research-history #73.
- The no-analyzer sentence default uses a separately trained C24/E8/H32 verifier and unrolled, bit-equivalent FNV scoring. Fixed2025NX F1 is95.91/96.28%, loaded sample mean20µs in Node and18µs in Chrome on M4; Node P95 is61µs. Do not conflate loaded means with all-input or cold-start guarantees. The analyzer-supplied path retains its full model. See research-history #74.
- Extending final one-POS corrections beyond exact cache matches improved aggregate fixed F1 but regressed literature, so the tested rules were not adopted. Distinguish absent cache surfaces from cached surfaces with different final analyses. Isolated evaluation also needs the ignored `content_dict.txt` used by the 2025 reference converter; matching predictions alone does not ensure matching references. See research-history #75.
- Joint selection among spacing alternatives improved morphological F1 on synthetic whitespace-deleted inputs, but regressed normal text and some domains while increasing loaded Node/WASM cost. It also lowered spacing-boundary F1 because it often retained the unspaced input. Do not describe it as better spacing restoration or enable it automatically. See research-history #76.

- Adoption now uses gate v2 (`training/gate/`, rules in `docs/wiki/benchmarks.md`): paired bootstrap over gold, 2025 SX and a new 2025 NX held-out. Seed-only reranker retrains already produce domain point regressions up to 0.12pp, so do not reject on small-domain point estimates. A spoken-only reranker and register-specific margins gave no spoken gain even with oracle routing (#77). A re-implementation of #75 passed gate v2 on all three sets and ships as Section 17; it is not byte-identical to the deleted #75 rules (#78). A #64 re-implementation (Section 18 search costs) fails gate v2 when applied everywhere (significant gold tech/literature regressions, partly NIKL XSN-splitting conventions) but passes when gated by a spoken router (+0.80pp 2025 SX); it ships at +80KB model and +7% native time on spoken text, with the unspaced `갈수있는데` regression accepted and its test ignored (#79). Retrain Section 14/17/18 candidates together when any of them changes.

- Residual re-diagnosis at 0.12.1 (#82) and its follow-up (#83, applied on main 2026-11-11). The sentence head and verifier ignore spaces; with normal spacing 0.12.1 scored 98.4% boundary F1 on fixed 2025 NX but 56–59% on NIKL online-corpus Instagram paragraphs. `splitSentences` now treats every line break as a boundary. Those Instagram sentence labels are punctuation-driven: train only on positions after terminal punctuation (weight 0 elsewhere) or the head learns not to split unpunctuated news. Trailing-ornament snapping and dropping word-internal boundaries in spaced input fixed most placement errors. Reviewed unpunctuated conversational boundaries remain at 47.8% F1 (`~/workspace/data/korean_text_boundaries/20261111-unpunct-review/`). Garu emitted 그/IC only for the 2021 transcription `그~`; strip tildes from 2021 SX when developing spoken changes (`training/ambiguity/spoken_geu.py` appends the resulting Section 17 rule; re-run it after re-mining Section 17). A local-context EC/EF re-decision is closed. Online-corpus document buckets (`sha256(id)[:8] mod 10`) 0–1 are the final test, 2–3 dev, 4–9 train.

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
- Optional contextual model research is allowed under the user's revised budget: self-trained roughly 5M parameters, a complete additional 4-bit model file within 3 MB, fully offline, and selectable separately from the base analyzer. Extended training and masked-character pretraining fit the size budget but retained name/domain regressions; neither was integrated and their discarded artifacts were removed. The single-seed pretraining comparison reused existing raw training texts and gave a small additional spoken-F1 gain with nearly unchanged gold F1. See `docs/wiki/research-history.md`. Keep data lineage, seed coverage, candidate availability and baseline-versus-added gains explicit. These experiments do not establish a model-capacity limit.

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
- [docs/wiki/research-history.md](docs/wiki/research-history.md) — **연구 이력과 폐기된 CNN**. 무엇을 왜 채택/폐기했는지, 같은 실패 반복 방지. 과거 맥락이 필요하거나 폐기 접근(CNN·perceptron) 재도입 검토 시.
- [docs/wiki/benchmarks.md](docs/wiki/benchmarks.md) — **빌드·평가 명령 전문**. NIKL MP 2021/2025 벤치마크, `--norm-2025` 정규화 범위와 천장 측정, 골드·구어 held-out 평가. 벤치마크를 돌리거나 2025판 점수를 해석할 때.
