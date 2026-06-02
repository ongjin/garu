> **언제 읽나**: 과거에 무엇을 시도했고 왜 채택/폐기했는지 맥락이 필요할 때, 또는 폐기된 접근(특히 CNN·perceptron)을 재도입하려 할 때. 같은 실패를 반복하지 않기 위한 기록.

# 연구 이력

1. **BiLSTM 지식 증류** → 실패 (행렬 연산이 WASM에서 비실용적)
2. **자소 시퀀스 라벨링** → 실패 (학습 너무 느림, 음절 의미 파괴)
3. **코드북 + Viterbi** → 채택, 76.1% F1
4. **NIKL 골드 데이터 학습** → +6%p (가장 큰 브레이크스루)
5. **파라미터 튜닝** (morpheme_penalty 3.0→0.25) → +2%p
6. **Wiki NNP 제거** → +0.4%p, 모델 크기 절반
7. **Multi-POS FST** → +0.3%p (있→VA/VX, 하→VV/XSV)
8. **Sparse trigram u8** → -270KB, 정밀도 손실 0
9. **스마트 어절 캐시** → +3%p (Viterbi 오답 10K 어절만 캐시, 328KB)
10. **문맥 기반 후처리 규칙** → +1.3%p (VX/JC/JKC/NNB/XSN/MM/XSV 교정, 0KB)
11. **문장 수준 Viterbi** → +0.2%p (캐시를 래티스 아크로 주입, 동형이의어 해소, 0KB)
12. **종성 분리 전략 (A3)** → 코드북에 없는 활용형 처리 (고친다→고치+ㄴ다, 0KB)
13. **모음 축약 복원 (A2b)** → 명령형 어미 교정 (건너라→건너+어라, 0KB)
14. **모델 gzip 압축** → 2.2MB→1.2MB (46% 절감, flate2 rust_backend)
15. **2-layer 1D CNN 재순위** → int8 526KB, 신뢰도 기반 POS 보정 (NP↔VV, XSV↔XSA 등) *(이후 폐기 — 아래 참조)*
16. **Word bigram 동형이의어 해소** → "나는" BOS→NP 보너스 강화
17. **N-best Viterbi + CNN 재순위** → top-5 후보 생성, CNN agreement 스코어링으로 최적 선택 (분절 교체 가능, 0KB) *(CNN은 이후 폐기, N-best Viterbi는 유지)*
18. **오타 강건성 (Strategy D)** → 경음화(ㅆ↔ㅅ), 모음혼동(ㅐ↔ㅔ) 등 11개 자모 규칙, OOV 위치에서 래티스 아크 주입 (0KB)
19. **VCP 후처리 규칙** → 이다/이고/이며/이라 등 계사 분리 (0KB)
20. **MM 관형사 후처리** → 전/그런/이런/저런/어떤/새/헌/옛/온 + 명사 → MM 교정 (0KB)
21. **CNN 노이즈 증강 학습** → 한글 오타+띄어쓰기 변형 데이터 3배 확장, val acc 96.95%→97.51% (CNN 408KB) *(이후 폐기)*
22. **ㅂ불규칙 활용 확장** → "어야" 접미사 추가, 곱다/돕다 모음조화 "와" 구분 (고와야, 도와야)
23. **WASM 사이즈 최적화** → [profile.release] opt-level=z + lto + codegen-units=1 + panic=abort + strip + wasm-opt -Oz (327KB→266KB raw, -19%)
24. **serde_json 제거** → cnn2 vocab 파싱을 수동 미니 파서로 교체, 의존성 제거 (raw -30KB)
25. **gzip → brotli q=11 압축** → base.gmdl 1238→1022KB (-216KB), cnn2.bin 733→718KB (-15KB). WASM에 brotli-decompressor 추가로 +78KB. 순절감 -196KB unpacked (-9%).
26. **eval_f1.py ep_norm 정합화** → jamo/모음조화/EP축약/태그 정규화의 측정 아티팩트 제거. norm 적용 시 overall +2.06pp (양쪽 분석기 모두 게인). 정규화 후 실제 Kiwi가 5/6 도메인 우위라는 사실 드러남.
27. **자모 정규화 옵트인** → `normalizeJamo: bool` 옵션 (기본 false). gold v15k가 호환/결합 자모 67:33 혼재라 `project_guuh_weakness.md` 양방향 검증 규칙 적용 → 기본값 false 유지.
28. **`~/SO` 캐시 자동 보강** → NIKL annotation 누락된 35개 trailing-tilde 캐시 항목에 SO morpheme 추가. 구어 `~/SO` 인식 9.6% → 100% (~0.34pp F1).
29. **in-place 캐시 패칭 도입** → `build_eojeol_cache.py` 전체 리빌드 대신 `eojeol_cache.bin`을 직접 파싱/수정/재기록. 옛 curated cache의 hand-tuned 가치를 보존 (full rebuild는 -2pp 회귀 위험).
30. **Phase 1: Averaged Structured Perceptron 시도 → 폐기** (2026-05-21). 8 feature(POS trigram/lex bigram/jongseong/last syl/surface trigram/morph len/cache hit/sent position) + Python POC + Rust integration. 시뮬레이션 F1 (dev 800) +0.85pp 보였으나 Rust 실측(전체 8K gold) +0.03pp에 그침. **핵심 원인**: Python `rescore_topk`는 raw 후보 점수만 비교, Rust `analyze_with_perceptron`는 후처리(`fix_*`) 추가 통과 — 측정 대상 불일치로 0.60pp 갭. 도메인 편차: SNS/구어/일상 +0.6~1.0pp 이득, 뉴스 -0.70pp 회귀(상쇄). 7,200 train 규모로는 뉴스 패턴 학습 부족. **재시도 시 주의**: (a) Python sim에도 동일 후처리 적용해 비교 가능하게, (b) 도메인-balanced sampling 또는 weight 분리, (c) 7,200 → silver corpus 확장 후 재시도가 본질적 해결책. 인프라(dump_for_training API, extract-training 바이너리, perceptron 학습 스크립트)는 폐기 — phase1 branch 삭제. 사용자가 부분 채택도 거부 (regression test 11건 영향).

## 폐기된 접근: CNN 재순위 (cnn2.bin)

**현재 분석기에 CNN은 없다.** codebook + N-best Viterbi + 어절 캐시 + 후처리 규칙만으로 동작하며 모델은 1.0 MB. `crates/garu-core/src/cnn.rs`도 삭제됨. POS 보정 후처리(`model.rs`)는 CNN이 골드에서 보이던 행동을 distill한 규칙으로 남아 있을 뿐 추론은 안 함. 신경 모델 재도입 금지(헤드룸 +1.5MB 있어도).

폐기 전 구조 (기록용, hidden=144 기준):

| 구성요소 | 파라미터 | int8 크기 |
|----------|----------|-----------|
| 임베딩 (3002×48) | 144K | 141 KB |
| Conv Layer 1 (k=3,5,9, 144ch) | 124K | 115 KB |
| Conv Layer 2 (k=3,7, 144ch) | 622K | 608 KB |
| 출력 FC (288→81) | 23K | 23 KB |
| 바이어스 + 스케일 + vocab | — | 24 KB |
| **합계** | ~913K | **brotli q=11 압축 후 701 KB** |
