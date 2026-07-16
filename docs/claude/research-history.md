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

31. **NIKL 2025 학습 주입 검증 → No-Go** (2026-07-02). 어절 캐시 in-place A/B 실측: broad 3000어절 −0.63pp, v15k-미참조 intrinsic −0.18pp, v15k-커닝 필터만 +0.09pp(테스트셋 오염이라 무효). 원인: 문맥 무시 캐시의 다의 구어 어절 강제 오버라이드 + 2025 컨벤션 오염 + 2025 SX 볼륨(2021의 7.3%)·레지스터 협소(게임/스트리밍 전사체). #30 perceptron·Phase α와 동일 계열 결론. **부산물 채택**: 2025→2021 역변환기(`training/reverse_2025_to_2021.py`, 무손실 99.94%) + held-out 구어 평가셋(`eval_nikl2025_guueh.py`, F1 0.9010, 학습 사용 금지).
32. **무위험 후처리 룰 5종** (2026-07-02) → **+0.16pp (v15k 0.9528→0.9544)**. fix_geureon_mm(그렇+ㄴ→MM)·fix_seyo_merge(시+아요→세요)·fix_jeoneun_np(절/VV+는→저/NP)·fix_yo_jx_merge stem 완화·fix_present_copula(명사+다/JX→이/VCP+다). `구`/`는데`(골드 양방향)·`야`(호격 동형)는 실측 근거로 제외. **룰 트랙 수렴 확정** — 잔여 무위험 헤드룸 ≈ +0.01pp뿐, 나머지 오답 ~68%는 문맥 POS(폐기 CNN 영역)·양방향 분절 컨벤션.
33. **Lossless 속도 스택** (2026-07-02) → **2.35× 가속** (WASM 0.62→0.26ms/sent), Kiwi 격차 5.8×→2.4×. opt-level z→3 + wasm-opt -Oz→-O3(WASM raw 341→391KB, brotli +9KB) + inline FxHash(SipHash 18% 제거) + word_bigram 조회 arc당 1회 hoist. 9,000문장 출력 byte-identical + F1 0.9544 불변 검증. 잔여 lossless 후보(DP dense화·할당 churn)는 #34에서 viterbi_nbest에 대해 착수 완료.
34. **문맥 POS 재순위 perceptron 채택** (2026-07-13) → **+0.31pp (v15k 0.9544→0.9575), 구어 held-out +0.90pp (0.9010→0.9100), 전 도메인 무회귀**. #30 폐기 원인 3종을 정면 수정: (a) 측정 정합 — Rust `Model::analyze_topn` 덤프(fix 39종+R룰 적용 후보)로 학습·시뮬, Python 재구현 금지, Rust-Python pick 일치 97.9% 검증 (b) 데이터 — NIKL 2021 골드 316K 문장 (**v15k 완전일치 오염 3,950문장(44%!) 제외 필수** — grep 실측) (c) 뉴스 회귀 — NX 15만 학습으로 소멸(+0.19pp). 구성: top-10 후보, feature v1(FNV-1a 해싱 2^20, `training/rerank/features.py` = `rerank.rs` 바이트 동일), averaged perceptron, dev(6K) 튜닝 확신 마진 τ=4, GMDL Section 14(sparse 652KB raw, brotli 후 모델 1.04→1.43MB). **속도 게이트는 viterbi_nbest 최적화로 통과**: state별 비용 hoist(rank 루프에서 trigram/wb/internal 재계산 제거) + rank 오름차순 조기 break + split_at_mut 무할당 DP 접근 → k=10이 최적화 전 k=5보다 빨라짐(2.46→2.44s), byte-identical 검증. 최종 native +6%/WASM Kiwi-격차 2.4×→2.05×(개선). k=5 재순위는 +0.25pp로 컷라인(+0.3pp) 미달이라 k=10 채택 — nbest(5)가 nbest(20)[:5]보다 열등한 빔 효과 실측. **부작용: NIKL 2021이 재순위 학습 데이터가 되어 eval_nikl_mp 2021 벤치는 이후 자기평가(참고용)**. oracle 천장(Phase 0): k=5 +0.87pp/k=10 +1.47pp/k=20 +1.50pp(포화) — 재순위가 k=10 천장의 21% 회수, 잔여는 feature 용량 한계 *(→ #36에서 정정: 용량이 아니라 surface feature의 정보 한계)*.
35. **nbest DP arena화 + 할당 churn 조사** (2026-07-16) → **native/WASM 각 ~3% 가속, byte-identical**. viterbi_nbest의 state별 `Vec<(f32,Backpointer)>` 힙 할당을 stride=k+1 단일 arena로 교체(`HashMap<(u8,u8),u32>` 슬롯 인덱스 — 키 삽입 순서·용량 성장 보존으로 f32 tie-break까지 동일, v15k 9000+구어 16.4K+top10 덤프 3종 해시 일치, 테스트 77개 통과, WASM raw +2.2KB=401KB). **한계 실측**: 심볼 프로파일 콜트리 귀속 결과 잔여 alloc ~48%는 analyze_topn 토큰/String 생성(2134 샘플)·build_lattice 사전 materialize(1225)·FST 조회(584)에 **얇게 분산** — DictEntry가 FST 조회마다 String materialize + arc가 재복제하는 2중 구조. 단일 대형 레버 없음: packed FST 반환/Rc 분석열 공유(29개 arc 사이트 + 4499 API 파급)로 추정 8~12% 추가 가능하나 ROI 낮아 보류. "DP dense화 20~30%" 기대는 반증됨(DP churn은 전체의 소수). 남은 대형 격차는 알고리즘 계층(Kiwi 격차 ~2.2×)이지 할당이 아님.
36. **재순위 feature v2 시도 → No-Go** (2026-07-16). 잔여 분해 실측(v15k oracle-final 갭 +0.89pp의 95%=픽 오류, 게이트 4.7%뿐·τ=0 F1 동일)에 근거해 rank1-대비 diff-span 템플릿(d+/d-/d+b/dn·endw) 설계, Python(extract_v2)+Rust(score_v2, Section 14 ver=2) 파리티 구현, NIKL 316K 재덤프(오염 4,121 제외·재현 검증 dev rank1 0.9425=#34 일치) 후 재학습. **결과: dev pick 0.9472→0.9475(+0.03pp)**. 해시 충돌 가설도 검증 — 2^20에서 feature 이름 ~45% 충돌 실측 → DIM 2^22 재학습 → **0.9475 완전 동일(충돌 무관)**. v15k 시뮬 +0.04pp(0.9575→0.9578, 문학 −0.12pp) → 게이트(+0.15pp) 미달, **전량 revert**(스크립트·Rust 모두). **진단 확정: 잔여 픽 오류는 surface feature(형태·POS·span 통계)의 정보 밖** — 서로 다른 두 용량 개입(템플릿 확장·해시 4배)이 같은 지점에서 포화. 부가 발견: dev(NIKL) oracle 갭 0.27pp vs v15k 0.89pp — **재순위 학습에 없는 도메인(문학·기술·엣지)이 잔여 갭의 상당분**(시뮬도 SNS +0.31/일상 +0.26 게인 vs 문학 회귀로 일관). 재시도 열쇠: 의미/문맥 모델(신경 금지로 사실상 닫힘), v15k 유사 도메인 골드 확보, 또는 후보 생성 다양화(트랙 ③). 같은 날 구어 held-out 신규 실측: **Garu 0.9100 vs Kiwi 0.9068 (+0.32pp 역전)** — eval_nikl2025_guueh 골드에 Kiwi 최초 측정.
37. **트랙③ 후보 생성 다양화 Phase 0+1** (2026-07-16) → **+0.04pp (v15k 0.9575→0.9579), 기술 +0.22·문학 +0.14·엣지 +0.12pp, 전 도메인·held-out(0.9100) 무회귀**. Phase 0 측정: top-10 oracle조차 틀리는 2,186문장의 FN 3,419토큰을 dump_arcs_batch로 분류 — ARC_EXISTS(스코어링) 67%/상한 +2.28pp, DIFFPOS(POS결손) 22.5%/+0.76pp, NO_ARC 표면 5.4%·재구성 5.0%. 복합명사 컨벤션은 5%뿐. Phase 1 랜딩(50c850d·eca490f): ①tech_supplement 보강(최적화·감수성·장중·그쵸, 웹툰 NNG — supplement가 suffix-충돌 제거에서 보호되는 설계 활용) ②구요/EF freq bump(SPOKEN_SUFFIX_BUMPS 신설 — MIN_SUFFIX_FREQ=75에 걸린 기존 항목 add-or-bump) ③ㄷ불규칙 과거형 생성(었/었다 케이스가 augment에 아예 없었음 — 걷·싣·깨닫·일컫만, **듣·묻은 들다/물다 동형어 압도로 순회귀 실측 제외**) ④뒀=두+었 축약. **기각(실측)**: 요즘/MAG dual(-3.27 문장F1, 골드 NNG 49:10), 와/IC(효과 0), 달/NNB(골드 5:4 양방향), 고요·면은·크게·위로·배가(분해 우세), 오/이/NR·주/되(동형어·컨벤션 천장). **부산물 발견: 모델 픽스처 분기 사고** — models/codebook.gmdl(테스트용)이 0a36108에서 멈춰 0.9.9 재순위 wrong-override 회귀 3건(이거 실화냐 **대박→대/XPN+박**, 갈리없는데→갈리/VV, 인가→이/VCP+ㄴ가)이 테스트를 통과한 채 배송돼 있었음. 픽스처 동기화로 현재 integration 3건 red — 처리 방향 미결(재순위 개선 vs 가드 룰 vs 테스트 주석). 스코어링 클래스(+2.28pp 상한)는 전역 비용 모델 영역이라 보류.

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
