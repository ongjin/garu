> **언제 읽나**: 모델 리빌드·WASM 빌드·골드/구어 held-out 평가·NIKL MP 벤치마크 명령의 전문이 필요할 때. 특히 2025판 `--norm-2025` 정규화가 무엇을 병합하는지, 어디까지 점수 비교가 가능한지 확인할 때.

# 빌드·평가 명령 (전문)

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

# 벤치마크 (NIKL MP 데이터: 기본 ~/workspace/data/nikl_mp_2021/. *.json glob)
#   kkma가 JVM 크래시로 스크립트 중단 → --analyzers garu,kiwi 권장
python3 training/eval_nikl_mp.py --n 2000 --analyzers garu,kiwi   # ⚠️ 2021판은 재순위 학습 데이터와 겹쳐 garu 자기평가(참고용). held-out은 위의 2025 구어 평가 사용
# 다른 코퍼스는 NIKL_MP_DIR로 override. 단 2025판은 분절 컨벤션이 거칠어져
#   (명사+하 병합, _복합어 결합) raw F1 비교불가 → --norm-2025 정규화 필요
#   (XSV/XSA 병합, 파생접미사 XSN 병합(적·성·화·권 등 16종 화이트리스트), _un-join,
#    인용 EF+고·며·는 병합, 직접인용 JKQ canonical, 되/VX→VV canonical까지 구현.
#    norm 후 2021 garu 0.9363 / 2025 garu 0.8806. 단위테스트 training/test_nikl_norm_2025.py.
#    천장 측정(2026-07): kiwi도 2021 0.879 / 2025+norm 0.8165 (하락폭 ~6pp 동일)
#    → 2025 고득점 자체가 불가능한 벤치. 2025 점수 목표 설정 금지.
#    되/VX는 2021/gold=VV·2025만 VX인 컨벤션차라 정규화(2025 되/VX ~48% vs 2021 ×4).
#    2025 잔여격차(~5.5pp)는 대부분 복합명사 분절(FN 28%)·태그컨벤션(있VA/하VX/와JKB)이라
#    분석기 결함 아닌 분절 컨벤션 문제. 하/VX·구어 그/IC는 진짜 차이라 미정규화)
NIKL_MP_DIR=~/workspace/data/nikl_mp_2025 python3 training/eval_nikl_mp.py --n 2000 --analyzers garu,kiwi --norm-2025

# 단일 문장 분석 (디버깅): GARU_MODEL 지정 + analyze_batch 예제
GARU_MODEL=js/models/base.gmdl cargo run -q --release --example analyze_batch <입력파일>
```

## 영문 원문 (구 경로 — 기록용)

데이터 경로가 옛 것(`~/Downloads/NIKL_MP(v1.1)/`)이라 현행 명령은 위 한국어 절을 따른다. 2026-09-04 AGENTS.md 에서 옮김.

### Common Commands (영문 원문)

```bash
# Rebuild model
python3 training/build_codebook_model.py

# Rust tests
cargo test

# WASM build
wasm-pack build crates/garu-wasm --target web --out-dir ../../js/pkg

# NIKL MP benchmark, requires ~/Downloads/NIKL_MP(v1.1)/
python3 training/eval_nikl_mp.py --n 2000

# Gold testset evaluation
python3 training/gold_testset/eval_f1.py
```

Use focused commands first when iterating. Run broader tests before claiming a general improvement.
