> **언제 읽나**: 모델 리빌드·WASM 빌드·골드/구어 held-out 평가·NIKL MP 벤치마크 명령의 전문이 필요할 때. 특히 2025판 `--norm-2025` 정규화가 무엇을 병합하는지, 어디까지 점수 비교가 가능한지 확인할 때. 분석기 변경의 채택 판정은 아래 '채택 게이트 v2'를 따른다.

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

# 2025 구어 held-out 평가 (SX 16.4K문장, 2025→2021 역변환 골드, garu F1 0.923. 학습 사용 금지)
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

## 채택 게이트 v2 (2026-11-02 고정)

기본 분석기 변경의 정확도 판정 규칙이다. 결과를 본 뒤 규칙·세트·분위를 바꾸지 않는다. 바꿀 때는 버전을 올리고, 이미 판정한 후보에 소급하지 않는다.

```bash
# 기준·후보 각각 예측과 문장별 채점 (gold·sx25·nx25, 약 25초)
python3 training/gate/score.py predict --model js/models/base.gmdl --out /tmp/base
python3 training/gate/score.py predict --model /path/cand.gmdl --out /tmp/cand
# 판정
python3 training/gate/gate.py /tmp/base /tmp/cand --json /tmp/gate.json
python3 -m unittest discover -s training/gate -p 'test_*.py'
```

**세트.** `gold`=v15k 9,000문장(층=7도메인, 단위=문장), `sx25`=NIKL 2025 구어 16,407문장(단위=문서 75개), `nx25`=NIKL 2025 문어(2023년 기사) 13,765문장(층=9주제, 단위=문서 1,027개). 2025 두 세트는 기존 held-out과 같은 필터·`reverse_convert`·ep_norm을 쓴다. nx25는 v15k·재순위 train/dev와 공백 제거 텍스트 중복이 없고, 역변환 왕복 성질은 13,692/13,765(99.47%)에서 성립한다.

**판정.** Δ = 후보 − 기준 ep_norm micro F1. 층 안에서 단위를 복원추출하는 paired bootstrap(B=10,000, seed 20261102)으로 분위를 구한다.
1. **이득**: held-out 합동(sx25∪nx25) Δ의 단측 2.5% 분위 > 0.
2. **유의 회귀 없음**: 하위 집합 m개(gold 전체·도메인, sx25 전체, nx25 전체·주제 = 19개) 각각에서 Δ의 단측 상위 0.05/m 분위가 0 미만이면 기각한다.
3. **비용**: 모델 크기·native 시간·WASM 영향은 따로 보고하고 사용자가 판단한다. 기존의 +0.15pp 최소 이득 기준은 비용 판단으로 옮긴다.

한 세트를 평가할 수 없는 후보(예: 산출물이 없어 기록된 편집만 복원 가능한 경우)는 가능한 세트만으로 m을 다시 세어 판정하고 결과를 **부분 판정**으로 표시한다.

**고정 전 보정.** 규칙을 고정하기 전에 대조군 두 종류를 돌렸다. 재순위를 seed 42로 재학습하면 배포본 Section 14와 바이트 동일했다. 학습 순서 seed만 43·44로 바꾼 음성 대조군은 둘 다 이득 미확인·유의 회귀 없음으로 기각됐다. 이 두 모델에서 점 추정이 음수인 하위 집합은 각각 8개였고, 최대 gold 일상 −0.121pp·기술 −0.045pp였다. 즉 학습 순서만 바꿔도 이전의 "전 도메인 점 추정 무회귀" 조건은 깨진다. 양성 대조군인 Section 16(연구 이력 #70) 유무 비교는 합동 Δ +0.0175pp(구간 하한 +0.0130)로 통과했다.

**해석 한계.** sx25는 문서 75개뿐이라 구간이 넓다. nx25의 기준 F1 86.33%는 역변환이 복합명사·`적/성/화` XSN·날짜(`1월`)·간접인용 `라고`의 2025 관례를 2021 관례로 되돌리지 못한 영향이 크다. 쌍대 Δ에는 공통 오차가 상쇄되지만, 복합명사 분절을 바꾸는 후보는 이 관례 차이에 따라 불리하게 측정될 수 있다. gold는 30%(2,693문장)가 검수 없는 `auto` 라벨이고 예전 Garu 출력이 출처에 들어 있어 안전 확인에만 쓴다. nx25도 사용할수록 노출되므로 사용 이력을 연구 이력에 남긴다.

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
