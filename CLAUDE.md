# Garu (가루) — 프로젝트 컨텍스트

## 프로젝트 개요

브라우저에서 실행되는 초경량 한국어 형태소 분석기. 신경망 없이 코드북 + Viterbi + 어절 캐시 + 후처리 규칙으로 동작.
- **F1 90.8%** (5,000문장 수동검증 골드 테스트셋, Kiwi 89.7% 대비 +1.1%p) / NIKL MP 93.5%
- **모델 1.2MB** (gzip 압축, npm 패키지에 포함, CDN 불필요)
- **WASM 119KB** — 브라우저에서 <1ms 추론

## 아키텍처

```
입력 텍스트 → 전체 문장 래티스 구축 (캐시 항목을 저비용 아크로 주입)
           → 문장 수준 Trigram Viterbi → 후처리 → 출력
           (동형이의어 "나는" 등을 어절 간 문맥으로 해소)
```

### 모델 구성 (codebook.gmdl, GMDL v3 포맷)
| Section | 내용 | 크기 |
|---------|------|------|
| 6 | 내용어 사전 (FST, 다중 POS) | 1,061 KB |
| 7 | 접미사 코드북 (31K 패턴, string table + u8 freq 양자화) | 847 KB |
| 8 | 트라이그램 비용 (sparse bitmap + u8 양자화) | 36 KB |
| 9 | 빈도 메타데이터 | 8 B |
| 10 | 분석기 파라미터 (mp=0.25, op=4.0, lb=1.5, sc=3.5) | 16 B |
| 11 | 모호성 테이블 (비활성) | 4 B |
| 12 | 단어 바이그램 비용 보정 (734 규칙) | 8 KB |
| 13 | 스마트 어절 캐시 (10K 엔트리, compact format) | 230 KB |
| — | **gzip 압축 후 전체** | **1,174 KB** |

### 핵심 Rust 코드
- `crates/garu-core/src/codebook.rs` — 래티스 구축, Viterbi 디코딩, 어절 캐시, 후처리
- `crates/garu-core/src/trie.rs` — FST 사전 (다중 POS: u64에 2개 POS pack)
- `crates/garu-core/src/types.rs` — 42개 세종 POS 태그 enum
- `crates/garu-wasm/src/lib.rs` — WASM 바인딩
- `crates/garu-tools/src/build_dict.rs` — FST 빌더 (다중 POS 지원)

### 학습 파이프라인 (Python)
- `training/extract_codebook.py` — Kiwi + kowikitext에서 코드북 추출
- `training/extract_nikl_codebook.py` — NIKL MP 골드 데이터에서 코드북 추출
- `training/build_codebook_model.py` — GMDL 바이너리 빌드 (FST, 코드북, 트라이그램, 캐시 통합)
- `training/eval_nikl_mp.py` — NIKL MP 벤치마크 (Garu vs Kiwi)

### JS/TS (npm 패키지)
- `js/src/index.ts` — Garu 클래스, load/analyze/tokenize API
- `js/models/base.gmdl` — 번들된 모델
- `js/pkg/` — wasm-pack 빌드 출력

## 연구 이력

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

## 빌드

```bash
# 모델 리빌드
python3 training/build_codebook_model.py

# Rust 테스트
cargo test

# WASM 빌드
wasm-pack build crates/garu-wasm --target web --out-dir ../../js/pkg

# 벤치마크 (NIKL MP 데이터 필요: ~/Downloads/NIKL_MP(v1.1)/)
python3 training/eval_nikl_mp.py --n 2000
```

## 규칙

- 커밋 메시지에 AI/Claude 관련 내용 포함 금지
- git email: dydwls140@naver.com
- 설계/계획 문서를 repo에 올리지 않음
