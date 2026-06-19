> **언제 읽나**: 모델(`base.gmdl`)·사전(FST)·접미사 코드북·어절 캐시를 리빌드하거나, 학습 파이프라인(Python) 스크립트를 만질 때. GMDL 바이너리 포맷의 섹션 구성과 각 학습 스크립트의 역할.

# 모델 구성 (codebook.gmdl, GMDL v3 포맷)

| Section | 내용 | 크기 (raw) |
|---------|------|------|
| 6 | 내용어 사전 (FST, 다중 POS) | 1,061 KB |
| 7 | 접미사 코드북 (31K 패턴, string table + u8 freq 양자화) | 847 KB |
| 8 | 트라이그램 비용 (sparse bitmap + u8 양자화) | 36 KB |
| 9 | 빈도 메타데이터 | 8 B |
| 10 | 분석기 파라미터 (mp=0.25, op=4.0, lb=1.5, sc=3.5) | 16 B |
| 11 | 모호성 테이블 (비활성) | 4 B |
| 12 | 단어 바이그램 비용 보정 (734 규칙) | 8 KB |
| 13 | 스마트 어절 캐시 (10K 엔트리, compact format) | 246 KB |
| — | **brotli q=11 압축 후** | **~1004 KB** |

`build_codebook_model.py`는 Section 13(어절 캐시)을 **기존 `eojeol_cache.bin`을 그대로 기록** — 캐시를 리빌드하지 않는다(curated 캐시 보존, full rebuild는 -2pp 회귀 위험). 출력은 `models/codebook.gmdl` → `js/models/base.gmdl`로 복사. 소스가 동기화돼 있으면 무변경 리빌드는 byte-identical(재현성 보장).

**dual-POS 강제 override** (Section 6, 사전은 단어당 POS 2개만 pack): content_dict가 명사 POS 하나만 가져 동사 읽기가 누락되는 어간을 build에서 secondary POS로 주입. 두 상수 — `RIEUL_DUAL`(ㄹ불규칙 어간, A4 발동용), `HOMOGRAPH_VERB_DUAL`(NNP/NNG 동형이의에 가린 동사 어간 박/팔/추 등 15개). 명사 primary는 보존하고 freq를 보수적으로 줘 trigram이 결정(POS-trigram이 어미-뒤-어간 vs 조사-뒤-명사를 자연 분리하므로 명사+조사는 거의 회귀 안 함). 단 **축약 과거형**(쟀다/뿌렸다=재/뿌리+었)은 어간만 추가해도 모음축약 재구성 갭 때문에 안 고쳐져 제외. 후보는 `find_missing_verb_stems.py`로 조사. **사전 변경은 반드시 골드 F1 무회귀 게이트** 통과 확인.

# 학습 파이프라인 (Python)

- `training/extract_codebook.py` — Kiwi + kowikitext에서 코드북 추출
- `training/extract_nikl_codebook.py` — NIKL MP 골드 데이터에서 코드북 추출
- `training/build_codebook_model.py` — GMDL 바이너리 빌드 (FST, 코드북, 트라이그램, 캐시 통합, 자동 brotli q=11 압축)
- `training/eval_nikl_mp.py` — NIKL MP 벤치마크 (Garu vs Kiwi)
- `training/gold_testset/eval_f1.py` — 골드 테스트셋 (9,000문장 v15k, ep_norm) F1 평가
- `training/find_missing_verb_stems.py` — NNP/NNG 동형이의에 가려 동사 읽기가 누락된 어간 조사 (Garu collapse vs Kiwi VV/VA, `HOMOGRAPH_VERB_DUAL` 후보 추출)
- `training/neural/prepare_data.py`, `training/neural/experiment_all.py` — *(폐기된 CNN 학습용. 현재 분석기는 CNN 미사용 — `research-history.md` 참조)*
