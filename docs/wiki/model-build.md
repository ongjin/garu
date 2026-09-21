> **언제 읽나**: 모델(`base.gmdl`)·사전(FST)·접미사 코드북·어절 캐시를 리빌드하거나, 학습 파이프라인(Python) 스크립트를 만질 때. GMDL 바이너리 포맷의 섹션 구성과 각 학습 스크립트의 역할.

# 모델 구성 (codebook.gmdl, GMDL v3 포맷)

| Section | 내용 | 크기 (raw) |
|---------|------|------|
| 6 | 내용어 사전 (FST, 다중 POS) | 1,086 KB |
| 7 | 접미사 코드북 (31K 패턴, string table + u8 freq 양자화) | 847 KB |
| 8 | 트라이그램 비용 (sparse bitmap + u8 양자화) | 36 KB |
| 9 | 빈도 메타데이터 | 8 B |
| 10 | 분석기 파라미터 (mp=0.25, op=4.0, lb=1.5, sc=3.5) | 16 B |
| 11 | 모호성 테이블 (비활성) | 4 B |
| 12 | 단어 바이그램 비용 보정 (734 규칙) | 8 KB |
| 13 | 스마트 어절 캐시 (10K 엔트리, compact format) | 246 KB |
| 14 | N-best 재순위 perceptron 가중치 (95K sparse, ver=2 varint+int16) | 278 KB |
| — | **brotli q=11 압축 후** | **~1217 KB** |

⚠️ **Section 7 string table은 u16 오프셋 한계(65,535 B)에 거의 닿아 있다** — 현행 `MIN_SUFFIX_FREQ=75`에서 64,562 B(98.5%)라 형태를 ~140개만 더 늘려도 넘친다. 빌더는 넘칠 때만 sub-version 2(u32 오프셋)로 승격하고 그 미만은 기존 포맷 그대로 기록하므로 현행 모델은 byte-identical이다(research-history #54). 디코더는 1·2 모두 읽는다. **승격된 모델은 0.9.16 이하 런타임이 못 읽으니 배포 시 버전 게이트 확인.** 임계를 낮춰 코드북을 키우는 것 자체는 F1 이득이 없다(th=10에서 +0.03pp에 모델 +53%).

`build_codebook_model.py`는 Section 13(어절 캐시)을 **기존 `eojeol_cache.bin`을 그대로 기록** — 캐시를 리빌드하지 않는다(curated 캐시 보존, full rebuild는 -2pp 회귀 위험). 출력은 `models/codebook.gmdl` → `js/models/base.gmdl`로 복사. 소스가 동기화돼 있으면 무변경 리빌드는 byte-identical(재현성 보장).

⚠️ **두 모델 파일은 반드시 동기 유지**: `models/codebook.gmdl`은 integration 테스트 픽스처, `js/models/base.gmdl`은 배포본. 0.9.9 때 `export_weights.py`가 js 쪽만 갱신해 픽스처가 Section 14 없는 구버전으로 남았고, 그 결과 재순위 wrong-override 회귀 3건(대박→대+박, 갈리없는데, 인가가)이 테스트에 안 잡힌 채 배송됨(2026-07-16 픽스처 동기화로 발견, research-history #37). gmdl을 직접 수정하는 도구를 쓸 때는 두 파일 모두 갱신할 것.

**Section 14 (재순위 perceptron, 2026-07-13 채택)**: `training/rerank/` 파이프라인(prep_nikl→dump_topk k=10→train_perceptron→tune_margin)으로 학습, `export_weights.py --blob`이 gmdl 주입 + `training/codebook_data/rerank_section14.bin` blob 갱신. **빌더는 이 blob을 passthrough** (Section 13 어절 캐시와 같은 패턴 — blob 없으면 섹션 생략). 포맷은 `crates/garu-core/src/rerank.rs` 참조(FNV-1a feature hashing — feature 문자열 규칙이 `training/rerank/features.py`와 바이트 단위 동일해야 함). 섹션이 없으면 분석기는 재순위 없이 k=5로 동작(구모델 호환). **ver=2(2026-08-05)**: bucket 오름차순 차분 varint + 가중치 int16 양자화(step = max|w|/32767)로 652→244KB raw, 모델 −219KB. 디코더는 ver=1도 계속 읽는다. 기존 gmdl을 변환할 때는 `reencode_section14.py <blob> <gmdl>...`(weights.npz 불필요, 섹션만 in-place 교체). 양자화는 비트 동일이 아니므로 **변환 후 9,000문장 출력 diff로 확인할 것** — 채택 시점 실측은 v15k 9,000 + 구어 held-out 16,407 전부 byte-identical. ⚠️ 재학습 시 v15k∩NIKL 오염 3,950문장 제외 필수(prep_nikl.py가 자동 처리) + NIKL 2021 벤치는 이후 자기평가임에 유의. ⚠️ **`RERANK_TRAIN_PER_SRC=1000000000`(전체 316K)로 돌릴 것** — 기본값 40K는 train 80K짜리 축소판이라 dev pick 0.9471·nnz 43K로 배포본에 못 미친다(전체는 0.9478·nnz 95K). 학습 산출물 `training/rerank/data/`는 680MB라 gitignore 대상이고 스크립트로 재생성한다(prep 25초 + dump 3분 + preprocess 80초 + 8 epoch 2분). ⚠️ **dev pick이 동률이어도 재학습 가치가 있을 수 있다** — 사전이 바뀌면 후보 분포가 달라져 dev(NIKL)엔 안 보이고 v15k·held-out에서만 이득이 나온다(research-history #49). 반드시 export 후 실측할 것.

**dual-POS 강제 override** (Section 6, 사전은 단어당 POS 2개만 pack): content_dict가 명사 POS 하나만 가져 동사 읽기가 누락되는 어간을 build에서 secondary POS로 주입. 두 상수 — `RIEUL_DUAL`(ㄹ불규칙 어간, A4 발동용), `HOMOGRAPH_VERB_DUAL`(명사 동형이의에 가린 동사 어간 박/팔/추/개 등 16개). 세 번째로 **`nng_dual.txt`**(데이터 파일, 594개) — 위키 제목 유입으로 **NNP 단독 등재된 일반명사**(대역폭·디코더·쿼리·노드·트래픽)에 NNG를 **동일 freq**로 주입해 trigram이 문맥으로 고르게 한다. 분절은 안 바뀌고 POS만 경쟁하므로 안전한 편이지만 **무차별 확대는 확실히 회귀**한다 — 위키 제목 전량(51,529개 적용)은 v15k −0.88pp(뉴스 −1.92)이고 freq 스케일을 0.1까지 낮춰도 baseline을 못 넘는다(research-history #45). 목록 추가 시 근거는 NIKL NNG 우세이며, 언어명(영어·일본어)·국호·지명은 골드 컨벤션이 NNP라 제외할 것.

앞의 두 동사 상수는 명사 primary를 보존하고 freq를 보수적으로 줘 trigram이 결정하게 한다(POS-trigram이 어미-뒤-어간 vs 조사-뒤-명사를 자연 분리하므로 명사+조사는 거의 회귀 안 함). 단 **축약 과거형**(쟀다/뿌렸다=재/뿌리+었)은 어간만 추가해도 모음축약 재구성 갭 때문에 안 고쳐져 제외. 후보는 `find_missing_verb_stems.py`로 조사(재실행 시 123개 나오지만 상위권 주·비·차·세·대는 전부 과거 기각 클래스다). **의존명사 동형어도 무조건 제외는 아니다** — 개(NNB freq 80,826, 기존 등재 항목 최대치의 3배)를 VV 1500으로 넣었는데 25,407문장에서 회귀 0이었다(research-history #43). 판단 기준은 명사 빈도가 아니라 **문맥이 갈리는가**: NIKL에서 개/NNB의 84.3%가 수사 선행이고 위험 구간인 `개/NNB+이/VCP` 114건은 100% 수사 선행이라 trigram이 완전히 분리한다. **사전 변경은 반드시 골드 F1 무회귀 게이트** 통과 확인.

**suffix-충돌 제거 오폭 주의**: 빌더는 코드북에 기능형태소 분석이 있는 명사류(REMOVABLE_POS)를 사전에서 제거하는데(배가/NNG가 배+가를 막는 것 방지), **실제 단어가 지워질 수 있다** — 의대/NNG(freq 480)가 코드북 `의/JKG+대/XPN`(freq 1955)에 밀려 삭제돼 `의대 증원`→의+대 과분해(research-history #38). tech_supplement 등재 단어는 이 제거에서 보호되므로 이 클래스는 supplement 1줄로 수리된다. 저빈도 복합명사 과분해(의상실 freq 3 → 의+상실)도 supplement freq bump(200이면 충분 — 저빈도 비선형 페널티 구간만 벗어나면 됨)로 수리.

**불규칙 활용 증강** (Section 7 코드북, `build_codebook_model.py`): `augment_irregular_conjugations`가 content_dict의 ㅂ/ㄷ/ㅅ/르/ㅡ탈락/ㅎ 어간을 `SUFFIX_COMBOS`(어/었/은/을/으니…) 활용형으로 펼쳐 코드북에 넣는다. ㅂ불규칙 과거(추웠다=춥+었), ㅅ불규칙(`IRREG_SIOT_STEMS`, 저었다=젓+었), ㅡ탈락 과거(아팠다=아프+었, ㅆ-병합)까지 커버. `augment_irregular_honorific`은 ㅂ/ㄷ/ㅅ 어간+존댓말(고우시다=곱+으시+다, 걸으신다=걷+으시, 저으신다=젓+으시)을 어절 단위로 주입 — `augment_honorific`(자음어간 접미사만)이 못 잡는 불규칙 surface prefix 붕괴를 막음. 회귀 가드 `training/test_irregular_restore.py`(과거·존댓말·정칙불변). 정칙 ㅂ(좁다)·계사(이었다)는 과대생성돼도 실텍스트에 없어 무해(v15k 무회귀로 확인).

## 도메인 원문과 검수 자료

도메인 보강용 원문은 레포 밖 `~/workspace/data/`에 둔다. 아래 두 말뭉치는 형태소 정답이 없는 원시 자료이며, 기존 `nikl_mp_2025`의 구어 held-out과 별개다.

| 자료 | JSON 위치 | 분류 기준 |
|---|---|---|
| 문어 말뭉치 | `nikl_written_2025/NIKL_WRITTEN 2025_v1.0/` | 파일의 `metadata.category`와 책의 제목·본문 |
| 온라인 게시 자료 | `nikl_online_2025/NIKL_Online Posting Materials Corpus 2025/` | `metadata.category`: 블로그 / 누리소통망 |

각 상위 폴더의 한국어 PDF가 배포 설명서다. 문어의 `기술과학` 분류에는 건강·육아·요리가 포함되고 IT 책은 사회과학·철학 등에도 있으므로, IT 표본은 분류명만으로 고르지 않는다. 누리소통망 자료의 매체는 인스타그램이다. 문어는 `document[].paragraph[].form`을 문장 분리해야 하고, 이 배포본의 누리소통망에는 `paragraph[].sentence[]`도 있다. 온라인의 `form`과 `original_form`은 이모지 등에서 다를 수 있으므로 둘 다 보존한다.

검수용 산출물은 `~/workspace/data/garu_domain_pilot_2025/`에 있다. `inventory.json`은 전체 JSON의 분류·건수, `manifest.json`은 추출 조건·출처 해시·중복 제외 범위, `baseline.json`은 분석에 사용한 코드·모델 식별자다. `pilot_300.jsonl`은 출처 ID와 문단 내 위치를 가진 원문 표본, `review_300.jsonl`은 Garu 최종 출력·top-10 후보·Kiwi 초안을 붙인 검수 자료다.

표본은 `diagnostic_only`이며 원래의 `review_300.jsonl`은 미검수 입력으로 보존한다. 개별 검토 결과는 같은 폴더의 `adjudication/REPORT.md`와 `adjudication/reviewed_300.jsonl`에 있다. `reference_morphemes`는 진단용 잠정 참조, `gold_morphemes`는 계속 null이며 독립된 사람이 확정한 골드로 취급하지 않는다. 불확실한 분절·품사·문장 경계는 `needs_review`로 제외한다. Garu와 Kiwi의 불일치 자체는 오류 판정이 아니다.

기술 표본은 IT 서적·키워드로 선별하므로 도메인 전체를 대표하는 무작위 벤치마크가 아니다. 온라인 `form`에는 &가 제거된 `brand/company/name` 치환 표지가 남기도 해 이를 제외한 집계도 확인한다. 현재 표본의 진단은 균등한 데이터 확대보다 SNS의 특정 오류 유형 재검증을 지지한다. 후보에 더 나은 분석이 없다는 판정은 top-10 범위에 한정하며, 래티스 결손·빔 밖 정답·후처리 손상을 구분하려면 별도 아크 조사가 필요하다. 구체 수치와 재현 명령은 위 보고서 및 `adjudication/diagnostic_summary.json`을 따른다.

중복 제외는 기존 v15k, 재순위 train/dev, 구어 held-out 원문의 NFC·공백 제거 후 일치 기준이다. 의역 중복과 기존 자료와의 문서 단위 중복까지 검증한 것은 아니다. 학습 자료로 확대할 때는 책·게시글 단위 분리를 유지하고 SNS 작성자와 동일 작품의 다른 판본도 확인한다. 원문이나 자동 분석 초안을 기존 골드 파일에 합치지 않는다.

## 전체 말뭉치 어휘 후보 발굴

원문 기반 발굴 결과와 재현 스크립트는 `~/workspace/data/garu_word_mining_2025/`에 둔다. `REPORT.md`가 처리 범위·제외 기준·검증 결과를 설명하고, `LEXICAL_REVIEW.md`와 `lexical_review_queue.jsonl`이 문서 수·표면 출현 수·실제 용례·Garu/Kiwi 단독 분석을 제공한다. 이 목록은 사전 추가 승인이나 형태소 정답이 아니다. 실제 사전 포함 여부는 소스 목록 대신 배포 GMDL Section 6을 조회한다.

발굴 전에는 파일럿 원문·동일 책을 제외하고, 새 평가 그룹을 책 제목+저자 또는 온라인 매체+작성자 단위로 남긴다. SNS 파일럿이 전체 작성자를 포함하므로 파일럿 작성자를 모두 제외하지는 않으며, 기존 파일럿을 향후 독립 성능 평가로 사용하지 않는다. v15k·구어 held-out·재순위 dev 문장 중복도 정해진 문자열 기준으로 제외한다. 구체 경계와 원본 해시는 `preparation.json`을 따른다.

전체 원문을 순회해도 추출 임계치를 통과한 한글 명사형 후보만 남으므로 완전한 어휘 열거는 아니다. 실제 사전 미등재여도 코드북·캐시·OOV로 처리할 수 있고, 통째로 출력되지 않아도 정상 파생·합성어 분석일 수 있다. 후보를 사전에 넣기 전 실제 문단 분석과 품사·분절 검수를 거친다. 모델이나 사전이 바뀌면 `baseline.json`과 결과를 다시 대조한다.

### 소규모 사전 추가 실험

`~/workspace/data/garu_dictionary_trial_2025/`에는 나이아신아마이드·세라마이드·가브리살(NNG), 에픽테토스(NNP)의 네 항목 추가 실험이 있다. `REPORT.md`, `protocol.json`, `selection.json`이 품사 근거·사전 고정한 기준·선택 과정을 설명한다. 발굴용 용례에서 가장 낮은 빈도값을 고른 뒤 새 보류 그룹과 기존 골드·구어를 평가하며, 보류 점수로 값을 다시 튜닝하지 않는다.

실험 모델은 복사한 GMDL의 Section 6에만 항목을 삽입해 기존 packed 사전값과 다른 섹션을 보존한다. 검증한 항목은 `training/codebook_data/tech_supplement.txt`에서 관리하고 정규 빌드로 두 모델 파일을 동기화한다. 새 보류 자료의 목표 단어/POS 인식률은 문장 전체 F1이 아니다. WASM 검증은 Node에서 수행하며 실제 브라우저 측정과 구분한다.

Native `analyze_batch`는 입력 줄을 trim하므로 WASM과 비교할 때 같은 입력을 사용한다. 경계 공백을 그대로 둔 WASM 결과와 비교하면 기존 모델에서도 동점 경로의 토큰 순서 차이가 생길 수 있다. 공개 API A/B는 원래 공백을 포함한 입력의 토큰·위치·score도 따로 비교한다.

확대 실험은 `~/workspace/data/garu_dictionary_expansion_2025/`에 있다. `active_targets.json`의 일반명사·고유명사를 각각 검증한 뒤 `models/combined_f200.gmdl`에 합쳤다. 신규 평가에서는 앞선 실험의 보류 문서를 제외하며, 문서 수와 용례 수를 구분한다. 주변 변화 감사의 미확정 품사와 잔여 오류는 `reviews/diff_audit.json`에 남긴다. 목표 단어 인식 성공을 문장 전체의 정답으로 해석하지 않는다.

사전 항목 추가는 토큰·위치가 같아도 공개 `score`(분석 비용)를 바꿀 수 있다. 확대 모델은 기존 골드의 글램핑 용례에서 이 차이가 있으므로 출력 무변화 여부를 텍스트/POS·위치·score로 나누어 보고한다. 검증한 20항목은 기본 모델에 정규 빌드로 반영됐으며 실험 모델과 바이트 단위로 동일하다. 실험 근거는 확대 실험의 `REPORT.md`와 `validation.json`, 정규 빌드·배포 검증 기록은 `~/workspace/data/garu_release_0_9_18/`에 있다.

# 학습 파이프라인 (Python)

- `training/extract_codebook.py` — Kiwi + kowikitext에서 코드북 추출
- `training/extract_nikl_codebook.py` — NIKL MP 골드 데이터에서 코드북 추출
- `training/build_codebook_model.py` — GMDL 바이너리 빌드 (FST, 코드북, 트라이그램, 캐시 통합, 자동 brotli q=11 압축)
- `training/eval_nikl_mp.py` — NIKL MP 벤치마크 (Garu vs Kiwi)
- `training/gold_testset/eval_f1.py` — 골드 테스트셋 (9,000문장 v15k, ep_norm) F1 평가
- `training/find_missing_verb_stems.py` — NNP/NNG 동형이의에 가려 동사 읽기가 누락된 어간 조사 (Garu collapse vs Kiwi VV/VA, `HOMOGRAPH_VERB_DUAL` 후보 추출)
- `training/neural/prepare_data.py`, `training/neural/experiment_all.py` — *(폐기된 CNN 학습용. 현재 분석기는 CNN 미사용 — `research-history.md` 참조)*

## 이력

- 2026-09-21 20항목을 supplement에 반영하고 0.9.18 정규 모델 생성. 실험 모델과 byte-identical, 모델 1,247,331B. 기존 모델에서 실패하는 회귀 테스트를 추가한 뒤 Rust·골드 norm/raw·구어 평가 통과.
- 2026-09-21 새16항목을 추가한 총20항목 실험 모델 검증. 97문서108용례에서 목표 인식11→108건, 기존25,407문장 토큰/POS·WASM 위치 유지. 최초 모델 대비 Brotli +1,138B. score1건 변화와 주변 품사2건 검토 한계를 기록, 기본 모델 미반영.
- 2026-09-21 네 단어 freq200 사전 추가 실험 통과. 새 보류 68문서에서 목표 인식 7→68건, 기존 25,407문장 출력 동일, Brotli +978B. Native/WASM 7회 교차 측정에서 저하 없음. 실험 모델만 보관.
- 2026-09-21 문어·온라인 전체에서 평가용을 제외한 원문으로 어휘 후보 발굴. 추출 10,283개 중 실제 사전 미등재 7,241개, 비교 분석으로 좁힌 어휘 검토 후보 728개를 저장하고 대표 10개를 문맥에서 확인. 사전·모델은 변경하지 않음.
- 2026-09-21 문어·온라인 JSON 전수 확인과 도메인 300건 추출 후 개별 진단 검토 완료. 잠정 참조 231건·보류 69건, top-10 개선 8건(치환 표지 제외 7건)으로 균등 3,000건 확대보다 SNS 오류 유형 재검증을 권고.
