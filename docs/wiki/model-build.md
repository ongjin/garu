> **언제 읽나**: 모델(`base.gmdl`)·사전(FST)·접미사 코드북·어절 캐시·문맥 POS 교정을 리빌드하거나, 학습 파이프라인(Python) 스크립트를 만질 때. 선택적 JS 경계 모델의 학습·배포 범위도 여기서 구분한다.

# 모델 구성 (codebook.gmdl, GMDL v3 포맷)

기본 형태소 모델은 사전·탐색 비용과 학습된 보정 데이터를 함께 보관한다. 리빌드할 때 큐레이션한 캐시와 채택된 보정 blob을 보존해야 기존 분석을 재현할 수 있다. 선택적 띄어쓰기·문장 경계 모델은 이 GMDL에 포함하지 않는다.

| Section | 내용 | 크기 (raw bytes, 2026-10-20) |
|---------|------|------|
| 6 | 내용어 사전 (FST, 다중 POS) | 1,132,342 |
| 7 | 접미사 코드북 (44,760패턴, string table + u8 freq 양자화) | 1,010,736 |
| 8 | 트라이그램 비용 (sparse bitmap + u8 양자화) | 37,061 |
| 9 | 빈도 메타데이터 | 8 |
| 10 | 분석기 파라미터 (mp=0.25, op=4.75, lb=1.5, sc=3.5) | 16 |
| 11 | 모호성 테이블 (비활성) | 4 |
| 12 | 단어 바이그램 비용 보정 (965규칙) | 8,209 |
| 13 | 스마트 어절 캐시 (10K 엔트리, compact format) | 251,500 |
| 14 | N-best 재순위 perceptron 가중치 (95K sparse, ver=2 varint+int16) | 284,651 |
| 16 | 문맥 조건부 POS 교정 (58어절, 220조건) | 3,054 |
| 17 | 최종 출력열 일치 POS 교정 (246규칙, 624조건, 2026-11-02) | 10,742 |
| 18 | 구어 탐색 비용 보정 (아크 가중치 20K + 2^15 int8 판별기, 2026-11-02) | 94,575 |
| — | **brotli q=11 압축 후** | **1,343,482 B** |

⚠️ **Section 7 string table은 u16 오프셋 한계(65,535 B)에 거의 닿아 있다** — 현행 `MIN_SUFFIX_FREQ=75`에서 64,562 B(98.5%)라 형태를 ~140개만 더 늘려도 넘친다. 빌더는 넘칠 때만 sub-version 2(u32 오프셋)로 승격하고 그 미만은 기존 포맷 그대로 기록하므로 현행 모델은 byte-identical이다(research-history #54). 디코더는 1·2 모두 읽는다. **승격된 모델은 0.9.16 이하 런타임이 못 읽으니 배포 시 버전 게이트 확인.** 임계를 낮춰 코드북을 키우는 것 자체는 F1 이득이 없다(th=10에서 +0.03pp에 모델 +53%).

`build_codebook_model.py`는 Section 13(어절 캐시)을 **기존 `eojeol_cache.bin`을 그대로 기록** — 캐시를 리빌드하지 않는다(curated 캐시 보존, full rebuild는 -2pp 회귀 위험). 출력은 `models/codebook.gmdl` → `js/models/base.gmdl`로 복사. 소스가 동기화돼 있으면 무변경 리빌드는 byte-identical(재현성 보장).

⚠️ **두 모델 파일은 반드시 동기 유지**: `models/codebook.gmdl`은 integration 테스트 픽스처, `js/models/base.gmdl`은 배포본. 0.9.9 때 `export_weights.py`가 js 쪽만 갱신해 픽스처가 Section 14 없는 구버전으로 남았고, 그 결과 재순위 wrong-override 회귀 3건(대박→대+박, 갈리없는데, 인가가)이 테스트에 안 잡힌 채 배송됨(2026-07-16 픽스처 동기화로 발견, research-history #37). gmdl을 직접 수정하는 도구를 쓸 때는 두 파일 모두 갱신할 것.

**Section 14 (재순위 perceptron, 2026-07-13 채택)**: `training/rerank/` 파이프라인(prep_nikl→dump_topk k=10→train_perceptron→tune_margin)으로 학습, `export_weights.py --blob`이 gmdl 주입 + `training/codebook_data/rerank_section14.bin` blob 갱신. **빌더는 이 blob을 passthrough** (Section 13 어절 캐시와 같은 패턴 — blob 없으면 섹션 생략). 포맷은 `crates/garu-core/src/rerank.rs` 참조(FNV-1a feature hashing — feature 문자열 규칙이 `training/rerank/features.py`와 바이트 단위 동일해야 함). 섹션이 없으면 분석기는 재순위 없이 k=5로 동작(구모델 호환). **ver=2(2026-08-05)**: bucket 오름차순 차분 varint + 가중치 int16 양자화(step = max|w|/32767)로 652→244KB raw, 모델 −219KB. 디코더는 ver=1도 계속 읽는다. 기존 gmdl을 변환할 때는 `reencode_section14.py <blob> <gmdl>...`(weights.npz 불필요, 섹션만 in-place 교체). 양자화는 비트 동일이 아니므로 **변환 후 9,000문장 출력 diff로 확인할 것** — 채택 시점 실측은 v15k 9,000 + 구어 held-out 16,407 전부 byte-identical. ⚠️ 재학습 시 v15k∩NIKL 오염 3,950문장 제외 필수(prep_nikl.py가 자동 처리) + NIKL 2021 벤치는 이후 자기평가임에 유의. ⚠️ **`RERANK_TRAIN_PER_SRC=1000000000`(전체 316K)로 돌릴 것** — 기본값 40K는 train 80K짜리 축소판이라 dev pick 0.9471·nnz 43K로 배포본에 못 미친다(전체는 0.9478·nnz 95K). 학습 산출물 `training/rerank/data/`는 680MB라 gitignore 대상이고 스크립트로 재생성한다(prep 25초 + dump 3분 + preprocess 80초 + 8 epoch 2분). ⚠️ **dev pick이 동률이어도 재학습 가치가 있을 수 있다** — 사전이 바뀌면 후보 분포가 달라져 dev(NIKL)엔 안 보이고 v15k·held-out에서만 이득이 나온다(research-history #49). 반드시 export 후 실측할 것.

**Section 16 (문맥 조건부 POS 교정)**: 정본은 Git으로 관리하는 `training/codebook_data/context_pos.bin`이다. 빌더는 이 blob을 그대로 기록하며, 파일이 없으면 섹션을 생략한다. `training/ambiguity/prepare.py`는 NIKL 2021의 문서 단위 train/dev 분할과 고정 평가셋의 공백 제거 중복 제외 후 train에서 대체 POS와 주변 어절 조건을 추출한다. `probe.py`는 고정 규칙을 모델에 넣어 실제 분석 결과를 비교한다. 포맷의 legacy 비용 필드 `-1.0`은 동일 모델 재현을 위해 남아 있으며 현재 추론이나 비용 탐색에는 사용하지 않는다.

**Section 17 (최종 출력열 일치 POS 교정, 2026-11-02 채택)**: 정본은 `training/codebook_data/context_rules.bin`이고 빌더가 그대로 기록한다. Section 16과 달리 원래 열이 캐시 항목일 필요가 없고, 규칙마다 원래 형태·POS열과 바꿀 위치·POS를 저장한다. 재생성은 `prepare.py`의 train/dev에 대해 `analyze_batch --json-spans`로 현행 출력을 덤프한 뒤 `context_rules.py mine` → `select`(dev 32설정) → `export ... --blob`이다. 절차와 판정 기록은 `training/ambiguity/README.md`·`review_20261102.json`. 사전·캐시·재순위가 바뀌면 원래 열이 달라지므로 규칙을 다시 채굴하고 게이트 v2로 다시 판정한다.

**Section 18 (구어 탐색 비용 보정, 2026-11-02 채택)**: 정본은 `training/codebook_data/search_cost.bin`이고 빌더가 그대로 기록한다. v2(`GSCW\x02`)는 아크 feature 가중치(FNV-1a 2^20 버킷 중 상위 20K, i16)와 문자 1–3-gram 나이브 베이즈 구어 판별기(2^15 int8, 임계값 0)를 담는다. 판별 점수가 임계값을 넘는 입력에만 래티스 아크 비용에 `scale(0.5) × Σw`를 더한다. 학습·내보내기 절차는 `training/search/README.md`에 있다. Section 14 재순위와 Section 17 규칙은 이 보정 전의 후보·출력으로 학습된 상태다. 셋 중 하나를 바꾸면 나머지의 입력 분포도 달라지므로, 함께 다시 만들고 게이트 v2로 판정한다.

교정은 후보 생성·재순위·기존 model 보정이 끝난 뒤 적용한다. 선택된 어절의 형태·POS열·문자 span이 원래 Section 13 캐시와 모두 같고, 대체 분석이 형태를 유지한 채 POS 하나만 바꿀 때만 적용한다. 어절 전체 또는 바꿀 형태소가 사용자 사전에 있으면 건너뛴다. 다른 토큰의 형태·순서·span·점수는 바꾸지 않는다. 평가의 도메인 합계가 비음수가 되어도 개별 문장은 회귀할 수 있으며, 기존 기본 모델의 학습 노출과 고정셋의 과거 연구 노출이 있어 새로운 독립 일반화 평가로 해석하지 않는다. 실험 비교는 [연구 이력 #70](research-history.md)을 따른다.

**dual-POS 강제 override** (Section 6, 사전은 단어당 POS 2개만 pack): content_dict가 명사 POS 하나만 가져 동사 읽기가 누락되는 어간을 build에서 secondary POS로 주입. 두 상수 — `RIEUL_DUAL`(ㄹ불규칙 어간, A4 발동용), `HOMOGRAPH_VERB_DUAL`(명사 동형이의에 가린 동사 어간 박/팔/추/개 등 16개). 세 번째로 **`nng_dual.txt`**(데이터 파일, 594개) — 위키 제목 유입으로 **NNP 단독 등재된 일반명사**(대역폭·디코더·쿼리·노드·트래픽)에 NNG를 **동일 freq**로 주입해 trigram이 문맥으로 고르게 한다. 분절은 안 바뀌고 POS만 경쟁하므로 안전한 편이지만 **무차별 확대는 확실히 회귀**한다 — 위키 제목 전량(51,529개 적용)은 v15k −0.88pp(뉴스 −1.92)이고 freq 스케일을 0.1까지 낮춰도 baseline을 못 넘는다(research-history #45). 목록 추가 시 근거는 NIKL NNG 우세이며, 언어명(영어·일본어)·국호·지명은 골드 컨벤션이 NNP라 제외할 것.

앞의 두 동사 상수는 명사 primary를 보존하고 freq를 보수적으로 줘 trigram이 결정하게 한다(POS-trigram이 어미-뒤-어간 vs 조사-뒤-명사를 자연 분리하므로 명사+조사는 거의 회귀 안 함). 단 **축약 과거형**(쟀다/뿌렸다=재/뿌리+었)은 어간만 추가해도 모음축약 재구성 갭 때문에 안 고쳐져 제외. 후보는 `find_missing_verb_stems.py`로 조사(재실행 시 123개 나오지만 상위권 주·비·차·세·대는 전부 과거 기각 클래스다). **의존명사 동형어도 무조건 제외는 아니다** — 개(NNB freq 80,826, 기존 등재 항목 최대치의 3배)를 VV 1500으로 넣었는데 25,407문장에서 회귀 0이었다(research-history #43). 판단 기준은 명사 빈도가 아니라 **문맥이 갈리는가**: NIKL에서 개/NNB의 84.3%가 수사 선행이고 위험 구간인 `개/NNB+이/VCP` 114건은 100% 수사 선행이라 trigram이 완전히 분리한다. **사전 변경은 반드시 골드 F1 무회귀 게이트** 통과 확인.

**suffix-충돌 제거 오폭 주의**: 빌더는 코드북에 기능형태소 분석이 있는 명사류(REMOVABLE_POS)를 사전에서 제거하는데(배가/NNG가 배+가를 막는 것 방지), **실제 단어가 지워질 수 있다** — 의대/NNG(freq 480)가 코드북 `의/JKG+대/XPN`(freq 1955)에 밀려 삭제돼 `의대 증원`→의+대 과분해(research-history #38). 수동 검토한 `tech_supplement.txt` 등재 단어만 이 제거에서 보호한다. 대량 반입 목록인 `bulk_supplement.tsv`는 같은 예외를 받지 않으므로, 빌드 후 실제 FST에 반영됐는지 확인한다. 저빈도 복합명사 과분해(의상실 freq 3 → 의+상실)도 수동 supplement freq bump(200이면 충분 — 저빈도 비선형 페널티 구간만 벗어나면 됨)로 수리.

**불규칙 활용 증강** (Section 7 코드북, `build_codebook_model.py`): `augment_irregular_conjugations`가 content_dict의 ㅂ/ㄷ/ㅅ/르/ㅡ탈락/ㅎ 어간을 `SUFFIX_COMBOS`(어/었/은/을/으니…) 활용형으로 펼쳐 코드북에 넣는다. ㅂ불규칙 과거(추웠다=춥+었), ㅅ불규칙(`IRREG_SIOT_STEMS`, 저었다=젓+었), ㅡ탈락 과거(아팠다=아프+었, ㅆ-병합)까지 커버. `augment_irregular_honorific`은 ㅂ/ㄷ/ㅅ 어간+존댓말(고우시다=곱+으시+다, 걸으신다=걷+으시, 저으신다=젓+으시)을 어절 단위로 주입 — `augment_honorific`(자음어간 접미사만)이 못 잡는 불규칙 surface prefix 붕괴를 막음. 회귀 가드 `training/test_irregular_restore.py`(과거·존댓말·정칙불변). 정칙 ㅂ(좁다)·계사(이었다)는 과대생성돼도 실텍스트에 없어 무해(v15k 무회귀로 확인).

## 학습 자료 위치

원본 말뭉치는 레포 밖 `~/workspace/data/`에 둔다.

| 자료 | 위치 | 용도 |
|---|---|---|
| NIKL 형태분석 2021 | `nikl_mp_2021/` | 기본 사전·코드북·재순위 학습 |
| NIKL 형태분석 2025 | `nikl_mp_2025/` | 2025판 변환·평가. 구어 held-out은 학습에 사용하지 않음 |
| NIKL 문어 2025 | `nikl_written_2025/` | 문학·기술과학 등 원문. 형태소 정답이 없는 원시 자료 |
| NIKL 온라인 게시 2025 | `nikl_online_2025/` | 블로그·인스타그램 원문. 형태소 정답이 없는 원시 자료 |
| KLUE-DP·UD KAIST·UD KSL | `korean_pos/` | 고정된 공개 원본·라이선스와 변환된 형태소/POS 자료 |
| MeCab·한국어기초사전·우리말샘 | `korean_lexicon/` | 전체 원본 사전·정규화 어휘·출처/라이선스/검토 근거. 문장 형태소 골드가 아님 |

NIKL 문어·온라인의 한국어 PDF가 설명서다. 문어의 `기술과학` 분류에는 건강·육아·요리가 포함되고 IT 책은 사회과학·철학 등에도 있으므로, IT 표본을 분류명만으로 고르지 않는다. 문어는 `document[].paragraph[].form`을 문장 분리하며, 온라인 `form`과 `original_form`은 이모지 등에서 다를 수 있어 둘 다 보존한다. 원문이나 자동 분석 초안을 골드로 취급하지 않는다.

`korean_pos/raw`는 공식 commit과 라이선스를 보존한다. `source_manifest.json`은 원본 해시와 URL, `filter_manifest.json`은 기존 변환·제외 기준, `normalized_manifest.json`은 문장 수와 원문·라벨 해시다. `normalized/train.jsonl`은 39,301문장이며 출처별 dev/test도 별도 파일이다. `origin.file`은 `korean_pos` 기준 상대 경로다. 정규화 파일의 품사·형태소열은 이동 전과 같고 원본 위치만 바꿨다. 기존 NIKL 약정과 CC BY-SA 이용 조건은 모델 공개 전에 개별 확인한다.

### 공개 POS 자료의 변환

UD의 `LEMMA`에서 보조용언 등의 어미가 생략되고 `XPOS`에는 어미 태그가 남을 수 있다. `OrigLemma`가 있으면 명시된 원래 형태소열을 쓰고, 없으면 문장을 제외한다. KAIST `xsa`는 부사파생접미사이므로 세종 형용사파생 `XSA`로 치환하지 않는다. KSL NF/NV/NA처럼 가루가 표현하지 못하는 태그도 문장 전체를 제외한다. 변환기는 `korean_pos/adapter/corpora.py`다. 이 검사로 자료 간 분절 관례가 완전히 일치한다고 보장하지는 않는다.

새 자료를 학습에 넣기 전 원본 출처·문서/작성자 분할과 기존 v15k·구어 held-out·train/dev의 NFC·공백 제거 중복을 확인한다. 기존 공개 자료의 같은 작품·작성자나 의역 중복까지 자동으로 해결되는 것은 아니다. 현재 정규화 자료는 앞선 검사를 통과한 부분집합이며 공식 평가셋 전량 점수로 해석하지 않는다.

### 재순위 학습과 비교

도메인 이름만으로 추가 학습 효과를 예상하지 않는다. 목표 분석이 후보에 있는지 먼저 확인하고, 기존 자료만 재학습한 대조군과 같은 설정의 자료 추가 모델을 비교한다. 후보 생성·모델·전처리가 바뀌면 실제 입력으로 후보와 캐시를 다시 만든다. 모델·마진은 dev에서 고정한 뒤 골드·구어의 전체·도메인 F1과 비용을 평가한다. 과거 5M 문맥·직접 분석·탐색 비용·편집 규칙 및 공개 POS 자료 추가 결과는 [연구 이력](research-history.md)에 있다.

공식 학습 덤프 `crates/garu-core/examples/dump_topk.rs`는 `Analyzer::analyze_topn_for_rerank`를 사용한다. 이 API는 실제 재순위 점수 입력과 `apply_rerank_rules`를 공유하며, Section 16의 최종 POS 교정과 표시용 자모 정규화를 제외한다. 공개 `analyze_topn`은 Section 16과 옵션에 따른 표시용 자모 정규화를 포함하므로 그 출력을 학습 후보로 대체하면 추론 피처와 달라진다. 덤프는 JSON으로 토큰·위치·점수를 직렬화하며 점수를 소수 네 자리로 자르지 않는다.

`train_perceptron.preprocess()`는 gold/candidate JSONL의 실제 바이트 SHA-256과 로드한 producer·feature·ep_norm 소스의 SHA-256을 NPZ 내부 `_manifest`와 대조한다. 일치하는 캐시만 재사용하며, 서명이 없는 과거 캐시·입력 불일치·손상된 archive는 다시 생성한다. 새 캐시는 같은 디렉터리의 임시 파일에 완성한 뒤 atomic replace하므로 저장 실패가 기존 캐시를 덮어쓰지 않는다. 반환 배열은 기존 여섯 키를 유지한다.

학습 목적함수를 바꾸면 점수 척도도 바뀌므로 마진을 개발셋에서 다시 선택한다. 실제 분석 경로에는 문맥 보너스와 조기 반환이 있어 rank1 기준 시뮬레이션만으로 선택하지 않는다. 재순위 이전 출력을 비교할 때 Section 14를 제거하면 k=10이 k=5로 바뀐다. 같은 탐색 폭을 유지하려면 Section 14에 영 가중치를 넣은 실험 모델을 사용한다. 고정 피처라도 학습 방법에 따라 비영 가중치 수와 압축 크기는 달라진다.

Native `analyze_batch`는 입력 줄을 trim하므로 WASM과 비교할 때 같은 입력을 사용한다. 경계 공백을 그대로 둔 WASM 결과와 비교하면 기존 모델에서도 동점 경로의 토큰 순서 차이가 생길 수 있다. 공개 API A/B는 원래 공백을 포함한 입력의 토큰·위치·score도 따로 비교한다.

### 선택적 텍스트 경계 모델

`garu-ko/text`의 `restoreSpacing`·`splitUnspaced`는 별도로 가져오는 JS API다. 구현은 `js/src/text.ts`, `boundaries.ts`, `boundary-data.ts`에 있으며 가중치는 기본 GMDL에 넣지 않는다. 기존 `analyze`·`tokenize`는 이 API를 자동 호출하지 않는다. 띄어쓰기 복원은 원래 문자와 기존 공백을 보존하면서 공백을 추가하므로, 복원 결과를 분석하려면 호출자가 명시적으로 연결한다.

학습·선택·내보내기 절차는 [training/boundaries/README.md](../../training/boundaries/README.md)를 따른다. NIKL 2021 문서로 분할하며 고정 골드·기존 reranker dev·2025 SX 중복을 학습에서 제외한다. 이어 붙인 원문 기록의 경계 공백은 합성이고, SX의 EF 조건은 문장 완결성의 약한 감독 신호다. 선택된 모델과 임계값을 고정한 뒤 원문에 정확히 정렬한 2025 NX 및 컴파일된 공개 JS API를 별도로 검증한다. 반복 실험에서 이미 본 고정셋 결과는 선택 후 검증으로 보고하며 미열람 최종 평가로 표현하지 않는다. Python 원시 모델과 JS의 기본 보수적 임계값·보호 규칙을 구분한다. 상세 비교는 [연구 이력 #71](research-history.md)에 둔다.

`garu-ko/sentences`는 기존 후보에 인용문 보호와 Q4 문맥 검증을 추가한다. `splitSentences(text)`는 분석기 없이 `js/src/sentence-fast-data.ts`의 문자 문맥 모델을 사용한다. 분석기를 전달하면 `sentence-data.ts`의 형태소 검증 모델을 선택한다. 두 모델은 별도 GMDL을 요구하지 않으며 분석기 모델은 형태소 경로에만 필요하다. 짝이 맞는 인용문은 내부에 여러 문장이 있어도 보존한다. `export_verifier.py`는 완성 JSON을 명시적 타입으로 내보내 `.d.ts`에 가중치 리터럴을 복제하지 않는다. 모델에는 문자 어휘, 위치별 문맥 임베딩, dense 가중치, 행별 Q4 스케일, float32 bias와 임계값이 들어간다.

검증기 학습은 NX의 원래 문단 ID 묶음과 보수적으로 걸러낸 SX 연속 발화를 사용한다. 문자 경계와 무공백·종결부호 삭제 변형의 중복을 문서 분할 및 기존 골드·개발·2025·이전 boundary 참조와 대조한다. 입력 특징은 실제 JS/WASM 경로에서 추출하며 cache/source SHA를 검사한다. 후보 바깥의 정답도 FN에 포함하고 양자화한 개발 지표로 epoch·임계값을 선택한다. MPS의 padding gradient 오류를 막는 명시적 마스크와 export의 영 벡터 검사, 비유한 점수 거부를 유지한다.

빠른 모델은 같은16개 특징 ABI에서 형태소9개 슬롯을0으로 두고 학습·추론한다. `mine_verifier.py --features context`는 형태소 분석을 생략하며 `verifier.py`의 `--context`·`--embedding`·`--hidden`으로 크기를 지정한다. `export.py`가 문자 경계 모델별 FNV 점수식을 펼쳐 생성하므로 JS 런타임에 동적 코드 생성을 요구하지 않는다. 모델을 다시 내보낼 때 점수식도 함께 갱신한다.

2021 source-record는 인용문 내부 경계를 포함할 수 있어 인용문 보존 정책과 동일한 정답 기준이 아니다. 2025 고정셋에서 관찰한 인용문 문제를 보고 보호 규칙을 추가했으므로 이 효과를 학습 이득·미열람 일반화와 구분한다. 원본2021 지표를 바꾸지 않고 함께 기록하며, 정상 형태소 모델과 기본 JS 경로는 유지한다. 채택 근거·정책 대조군·지연·브라우저 검증은 `training/boundaries/review_verifier_20261020.json`, `review_speed_20261020.json`과 [연구 이력 #73·#74](research-history.md)에 둔다.

### 사전 변경과 임시 실험 정리

수동 보강 어휘는 `training/codebook_data/tech_supplement.txt`, 공개 사전에서 검토·채택한 어휘는 `training/codebook_data/bulk_supplement.tsv`가 정본이다. 후자의 `freq=7`은 보수적인 사전 가중치(양자화 값1)를 부여하기 위한 설정이며 말뭉치 출현 빈도가 아니다. MeCab의 비용·문맥 ID를 빈도로 옮기지 않는다. 정규 빌드로 두 모델 파일을 동기화하고, 직접 만든 실험 모델과의 해시 및 Section6 이외의 섹션 보존을 확인한다.

사전의 `명사`는 고유명사를 포함할 수 있고, 단일 표제어가 단일 형태소임을 보장하지 않는다. 뜻풀이·원어·원문 용례로 이름, 활용형과의 동형 충돌, 내부 분절을 검토한다. 자료가 없다는 이유만으로 일괄 제외하거나, MeCab NNG라는 이유만으로 문장 정답으로 승격하지 않는다. 표적 인식은 출처에 맞춘 단어·품사 일치 지표이며 문장 F1이 아니다. 평가 문맥을 보고 후보를 제외한 결과는 선택 후 검증으로 보고하고, 고유명사 판별 오류가 남을 수 있음을 명시한다.

어휘 채택·보류와 최초 검증 기록은 `training/lexicon/review_20261019_bulk.json`에 보존한다. 현재 모델·런타임 재현의 기본 기록은 `review_20261020_context.json`이며, `case_source`로 bulk 기록 파일과 SHA-256을 지정해 기존 6,539개 사례와 55개 문맥 가드를 재사용한다. 과거 보고서를 덮어쓰거나 사례·원문 사본을 중복 저장하지 않는다. 앞선 기록은 `review_20261019.json`(첫88개)과 `review_20261019_round2.json`(추가31개)이다.

`replay.py`는 참조 파일의 해시를 검증한 뒤 원본 문단을 다시 읽어 WASM 결과를 재집계한다. 목표 어휘와 같은 어절·변경 묶음에서도 조사·계사·이웃 단어의 품사가 바뀔 수 있으므로, 문단 전체의 전후 토큰을 검사하고 원인 후보 제거 A/B로 복구를 확인한다. 표적을 제외한 diff만 보는 검사는 이러한 변화를 놓칠 수 있다.

재현에는 원본 말뭉치와 빌드된 JS 패키지가 필요하다. 아래 기본 명령은 커밋3229d92의 모델(`190e1d2b…`)부터 현재 모델까지의 **누적 비교**다. Section 16만 추가한 **증분 비교**는 `training/ambiguity/review_20261020.json`에 분리한다. `--review`로 과거 기록을 고를 때는 그 기록의 `--baseline`과 `--candidate` 모델도 함께 지정한다. 모델·원문·런타임 해시가 다르면 재평가를 중단한다. 명시적 표적 위치만 세어 긴 단어 내부의 부분 일치를 양성으로 세지 않으며, 문맥 가드는 기록된 토큰열 복구를 검사한다. 이 재현 검사는 새 문맥의 의미 검토나 문장 전체 F1을 대신하지 않는다.

```bash
git show 3229d92:js/models/base.gmdl > /tmp/garu-baseline-3229d92.gmdl
python3 training/lexicon/replay.py --baseline /tmp/garu-baseline-3229d92.gmdl
```

가루 엔진·패키지 코드는 Apache-2.0이며, 반입 사전 데이터에는 출처별 조건이 적용된다. [데이터 NOTICE](../../js/models/NOTICE.md), `js/models/licenses/`, `js/models/bulk-provenance.json`의 출처·원본 ID·변경 고지를 모델과 함께 보존한다. 원본 사전·정규화 어휘·간결한 검토 근거는 `~/workspace/data/korean_lexicon/`에 둔다. 예문·뜻풀이·멀티미디어·원문 문단은 반입 어휘 목록에 포함하지 않는다.


원본을 새로 정규화할 때 사용하는 수집기는 `training/lexicon/import_mecab.py`, `import_krdict.py`, `import_opendict.py`다. MeCab에는 `--source-dir`로 snapshot의 `source/seed`, 한국어기초사전에는 위치 인자로 JSON ZIP, 우리말샘에는 `--source-zip`으로 JSON ZIP을 지정한다. 모두 `--output`을 받고, 요약 파일은 MeCab·우리말샘의 `--summary`, 한국어기초사전의 `--report`로 지정한다. 기존 정규화 파일을 덮어쓰기 전에 새 출력의 행 수·품사 매핑·해시를 검증한다. 정규화 출력 전체를 그대로 모델에 넣지 않고 채택한 목록만 bulk supplement에 반영한다.

폐기한 실험은 결과를 연구 이력에 기록한 뒤 작업 폴더·모델·덤프·체크포인트를 삭제한다. 채택된 변경은 본 저장소가 정본이다. `~/workspace/data/README.md`와 `cleanup.json`이 현재 보관 범위와 정리 기록이며, 과거 연구 이력의 `garu_*` 경로는 당시 위치다. 삭제된 실험 코드·모델의 재실행 절차로 사용하지 않는다.

# 학습 파이프라인 (Python)

- `training/extract_codebook.py` — Kiwi + kowikitext에서 코드북 추출
- `training/extract_nikl_codebook.py` — NIKL MP 골드 데이터에서 코드북 추출
- `training/build_codebook_model.py` — GMDL 바이너리 빌드 (FST, 코드북, 트라이그램, 캐시 통합, 자동 brotli q=11 압축)
- `training/ambiguity/{prepare,probe,context_cache}.py` — 문서 분할·문맥 POS 규칙 추출, Section 16 직렬화·실제 분석 비교
- `training/ambiguity/context_rules.py` — 최종 출력열 기준 한 POS 규칙 채굴·dev 선택·Section 17 직렬화
- `training/search/{router,verify_gold,export}.py` + `crates/garu-core/examples/search_train.rs` — 구어 판별기·정답 경로 검증·native 탐색 비용 학습·Section 18 직렬화
- `training/boundaries/` — 선택적 JS 띄어쓰기·문장 경계 모델 학습·선택·내보내기·공개 API 평가
- `training/boundaries/{prepare_verifier,mine_verifier,verifier,export_verifier,eval_verifier}.py` 및 대응 `.mjs` — 원문 문단·hard negative·Q4 문맥 검증기 학습과 실제 JS 평가
- `training/eval_nikl_mp.py` — NIKL MP 벤치마크 (Garu vs Kiwi)
- `training/gold_testset/eval_f1.py` — 골드 테스트셋 (9,000문장 v15k, ep_norm) F1 평가
- `training/find_missing_verb_stems.py` — NNP/NNG 동형이의에 가려 동사 읽기가 누락된 어간 조사 (Garu collapse vs Kiwi VV/VA, `HOMOGRAPH_VERB_DUAL` 후보 추출)
- `training/neural/prepare_data.py`, `training/neural/experiment_all.py` — *(폐기된 CNN 학습용. 현재 분석기는 CNN 미사용 — `research-history.md` 참조)*

## 이력

- 2026-11-02 Section 18 구어 탐색 비용 보정 채택. 모델 1,343,482B(+79,588B), SHA-256 `daec0d758cb6024dbd6c3243b11582a63b5b8d1197b6d8e9b7e23df1e7434fb5`. 게이트 v2 통과, 집중 테스트 1건 알려진 회귀. 상세는 연구 이력 #79.
- 2026-11-02 Section 17 최종 출력열 일치 POS 교정 채택. 모델 1,263,894B(+3,418B), SHA-256 `670a77bf4a6dc1cac120b592a7200f20fcdb3af67da194f96ded04051fbe8146`. 게이트 v2 세 세트 통과, 상세는 연구 이력 #78.
- 2026-10-20 Section 16 문맥 POS 교정을 채택하고 학습 후보 API를 최종 교정과 분리. 기존 Section 6–14를 보존한 정규 빌드 모델은 1,260,476B(+2,042B), SHA-256 `f9ca34a2ad1d0a152f434dc21a755a802aef5a4f8d8246ff29625367d12a9ef7`. 고정 골드 모든 도메인 합계는 비음수지만 개별 문장 회귀는 남음. 선택적 JS 경계 모델을 기본 모델과 구분하고 누적 어휘 재현 기록을 현재 런타임으로 갱신. 상세는 연구 이력 #70·#71.
- 2026-10-19 공개 사전3종 전체를 수집하고 전체 합집합 모델은 폐기, NNG1,449개를 별도 bulk supplement로 채택. 정규 빌드 동일성·55문맥 가드·고정 F1과 비용을 확인하고 기본 재현 기록을 bulk로 전환. 상세는 research-history #69.
- 2026-10-19 어휘 94개 실험 후 문맥 회귀 6개를 보류하고 88개를 사전·두 모델에 적용. 골드 유지·구어 4문장 개선, 모델 +1,176B. 표적 원문 참조·해시와 WASM 재평가 도구 보존. 상세는 research-history #67.
- 2026-10-07 탐색 중 선형 비용 학습과 조건부 편집 규칙을 독립 평가. 탐색은 작은 골드·구어 이득, 규칙은 구어 이득과 골드 회귀가 나타나 기본 모델 유지. 고정 top-k 상한 해석의 범위와 참조 분절 관례 차이를 기록.
- 2026-10-07 독립 CNN·문맥 직접 분석 모델 학습·평가. 전체 Q4 3MB 조건과 구어 이득을 확인했으나 골드·별명 회귀로 기본 모델 유지. 기존 분석기와 연구 산출물 보존.
- 2026-10-06 전처리 캐시 입력·소스 검증과 atomic 저장, 배치 JSON 표준 직렬화 반영. 실험 디코더는 출력 동등성을 유지하며 장문 RSS를 줄였고 재순위 guard는 정상 이름 반례로 미반영. 원본 골드와 별도 검수판을 구분해 보존.
- 2026-10-06 회귀 선택 단계·prefix 실행 비용을 진단하고 전처리 캐시 입력 불일치와 CLI 역슬래시 JSON 오류를 재현. 코드 수정 없이 운영상 경계와 재현 위치를 기록.
- 2026-10-02 탐색 중 경로 병합·비지도 단어 군집 실험 완료. 경로 병합은 구어 이득과 기술·문학 회귀가 함께 나타났고, 학습 군집은 무작위 군집 대조군을 이기지 못해 기본 모델 유지. 상세는 research-history #56과 구조 실험 보고서.
- 2026-10-02 후보 누락 조사·SNS 유형별 보강·F1 손실 기반 PA-I 학습을 비교. SNS 잠정 평가의 국소 이득과 PA-I 구어 이득은 있었으나 전체 정확도·도메인 회귀·크기 조건을 충족한 교체안이 없어 기본 모델 유지. 상세는 research-history #55와 레포 밖 실험 보고서.
- 2026-09-21 20항목을 supplement에 반영하고 0.9.18 정규 모델 생성. 실험 모델과 byte-identical, 모델 1,247,331B. 기존 모델에서 실패하는 회귀 테스트를 추가한 뒤 Rust·골드 norm/raw·구어 평가 통과.
- 2026-09-21 새16항목을 추가한 총20항목 실험 모델 검증. 97문서108용례에서 목표 인식11→108건, 기존25,407문장 토큰/POS·WASM 위치 유지. 최초 모델 대비 Brotli +1,138B. score1건 변화와 주변 품사2건 검토 한계를 기록, 기본 모델 미반영.
- 2026-09-21 네 단어 freq200 사전 추가 실험 통과. 새 보류 68문서에서 목표 인식 7→68건, 기존 25,407문장 출력 동일, Brotli +978B. Native/WASM 7회 교차 측정에서 저하 없음. 실험 모델만 보관.
- 2026-09-21 문어·온라인 전체에서 평가용을 제외한 원문으로 어휘 후보 발굴. 추출 10,283개 중 실제 사전 미등재 7,241개, 비교 분석으로 좁힌 어휘 검토 후보 728개를 저장하고 대표 10개를 문맥에서 확인. 사전·모델은 변경하지 않음.
- 2026-09-21 문어·온라인 JSON 전수 확인과 도메인 300건 추출 후 개별 진단 검토 완료. 잠정 참조 231건·보류 69건, top-10 개선 8건(치환 표지 제외 7건)으로 균등 3,000건 확대보다 SNS 오류 유형 재검증을 권고.
