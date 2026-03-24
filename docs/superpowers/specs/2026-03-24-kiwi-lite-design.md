# Kiwi Lite — 초경량 한국어 형태소 분석기 설계 문서

## 1. 개요

**한 줄 정의:** 웹 브라우저에서 바로 돌아가는 5MB 이하 한국어 형태소 분석기

**프로젝트명:** kiwi-lite

**타겟 사용자:** NLP 개발자, 프론트엔드 개발자, 검색/에디터 등 클라이언트 사이드 한국어 처리가 필요한 개발자

## 2. 핵심 차별점

| | Kiwi (기존) | kiwi-lite (목표) |
|---|---|---|
| WASM 모델 크기 | ~83MB | ~5MB |
| 초기 로딩 | 수십 초 | 1초 이내 |
| 모바일 웹 | 사실상 불가 | 가능 |
| 오프라인 | 무리 | Service Worker 캐싱 |
| 품질 | 최상위 | Mecab-ko의 80~90% |

**트레이드오프:** 모델 크기를 줄이는 만큼 정확도에서 양보. 하지만 웹에서 "쓸 수 있다 vs 못 쓴다"의 차이가 핵심 가치.

## 3. 기술 아키텍처

### 3.1 기술 스택

| 영역 | 선택 | 근거 |
|---|---|---|
| 코어 언어 | **Rust** | WASM 컴파일 최적, wasm-pack 생태계, 메모리 안전성 |
| WASM 툴체인 | **wasm-pack + wasm-bindgen** | Rust→WASM 표준 파이프라인, npm 배포 통합 |
| 추론 엔진 | **커스텀 Rust 추론** (ONNX Runtime 사용 안 함) | ONNX Runtime Web만 5~10MB라 예산 초과. 직접 BiLSTM 순전파만 구현 (역전파 불필요) |
| ML 학습 | **PyTorch** (학습만, 배포에 미포함) | 학습 후 가중치를 커스텀 바이너리 포맷으로 export |
| 사전 구축 | **세종 코퍼스 + 국립국어원 사전** 기반 빈도 사전 | 공개 데이터로 구축 가능 |

### 3.2 모델 전략 (5MB 바이트 예산)

```
총 예산: 5MB (5,242,880 bytes)
─────────────────────────────────────
WASM 바이너리 (추론엔진 + 글루코드):  ~500KB
BiLSTM 가중치 (INT8 양자화):          ~1.5MB
  - 임베딩: 자소 입력(~200) + 어절 출력(~8,000 태그 조합) = ~500KB
  - LSTM 2층: ~800KB
  - 출력층: ~200KB
압축 사전 (LOUDS Trie):               ~2.5MB
  - 빈도 상위 30만 어절
  - 품사 태그 + 분석 결과 포함
JS 글루코드 + 기타:                    ~500KB
─────────────────────────────────────
합계:                                  ~5.0MB
```

**참고 사례:**
- MeCab 사전(ipadic): ~50MB → 빈도 기반 pruning으로 30만 어절 시 ~2.5MB 달성 가능
- 한국어 BiLSTM 품사 태거: 풀사이즈 ~20MB → INT8 양자화 + 작은 vocab으로 ~1.5MB 가능
- 자소(Jamo) 단위 입력 방식은 vocab을 ~200개로 극한 축소하여 임베딩 테이블 크기를 최소화

**모델 아키텍처:**
- 입력: 자소(Jamo) 단위 인코딩 (ㄴ,ㅏ,ㄴ,ㅡ,ㄴ → "나는") — vocab ~200개로 극한 축소
- BiLSTM 2층 (hidden 128) + CRF 출력층
- 출력: BIO 태깅 + 품사 라벨

### 3.3 평가 기준

| 항목 | 정의 |
|---|---|
| 평가 데이터셋 | 세종 형태소 분석 코퍼스 테스트셋 (표준 분할) |
| 주요 메트릭 | **형태소 단위 F1 score** |
| 목표 | Mecab-ko F1의 80~90% (Mecab-ko가 약 97% F1이라면, 목표는 78~87% F1) |
| 보조 메트릭 | 어절 단위 정확도, 초당 처리 토큰 수, 로딩 시간 |
| 벤치마크 환경 | Chrome latest, M1 MacBook Air, 4G 네트워크 시뮬레이션 |

### 3.4 배포 형태

```
kiwi-lite/
├── npm 패키지 (kiwi-lite)         # npm install kiwi-lite
├── CDN 배포                        # <script src="cdn/kiwi-lite.js">
├── Python 바인딩 (선택)            # pip install kiwi-lite
└── WASM + JS 글루코드
```

### 3.5 API 설계

#### MVP API (v0.1)

```js
import { Kiwi } from 'kiwi-lite';

// ── 초기화 ──────────────────────────────────────
const kiwi = await Kiwi.load();   // 5MB 기본 모델 로딩

// ── 형태소 분석 ─────────────────────────────────
const result = kiwi.analyze('나는 어제 학교에 갔다');
// {
//   tokens: [
//     { text: '나',   pos: 'NP',  start: 0, end: 1 },
//     { text: '는',   pos: 'JX',  start: 1, end: 2 },
//     { text: '어제', pos: 'MAG', start: 3, end: 5 },
//     { text: '학교', pos: 'NNG', start: 6, end: 8 },
//     { text: '에',   pos: 'JKB', start: 8, end: 9 },
//     { text: '가',   pos: 'VV',  start: 10, end: 11 },
//     { text: 'ㅆ',   pos: 'EP',  start: 11, end: 12 },
//     { text: '다',   pos: 'EF',  start: 12, end: 13 },
//   ],
//   score: -12.34,
//   elapsed: 2.1,  // ms
// }

// ── N-best 분석 ─────────────────────────────────
const nbest = kiwi.analyze('나는 배를 먹었다', { topN: 3 });
// [
//   { tokens: [...], score: -8.2 },   // 배 = 과일
//   { tokens: [...], score: -11.5 },  // 배 = 신체
//   { tokens: [...], score: -14.1 },  // 배 = 선박
// ]

// ── 토크나이저 모드 (검색/인덱싱 특화) ──────────
const tokens = kiwi.tokenize('서울특별시 강남구');
// ['서울', '특별시', '강남구']

// ── 유틸리티 ────────────────────────────────────
kiwi.isLoaded();                    // boolean
kiwi.modelInfo();                   // { version, size, accuracy }
kiwi.destroy();                     // WASM 메모리 해제
```

#### 에러 처리

| 상황 | 동작 |
|---|---|
| 빈 문자열 입력 | `{ tokens: [], score: 0, elapsed: 0 }` 반환 |
| 비한국어 텍스트 | 공백 기준 분리, `pos: 'SL'`(외국어) 또는 `pos: 'SN'`(숫자)로 태깅 |
| 한영 혼용 | 한국어 부분은 형태소 분석, 영문은 `SL`로 패스스루 |
| WASM 초기화 실패 | `Kiwi.load()`에서 에러 throw, 메시지에 원인 포함 |
| 매우 긴 입력 (>10만자) | 내부적으로 청크 분할 처리, 결과는 합쳐서 반환 |

#### Future API (v0.2+)

아래는 MVP 이후 추가 예정. 현재 구현하지 않음.

```js
// 배치 분석 (Web Worker 병렬)
kiwi.analyzeAll(sentences, { concurrency: navigator.hardwareConcurrency });

// 명사 추출
kiwi.extractNouns('자연어처리는 재미있다');

// 후처리 파이프라인
const pipeline = kiwi.pipe(
  PostProcessor.normalize(),
  PostProcessor.deInflect(),
  PostProcessor.compound(),
  PostProcessor.stopwords('ko'),
);

// 사용자 사전
kiwi.addWord('뉴진스', 'NNP', 9.0);
kiwi.addPreanalyzedWord('고려대학교', [
  { text: '고려', pos: 'NNP' },
  { text: '대학교', pos: 'NNG' },
]);

// 스트리밍 분석
const reader = kiwi.stream();
reader.on('token', (token) => console.log(token));
reader.feed('실시간으로 입력되는 텍스트');
reader.end();
```

### 3.6 TypeScript 타입

```ts
interface AnalyzeOptions {
  topN?: number;
  threshold?: number;       // score 컷오프
  concurrency?: number;     // Web Worker 수
}

interface Token {
  text: string;
  pos: POS;                 // 세종 태그셋 enum
  start: number;            // 원문 오프셋
  end: number;
  score?: number;
  subTokens?: Token[];      // 복합어 분해 결과
}

interface AnalyzeResult {
  tokens: Token[];
  score: number;
  elapsed: number;          // ms
}

type POS = 'NNG' | 'NNP' | 'NNB' | 'NR' | 'NP'
         | 'VV' | 'VA' | 'VX' | 'VCP' | 'VCN'
         | 'MAG' | 'MAJ' | 'MM'
         | 'IC'
         | 'JKS' | 'JKC' | 'JKG' | 'JKO' | 'JKB' | 'JKV' | 'JKQ' | 'JX' | 'JC'
         | 'EP' | 'EF' | 'EC' | 'ETN' | 'ETM'
         | 'XPN' | 'XSN' | 'XSV' | 'XSA' | 'XR'
         | 'SF' | 'SP' | 'SS' | 'SE' | 'SO' | 'SW' | 'SH' | 'SL' | 'SN';
```

## 4. 수익화 전략

### Phase 1 — 오픈소스 출시 & 인지도 확보 (0~6개월)

- MIT 라이선스로 코어 공개
- npm 패키지 + CDN 배포
- GitHub Sponsors / Open Collective 후원 채널 개설
- 기술 블로그 시리즈 (개발기, 벤치마크, Kiwi 대비 비교)
- 목표: GitHub Stars 500+, npm 주간 다운로드 1,000+

### Phase 2 — 프리미엄 모델 판매 (6~12개월)

| 티어 | 가격 | 내용 |
|---|---|---|
| Core (무료) | $0 | 5MB 기본 모델, 80% 정확도 |
| Pro | $29/월 또는 $199 영구 | 고정밀 모델(~15MB), 90%+ 정확도 |
| Domain | $49/월 | 의료/법률/커머스 등 도메인 특화 모델 |

- Gumroad 또는 Lemon Squeezy로 판매 (결제 인프라 최소화)
- 라이선스 키 기반 모델 다운로드

### Phase 3 — 기업 수익 (12개월~)

- 기업 라이선스: 연 $999~$2,999 (SLA + 우선 지원)
- 커스텀 사전/모델 구축 컨설팅: 건당 $3,000~$10,000
- 도입 사례가 쌓이면 SaaS API 추가 검토

### Phase 4 — 콘텐츠 수익 (Phase 1부터 병행)

- "5MB 형태소 분석기 만들기" 유료 강의 (Inflearn) — $30~50
- 개발 과정 뉴스레터/전자책
- 개발 과정 자체가 콘텐츠이자 마케팅

### 예상 수익 시나리오 (가설, 시장 검증 전, 12개월 후 목표)

```
후원:           $200/월
Pro 모델 판매:  $500/월 (17명)
강의 수익:      $300/월
───────────────────────
합계:          ~$1,000/월
```

## 5. 리스크 및 완화 전략

| 리스크 | 완화 |
|---|---|
| 5MB로 80% 정확도 달성 불가 | 사전 크기 vs 모델 크기 비율 조정, 단계적 로딩으로 10MB 허용 |
| Kiwi가 경량 모델 출시 | 속도와 DX에서 차별화, 커뮤니티 선점 |
| 시장 규모 불확실 | 콘텐츠 수익으로 리스크 헷지, 최소 비용 운영 |
| 혼자 개발 속도 한계 | MVP 먼저 출시, 커뮤니티 기여 유도 |

## 6. MVP 범위

첫 출시에 포함할 최소 기능:

- [x] WASM 빌드 + npm 패키지
- [x] `Kiwi.load()` + `analyze()` + `tokenize()`
- [x] 기본 모델 1종 (5MB 이하, 모델 티어 분리는 v0.2+)
- [x] TypeScript 타입 정의
- [x] 벤치마크 페이지 (Kiwi WASM 대비 로딩 속도/크기 비교)

첫 출시에서 제외:

- 후처리 파이프라인 (`PostProcessor`)
- 스트리밍 분석
- Python 바인딩
- Pro/Domain 모델
