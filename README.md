# 가루 (Garu)

**세계 최초의 브라우저 네이티브 한국어 형태소 분석기.**

기존의 한국어 형태소 분석기(Kiwi, MeCab-ko, Komoran 등)는 모두 서버 환경을 전제로 설계되어, 20~50MB의 모델과 네이티브 런타임이 필요했습니다. Garu는 **2.1MB 모델과 93KB WASM 엔진**만으로 브라우저에서 직접 실행되며, 서버 통신 없이 완전한 오프라인 형태소 분석을 수행합니다.

---

## 특징

- **브라우저 최초** -- 웹 브라우저에서 실용적으로 동작하는 최초의 한국어 형태소 분석기
- **초경량** -- 2.1MB 모델 (npm 패키지에 포함, CDN 불필요)
- **높은 정확도** -- F1 90.8% (5,000문장 수동 검증 골드 테스트셋 기준, Kiwi 89.7% 대비 +1.1%p)
- **서버 불필요** -- WebAssembly로 클라이언트에서 직접 실행
- **신경망 없음** -- 코드북 + Trigram Viterbi + 어절 캐시, 학습 파라미터 0개
- **오프라인 지원** -- 네트워크 없이도 완전한 형태소 분석 가능
- **[라이브 데모](https://garu.zerry.co.kr)** -- 브라우저에서 바로 체험

---

## 비교

|  | Kiwi | MeCab-ko | Garu |
|---|---|---|---|
| 모델 크기 | ~40MB | ~50MB | **2.1MB** |
| npm 패키지 | 없음 | 없음 | **포함** |
| F1 (골드 테스트셋) | 89.7% | — | **90.8%** |
| F1 (NIKL MP) | 87.9% | ~85% | 93.5% |
| 추론 방식 | BiLSTM+CRF | CRF+사전 | **Lattice+Viterbi** |
| 학습 파라미터 | 수백만 | 수십만 | **0** |
| 브라우저 실행 | 비실용적 | 불가 | **지원** |
| 모바일 웹 | 비실용적 | 불가 | **최적** |

> **벤치마크 참고**: 골드 테스트셋은 뉴스·일상·SNS·기술·문학·엣지케이스 5,000문장을 사람이 직접 검증한 데이터입니다. NIKL MP는 표준어 신문 중심 데이터로 Garu의 구어체·신조어 개선이 반영되지 않습니다. v0.3.0부터 나무위키 구어체 코퍼스를 추가해 골드 테스트셋 F1이 향상되었으며, NIKL MP 수치(93.5%)는 이전 버전 대비 소폭 하락했습니다.

---

## Quick Start

```bash
npm install garu-ko
```

```typescript
import { Garu } from 'garu-ko';

// 모델 로드 (npm 패키지에 포함되어 별도 다운로드 불필요)
const garu = await Garu.load();

// 형태소 분석
const result = garu.analyze('배가 아파서 약을 먹었다');
console.log(result.tokens);
// [
//   { text: '배',   pos: 'NNG', start: 0, end: 2 },
//   { text: '가',   pos: 'JKS', start: 0, end: 2 },
//   { text: '아프', pos: 'VA',  start: 3, end: 6 },
//   { text: '어서', pos: 'EC',  start: 3, end: 6 },
//   { text: '약',   pos: 'NNG', start: 7, end: 9 },
//   { text: '을',   pos: 'JKO', start: 7, end: 9 },
//   { text: '먹',   pos: 'VV',  start: 10, end: 13 },
//   { text: '었',   pos: 'EP',  start: 10, end: 13 },
//   { text: '다',   pos: 'EF',  start: 10, end: 13 },
// ]

// 간단 토큰화 (표면형만)
const tokens = garu.tokenize('나는 학교에 간다');
console.log(tokens);
// ['나', '는', '학교', '에', '간다']

// 리소스 해제
garu.destroy();
```

### 커스텀 모델 로드

```typescript
// 커스텀 URL에서 모델 로드
const garu = await Garu.load({ modelUrl: '/my-models/custom.gmdl' });

// ArrayBuffer로 직접 전달
const response = await fetch('/my-models/custom.gmdl');
const modelData = await response.arrayBuffer();
const garu = await Garu.load({ modelData });
```

---

## 아키텍처

```
┌─────────────────────────────────────────┐
│              입력 텍스트                   │
└──────────────────┬──────────────────────┘
                   ↓
         ┌─────────┴──────────┐
         │  어절 캐시 조회     │  Hit → 즉시 반환 (10K entries, ~235KB)
         └─────────┬──────────┘
                   ↓ Miss
         ┌─────────┴──────────┐
         │  ASCII Run 감지     │  (영문 SL, 숫자 SN)
         └─────────┬──────────┘
                   ↓
┌──────────────────┴──────────────────────┐
│      래티스 구축 (Multi-POS FST)          │
│  ┌─ A: Content Word + Suffix           │
│  │   (다중 POS: 있→VA/VX, 하→VV/XSV)    │
│  ├─ B: Standalone Suffix               │
│  └─ C: Contracted Forms (했다→하+었+다)  │
└──────────────────┬──────────────────────┘
                   ↓
         ┌─────────┴──────────┐
         │  Trigram Viterbi    │  Sparse u8 (34KB)
         │  + Word Bigrams     │  Context bonus (6KB)
         └─────────┬──────────┘
                   ↓
┌──────────────────┴──────────────────────┐
│     형태소 시퀀스 + POS 태그 출력          │
└─────────────────────────────────────────┘
```

신경망 없이 사전과 통계만으로 동작합니다. BiLSTM 지식 증류, 자소 단위 시퀀스 라벨링 등 두 차례의 신경망 기반 접근이 브라우저 환경의 제약으로 실패한 뒤, 완전히 비신경망 아키텍처로 전환하였습니다.

전체 문장에 대해 래티스를 구축하고, 어절 캐시 항목을 저비용 후보 아크로 주입한 뒤 문장 수준 Viterbi 디코딩을 수행합니다. 이를 통해 "나는 하늘을 나는 새"에서 두 번째 "나는"을 VV+ETM(날다)으로 정확히 구분하는 등 어절 간 문맥 의존적 동형이의어를 해소합니다. 디코딩 결과에는 보조용언(VX), 접속조사(JC), 보격조사(JKC) 등의 문맥 의존적 품사를 교정하는 후처리 규칙이 적용됩니다.

코드북 데이터는 모두의 말뭉치(NIKL MP) 골드 어노테이션과 Kiwi 분석기 출력의 하이브리드로 추출합니다. 학습 파라미터는 0개이며, 모든 지식은 빈도 통계와 패턴 테이블로 표현됩니다. 자세한 연구 경과는 [기술 논문](docs/paper.md)을 참고하세요.

---

## API Reference

### `Garu.load(options?): Promise<Garu>`

WASM 모듈을 초기화하고 모델을 로드합니다.

| Option | Type | Description |
|---|---|---|
| `modelData` | `ArrayBuffer` | 모델 바이트를 직접 전달 |
| `modelUrl` | `string` | 커스텀 URL에서 모델 로드 |

옵션을 지정하지 않으면 npm 패키지에 포함된 기본 모델을 로드합니다.

### `garu.analyze(text, options?): AnalyzeResult | AnalyzeResult[]`

형태소 분석을 수행합니다.

- `text` -- 분석할 한국어 텍스트
- `options.topN` -- 1보다 크면 N-best 결과를 배열로 반환 (아직 완전히 지원되지 않으며, 결과가 더 적을 수 있음)

```typescript
interface AnalyzeResult {
  tokens: Token[];  // 형태소 토큰 배열
  score: number;    // 디코더 경로 점수
  elapsed: number;  // 처리 시간 (밀리초)
}

interface Token {
  text: string;     // 표면형
  pos: POS;         // 품사 태그 (세종 태그셋)
  start: number;    // 어절 시작 오프셋 (글자 단위)
  end: number;      // 어절 끝 오프셋 (글자 단위)
}
```

### `garu.tokenize(text): string[]`

표면형 문자열 배열을 반환합니다. 분절된 텍스트만 필요할 때 `analyze()`의 경량 대안으로 사용합니다.

### `garu.isLoaded(): boolean`

WASM 분석기가 초기화되어 사용 가능한 상태이면 `true`를 반환합니다.

### `garu.modelInfo(): ModelInfo`

로드된 모델의 메타데이터를 반환합니다.

```typescript
interface ModelInfo {
  version: string;  // 모델 버전
  size: number;     // 모델 크기 (바이트)
  accuracy: number; // 보고된 정확도 (0.953)
}
```

### `garu.destroy(): void`

WASM 인스턴스를 해제하고 메모리를 반환합니다. 호출 후 인스턴스를 재사용할 수 없습니다.

---

## 개발 가이드

### 요구 사항

- Rust toolchain (stable)
- `wasm-pack`
- Node.js >= 18
- Python >= 3.10 (학습 파이프라인용)

### 프로젝트 구조

```
crates/
  garu-core/     # Rust 코어: Codebook 분석기, FST 사전, 모델 로더
  garu-wasm/     # WASM 바인딩 (wasm-bindgen)
  garu-tools/    # CLI 도구 (FST 빌더)
js/              # TypeScript API 래퍼 (npm 패키지)
  models/        # 번들된 모델 파일 (base.gmdl)
  pkg/           # WASM 빌드 출력
training/        # 코드북 추출/빌드 파이프라인 (Python)
models/          # 컴파일된 모델 파일 (.gmdl)
docs/            # 기술 문서
```

### 빌드

```bash
# Rust 코어 라이브러리 빌드
cargo build --release

# WASM 패키지 빌드
wasm-pack build crates/garu-wasm --target web --out-dir ../../js/pkg

# TypeScript 래퍼 빌드
cd js && npx tsc

# 전체 빌드
cd js && npm run build
```

### 테스트

```bash
# Rust 테스트
cargo test

# JS/TS 테스트
cd js && npm test
```

---

## 기술 문서

아키텍처와 설계 결정에 대한 상세한 내용은 [기술 논문](docs/paper.md)을 참고하세요.

---

## 데이터 출처

이 프로젝트의 형태소 분석 모델은 다음 데이터를 기반으로 학습되었습니다:

- **모두의 말뭉치 형태 분석 말뭉치(v1.1)** — 국립국어원(National Institute of Korean Language) 제공. 모델에는 원본 텍스트가 포함되지 않으며, 빈도 통계 및 패턴만 사용됩니다.
- **세종 태그셋** — 42개 품사 태그 체계

## 라이선스

MIT
