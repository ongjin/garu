# 가루 (Garu)

텍스트를 곱게 갈아주는 초경량 한국어 형태소 분석기.

서버 없이 브라우저에서 직접 실행되는 WebAssembly 기반 형태소 분석기로, 모바일 웹 환경에서도 실용적으로 동작합니다.

---

## 특징

- **초경량 모델** -- Codebook v3 기준 2.94MB, 모바일 웹에서도 즉시 로드
- **서버 불필요** -- WebAssembly로 브라우저에서 직접 실행
- **두 가지 백엔드** -- 용도에 따라 BiLSTM v2 또는 Codebook v3 선택
- **높은 정확도** -- Codebook v3 F1 91.8%, BiLSTM v2 F1 ~88%
- **오프라인 지원** -- 네트워크 없이도 완전한 형태소 분석 가능

---

## 비교

|  | Kiwi WASM | Garu BiLSTM v2 | Garu Codebook v3 |
|---|---|---|---|
| 모델 크기 | ~40MB | 4.65MB | **2.94MB** |
| 로드 시간 | 2-5초 | ~500ms | **~20ms** |
| 추론 방식 | BiLSTM | BiLSTM+CRF | Lattice+Viterbi |
| 학습 파라미터 | 수백만 | ~50만 | **0** |
| 모바일 웹 | 비실용적 | 지원 | **최적** |

---

## Quick Start

```bash
npm install garu
```

```typescript
import { Garu } from 'garu';

// Codebook v3 모델로 로드 (기본값)
const garu = await Garu.load();

// 형태소 분석
const result = garu.analyze('아버지가방에들어가셨다');
console.log(result.tokens);
// [
//   { text: '아버지', pos: 'NNG', start: 0, end: 3 },
//   { text: '가방',   pos: 'NNG', start: 3, end: 5 },
//   { text: '에',     pos: 'JKB', start: 5, end: 6 },
//   { text: '들어가', pos: 'VV',  start: 6, end: 9 },
//   { text: '시',     pos: 'EP',  start: 9, end: 10 },
//   { text: '었',     pos: 'EP',  start: 10, end: 11 },
//   { text: '다',     pos: 'EF',  start: 11, end: 12 },
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
// BiLSTM v2 모델 사용
const garu = await Garu.load({ modelUrl: '/models/base.gmdl' });

// ArrayBuffer로 직접 전달
const response = await fetch('/models/codebook.gmdl');
const modelData = await response.arrayBuffer();
const garu = await Garu.load({ modelData });
```

---

## 모델 선택 가이드

### Codebook v3 (`codebook.gmdl`, 2.94MB)

신경망 없이 사전 기반 래티스 구축과 trigram Viterbi 디코딩으로 동작합니다. 학습 파라미터가 0개이며 로드 시간이 ~20ms로 극히 짧습니다.

**적합한 경우:** 모바일 웹, 빠른 로드가 중요한 경우, 번들 크기에 민감한 프로젝트

### BiLSTM v2 (`base.gmdl`, 4.65MB)

INT8 양자화된 BiLSTM + CRF 디코더와 FST 사전을 사용합니다. 신경망 기반이므로 미등록어 처리에 강점이 있습니다.

**적합한 경우:** 데스크톱 웹, 정확도가 최우선인 경우, 다양한 도메인의 텍스트 처리

---

## Codebook v3 아키텍처

```
┌─────────────────────────────────────────┐
│              입력 텍스트                   │
└──────────────────┬──────────────────────┘
                   ↓
         ┌─────────┴──────────┐
         │  ASCII Run 감지     │  (영문 NNP, 숫자 SN)
         └─────────┬──────────┘
                   ↓
┌──────────────────┴──────────────────────┐
│           래티스 구축                      │
│  ┌─ A: Content Word + Suffix           │
│  ├─ B: Standalone Suffix               │
│  └─ C: Contracted Forms (했다→하+었+다)  │
└──────────────────┬──────────────────────┘
                   ↓
         ┌─────────┴──────────┐
         │  Trigram Viterbi    │  (42×42×42 POS costs)
         └─────────┬──────────┘
                   ↓
┌──────────────────┴──────────────────────┐
│     형태소 시퀀스 + POS 태그 출력          │
└─────────────────────────────────────────┘
```

입력 텍스트에서 먼저 ASCII 영역(영문, 숫자)을 분리한 뒤, 한글 구간에 대해 사전 기반으로 래티스를 구축합니다. 래티스의 각 경로는 내용어+접사, 단독 접사, 축약형 세 가지 패턴으로 생성됩니다. 최종적으로 42개 품사 태그의 trigram 비용 테이블을 이용한 Viterbi 디코딩으로 최적 경로를 선택합니다.

---

## API Reference

### `Garu.load(options?): Promise<Garu>`

WASM 모듈을 초기화하고 모델을 로드합니다.

| Option | Type | Description |
|---|---|---|
| `modelData` | `ArrayBuffer` | 모델 바이트를 직접 전달 |
| `modelUrl` | `string` | 커스텀 URL에서 모델 로드 |

옵션을 지정하지 않으면 기본 CDN에서 모델을 로드합니다.

### `garu.analyze(text, options?): AnalyzeResult | AnalyzeResult[]`

형태소 분석을 수행합니다.

- `text` -- 분석할 한국어 텍스트
- `options.topN` -- 1보다 크면 N-best 결과를 배열로 반환

```typescript
interface AnalyzeResult {
  tokens: Token[];  // 형태소 토큰 배열
  score: number;    // 디코더 경로 점수
  elapsed: number;  // 처리 시간 (밀리초)
}

interface Token {
  text: string;     // 표면형
  pos: POS;         // 품사 태그 (세종 태그셋)
  start: number;    // 입력 텍스트 내 시작 오프셋
  end: number;      // 입력 텍스트 내 끝 오프셋
  score?: number;   // 토큰 수준 점수
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
  accuracy: number; // 보고된 정확도
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
  garu-core/     # Rust 코어: 자모, 트라이, BiLSTM, CRF, Codebook, 모델 로더
  garu-wasm/     # WASM 바인딩 (wasm-bindgen)
  garu-tools/    # CLI 도구
js/              # TypeScript API 래퍼 (npm 패키지)
training/        # Python 학습 파이프라인
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

## 라이선스

MIT
