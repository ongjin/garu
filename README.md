# 가루 (Garu)

텍스트를 곱게 갈아주는 초경량 한국어 형태소 분석기.

서버 없이 브라우저에서 직접 실행되는 WebAssembly 기반 형태소 분석기로, 모바일 웹 환경에서도 실용적으로 동작합니다.

---

## 특징

- **초경량 모델** -- 2.1MB (npm 패키지에 포함, CDN 불필요)
- **서버 불필요** -- WebAssembly로 브라우저에서 직접 실행
- **신경망 없음** -- 코드북 + Trigram Viterbi + 어절 캐시, 학습 파라미터 0개
- **높은 정확도** -- F1 93.8% (NIKL MP 기준, Kiwi 87.9% 상회)
- **오프라인 지원** -- 네트워크 없이도 완전한 형태소 분석 가능

---

## 비교

|  | Kiwi | MeCab-ko | Garu |
|---|---|---|---|
| 모델 크기 | ~40MB | ~50MB | **2.1MB** |
| npm 패키지 | 없음 | 없음 | **포함** |
| F1 (NIKL MP) | 87.9% | ~85% | **93.8%** |
| 추론 방식 | BiLSTM+CRF | CRF+사전 | **Lattice+Viterbi** |
| 학습 파라미터 | 수백만 | 수십만 | **0** |
| 브라우저 실행 | 비실용적 | 불가 | **지원** |
| 모바일 웹 | 비실용적 | 불가 | **최적** |

---

> **Note**: npm 패키지는 아직 배포 전입니다. 현재는 소스에서 직접 빌드하여 사용할 수 있습니다.

## Quick Start

```bash
npm install garu  # (배포 후 사용 가능)
```

```typescript
import { Garu } from 'garu';

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
         │  어절 캐시 조회     │  Hit → 즉시 반환 (5K entries, 171KB)
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

어절 캐시에 등록된 어절은 비터비 디코딩 없이 즉시 결과를 반환합니다. 캐시에 없는 어절은 FST 사전(다중 POS 지원)과 접미사 코드북으로 래티스를 구축하고, 희소 트라이그램 비용 테이블 + 단어 수준 bigram 보정을 이용한 Viterbi 디코딩으로 최적 경로를 선택합니다.

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
  accuracy: number; // 보고된 정확도 (0.938)
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

## 라이선스

MIT
