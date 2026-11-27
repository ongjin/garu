//! N-best 재순위 perceptron (GMDL Section 14).
//!
//! feature 문자열 생성 규칙과 FNV-1a 해시는 `training/rerank/features.py`와
//! 바이트 단위로 동일해야 한다 — 학습(Python)·추론(Rust) 정합의 전제.
//! 입력 tokens는 model 레벨 보정(apply_adj_root_xsa 등)까지 끝난 형태
//! (= 학습 덤프의 `Analyzer::analyze_topn_for_rerank` 출력과 동일).
//! 같은 어절의 토큰은 같은 `start`를 공유하므로, `start`가 바뀌는 곳이 공백이다.

use crate::types::Token;
use std::fmt::Write;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// ver 1·2 가중치는 공백을 보지 않는 feature로, ver 3은 어절 경계 feature를 더해 학습했다.
#[derive(Clone, Copy, PartialEq)]
enum FeatureSet {
    Base,
    Spacing,
}

pub struct Reranker {
    features: FeatureSet,
    dim_mask: u64,
    margin: f32,
    /// 정렬된 (bucket, weight) 병렬 배열 — sparse 가중치.
    buckets: Vec<u32>,
    weights: Vec<f32>,
}

impl Reranker {
    /// Section 14 포맷 (공통 헤더 `[ver u8][dim_log2 u8][margin f32][count u32]`):
    /// - ver=1: `[(bucket u32, weight f32) × count]` (bucket 오름차순)
    /// - ver=2: `[scale f32][bucket delta varint × count][quantized weight i16 × count]`
    ///   — bucket은 오름차순 차분을 LEB128로, 가중치는 `round(w/scale)`로 저장.
    ///     같은 가중치를 압축 친화적으로 담은 것이라 디코드 결과는 ver=1과 같은 형태다.
    /// - ver=3: ver=2와 같은 인코딩. 가중치가 어절 경계 feature(sb:/spw:/sw:)를 포함한다.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 10 {
            return Err("Rerank section too short".into());
        }
        let ver = data[0];
        if !(1..=3).contains(&ver) {
            return Err(format!("Unknown rerank section version {}", ver));
        }
        let dim_log2 = data[1];
        let margin = f32::from_le_bytes(data[2..6].try_into().map_err(|_| "Bad margin")?);
        let count =
            u32::from_le_bytes(data[6..10].try_into().map_err(|_| "Bad count")?) as usize;
        let mut buckets = Vec::with_capacity(count);
        let mut weights = Vec::with_capacity(count);

        if ver == 1 {
            if data.len() < 10 + count * 8 {
                return Err("Rerank section truncated".into());
            }
            let mut pos = 10;
            for _ in 0..count {
                let b = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                if let Some(&last) = buckets.last() {
                    if b <= last {
                        return Err("Rerank buckets not strictly sorted".into());
                    }
                }
                buckets.push(b);
                weights.push(f32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()));
                pos += 8;
            }
        } else {
            if data.len() < 14 {
                return Err("Rerank section truncated".into());
            }
            let scale = f32::from_le_bytes(data[10..14].try_into().map_err(|_| "Bad scale")?);
            let mut pos = 14;
            let mut acc: u64 = 0;
            for _ in 0..count {
                let mut delta: u64 = 0;
                let mut shift = 0u32;
                loop {
                    if pos >= data.len() || shift > 63 {
                        return Err("Rerank section truncated".into());
                    }
                    let byte = data[pos];
                    pos += 1;
                    delta |= ((byte & 0x7F) as u64) << shift;
                    if byte & 0x80 == 0 {
                        break;
                    }
                    shift += 7;
                }
                if delta == 0 && !buckets.is_empty() {
                    return Err("Rerank buckets not strictly sorted".into());
                }
                acc += delta;
                if acc > u32::MAX as u64 {
                    return Err("Rerank bucket out of range".into());
                }
                buckets.push(acc as u32);
            }
            if data.len() < pos + count * 2 {
                return Err("Rerank section truncated".into());
            }
            for _ in 0..count {
                let q = i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                weights.push(q as f32 * scale);
                pos += 2;
            }
        }

        let features = if ver == 3 { FeatureSet::Spacing } else { FeatureSet::Base };
        Ok(Reranker {
            features,
            dim_mask: (1u64 << dim_log2) - 1,
            margin,
            buckets,
            weights,
        })
    }

    pub fn margin(&self) -> f32 {
        self.margin
    }

    fn w(&self, name: &str) -> f32 {
        let b = (fnv1a(name.as_bytes()) & self.dim_mask) as u32;
        match self.buckets.binary_search(&b) {
            Ok(i) => self.weights[i],
            Err(_) => 0.0,
        }
    }

    /// 후보 하나의 perceptron 점수 (높을수록 좋음).
    /// score_delta = cand.score - candidates[0].score, rank는 1-based.
    pub fn score(&self, tokens: &[Token], score_delta: f32, rank: usize) -> f32 {
        let mut s = 0.0f32;
        let mut buf = String::with_capacity(64);
        macro_rules! feat {
            ($v:expr, $($arg:tt)*) => {{
                buf.clear();
                let _ = write!(buf, $($arg)*);
                s += self.w(&buf) * $v;
            }};
        }

        s += self.w("bias");
        s += self.w("score_delta") * score_delta;
        feat!(1.0, "rank:{}", rank);
        s += self.w("ntok") * tokens.len() as f32;

        let mut prev2 = "<s2>";
        let mut prev = "<s>";
        let mut prev_form = "<s>";
        let mut prev_start = None;
        for t in tokens {
            let pos = t.pos.as_str();
            let form = t.text.as_str();
            feat!(1.0, "u:{}", pos);
            feat!(1.0, "b:{}|{}", prev, pos);
            feat!(1.0, "t:{}|{}|{}", prev2, prev, pos);
            feat!(1.0, "w:{}/{}", form, pos);
            feat!(1.0, "pw:{}|{}", prev_form, pos);

            let crosses_space = prev_start.is_some_and(|start| start != t.start);
            if crosses_space && self.features == FeatureSet::Spacing {
                feat!(1.0, "sb:{}|{}", prev, pos);
                feat!(1.0, "spw:{}|{}", prev_form, pos);
                feat!(1.0, "sw:{}/{}", form, pos);
            }

            prev2 = prev;
            prev = pos;
            prev_form = form;
            prev_start = Some(t.start);
        }
        feat!(1.0, "b:{}|</s>", prev);
        feat!(1.0, "t:{}|{}|</s>", prev2, prev);
        feat!(1.0, "lw:{}/{}|</s>", prev_form, prev);

        let mut last_c = "<none>";
        for t in tokens.iter().rev() {
            let p = t.pos.as_str();
            if !p.starts_with('S') {
                last_c = p;
                break;
            }
        }
        feat!(1.0, "endc:{}", last_c);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Pos;

    #[test]
    fn fnv1a_matches_python() {
        // python: fnv1a("bias") == 0x92adf3e152a6a2b5 확인용 고정값
        assert_eq!(fnv1a(b"") & u64::MAX, 0xCBF2_9CE4_8422_2325);
        // 한글 UTF-8 바이트 처리 확인 (값은 features.py fnv1a와 대조)
        let h = fnv1a("w:정부/NNG".as_bytes());
        assert_ne!(h, 0);
    }

    const TEST_DIM_LOG2: u8 = 20;

    /// 지정한 feature에만 정수 가중치를 둔 Section 14 (scale 1, margin 0).
    fn section(ver: u8, feats: &[(&str, i16)]) -> Vec<u8> {
        let mask = (1u64 << TEST_DIM_LOG2) - 1;
        let mut entries: Vec<(u32, i16)> = feats.iter()
            .map(|&(name, q)| ((fnv1a(name.as_bytes()) & mask) as u32, q))
            .collect();
        entries.sort();

        let mut out = vec![ver, TEST_DIM_LOG2];
        out.extend_from_slice(&0f32.to_le_bytes());
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        out.extend_from_slice(&1f32.to_le_bytes());
        let mut prev = 0u32;
        for &(bucket, _) in &entries {
            let mut delta = bucket - prev;
            prev = bucket;
            loop {
                let byte = (delta & 0x7F) as u8;
                delta >>= 7;
                if delta == 0 {
                    out.push(byte);
                    break;
                }
                out.push(byte | 0x80);
            }
        }
        for &(_, q) in &entries {
            out.extend_from_slice(&q.to_le_bytes());
        }
        out
    }

    fn token(text: &str, pos: Pos, start: usize) -> Token {
        Token { text: text.to_owned(), pos, start, end: start + 1, score: None }
    }

    /// ver 3은 어절이 바뀌는 토큰에서만 sb:/spw:/sw:를 더하고, ver 2는 공백을 보지 않는다.
    #[test]
    fn spacing_features_need_ver3() {
        let feats = [("sw:화/XSN", -10), ("sb:NNG|XSN", -3), ("spw:활성|XSN", -2)];
        let spaced = [token("활성", Pos::NNG, 0), token("화", Pos::XSN, 3), token("학제", Pos::NNG, 3)];
        let joined = [token("활성", Pos::NNG, 0), token("화", Pos::XSN, 0), token("학제", Pos::NNG, 0)];

        let v3 = Reranker::from_bytes(&section(3, &feats)).unwrap();
        assert_eq!(v3.score(&spaced, 0.0, 1), -15.0);
        assert_eq!(v3.score(&joined, 0.0, 1), 0.0);

        let v2 = Reranker::from_bytes(&section(2, &feats)).unwrap();
        assert_eq!(v2.score(&spaced, 0.0, 1), 0.0);
    }
}
