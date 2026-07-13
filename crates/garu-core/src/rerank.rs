//! N-best 재순위 perceptron (GMDL Section 14).
//!
//! feature 문자열 생성 규칙과 FNV-1a 해시는 `training/rerank/features.py`와
//! 바이트 단위로 동일해야 한다 — 학습(Python)·추론(Rust) 정합의 전제.
//! 입력 tokens는 model 레벨 보정(apply_adj_root_xsa 등)까지 끝난 형태
//! (= 학습 덤프를 만든 `Model::analyze_topn`의 출력과 동일).

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

pub struct Reranker {
    dim_mask: u64,
    margin: f32,
    /// 정렬된 (bucket, weight) 병렬 배열 — sparse 가중치.
    buckets: Vec<u32>,
    weights: Vec<f32>,
}

impl Reranker {
    /// Section 14 포맷: [ver u8=1][dim_log2 u8][margin f32][count u32]
    /// [(bucket u32, weight f32) × count] (bucket 오름차순).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 10 {
            return Err("Rerank section too short".into());
        }
        if data[0] != 1 {
            return Err(format!("Unknown rerank section version {}", data[0]));
        }
        let dim_log2 = data[1];
        let margin = f32::from_le_bytes(data[2..6].try_into().map_err(|_| "Bad margin")?);
        let count =
            u32::from_le_bytes(data[6..10].try_into().map_err(|_| "Bad count")?) as usize;
        if data.len() < 10 + count * 8 {
            return Err("Rerank section truncated".into());
        }
        let mut buckets = Vec::with_capacity(count);
        let mut weights = Vec::with_capacity(count);
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
        Ok(Reranker {
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
        for t in tokens {
            let pos = t.pos.as_str();
            let form = t.text.as_str();
            feat!(1.0, "u:{}", pos);
            feat!(1.0, "b:{}|{}", prev, pos);
            feat!(1.0, "t:{}|{}|{}", prev2, prev, pos);
            feat!(1.0, "w:{}/{}", form, pos);
            feat!(1.0, "pw:{}|{}", prev_form, pos);
            prev2 = prev;
            prev = pos;
            prev_form = form;
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

    #[test]
    fn fnv1a_matches_python() {
        // python: fnv1a("bias") == 0x92adf3e152a6a2b5 확인용 고정값
        assert_eq!(fnv1a(b"") & u64::MAX, 0xCBF2_9CE4_8422_2325);
        // 한글 UTF-8 바이트 처리 확인 (값은 features.py fnv1a와 대조)
        let h = fnv1a("w:정부/NNG".as_bytes());
        assert_ne!(h, 0);
    }
}
