//! 탐색 전 아크 비용 보정 (GMDL Section 18).
//!
//! 래티스의 각 아크에 `scale × Σ w[feature]`를 더한 뒤 기존 Viterbi를 그대로 돌린다.
//! feature는 아크의 품사열·길이·형태소·어절 경계 위치를 FNV-1a로 해시한 버킷이다.
//! 학습은 `examples/search_train.rs`가 같은 함수를 써서 native로 수행하므로
//! Python 재구현이 없다.
//!
//! v2는 구어/문어 판별기(`Router`)를 함께 담아, 입력이 구어로 판별될 때만 보정한다.
//! 판별기의 문자 n-gram 해시는 `training/search/router.py`와 같아야 한다.

use crate::types::Pos;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
const MAGIC_V1: &[u8; 5] = b"GSCW\x01";
const MAGIC_V2: &[u8; 5] = b"GSCW\x02";
const ROUTER_HEADER_LEN: usize = 1 + 4 + 4 + 4;
const MAX_ROUTER_DIM_LOG2: u8 = 20;
const HEADER_LEN: usize = 5 + 1 + 4 + 4 + 4;
const MAX_LEN_BUCKET: usize = 8;
const MAX_MORPH_BUCKET: usize = 4;

/// 학습·추론 공통 해시 공간 크기.
pub const DIM_LOG2: u8 = 20;

struct Fnv(u64);

impl Fnv {
    fn new(tag: u8) -> Self {
        Fnv(FNV_OFFSET).byte(tag)
    }

    fn byte(mut self, b: u8) -> Self {
        self.0 ^= b as u64;
        self.0 = self.0.wrapping_mul(FNV_PRIME);
        self
    }

    fn bytes(self, data: &[u8]) -> Self {
        data.iter().fold(self, |h, &b| h.byte(b))
    }

    fn bucket(&self, mask: u64) -> u32 {
        (self.0 & mask) as u32
    }
}

/// 아크 하나의 feature 버킷을 `out`에 덧붙인다.
///
/// `at_word_start`/`at_word_end`는 아크가 어절 경계(공백·문장 끝)에 닿는지다.
pub fn arc_features(morphemes: &[(String, Pos)], chars: usize, at_word_start: bool,
                    at_word_end: bool, out: &mut Vec<u32>) {
    if morphemes.is_empty() {
        return;
    }
    let mask = (1u64 << DIM_LOG2) - 1;
    let pos_seq = morphemes.iter().enumerate().fold(Fnv::new(b'P'), |h, (i, (_, p))| {
        let h = if i > 0 { h.byte(b'+') } else { h };
        h.bytes(p.as_str().as_bytes())
    });
    out.push(pos_seq.bucket(mask));

    let shape = Fnv::new(b'L')
        .byte(chars.min(MAX_LEN_BUCKET) as u8)
        .byte(morphemes.len().min(MAX_MORPH_BUCKET) as u8);
    out.push(shape.bucket(mask));

    let first = morphemes[0].1.as_str().as_bytes();
    let last = morphemes[morphemes.len() - 1].1.as_str().as_bytes();
    let edge = Fnv::new(b'B').byte(at_word_start as u8).byte(at_word_end as u8)
        .bytes(first).byte(b':').bytes(last);
    out.push(edge.bucket(mask));

    let mut whole = Fnv::new(b'A');
    for (i, (form, pos)) in morphemes.iter().enumerate() {
        let lexical = Fnv::new(b'W').bytes(form.as_bytes()).byte(b'/').bytes(pos.as_str().as_bytes());
        out.push(lexical.bucket(mask));
        if i > 0 {
            whole = whole.byte(b'+');
        }
        whole = whole.bytes(form.as_bytes()).byte(b'/').bytes(pos.as_str().as_bytes());
    }
    out.push(whole.bucket(mask));
}

/// 문자 1~3-gram 나이브 베이즈 로그비. 점수가 임계값보다 크면 구어로 본다.
struct Router {
    mask: u64,
    qscale: f32,
    bias: f32,
    threshold: f32,
    weights: Vec<i8>,
}

impl Router {
    fn score(&self, text: &str) -> f64 {
        let chars: Vec<char> = std::iter::once('^').chain(text.trim().chars())
            .chain(std::iter::once('$')).collect();
        let mut features = Vec::with_capacity(chars.len() * 3);
        let mut buf = [0u8; 4];
        for n in 1..=3usize {
            for window in chars.windows(n) {
                let h = window.iter().fold(Fnv(FNV_OFFSET).byte(n as u8),
                    |h, c| h.bytes(c.encode_utf8(&mut buf).as_bytes()));
                features.push(h.bucket(self.mask));
            }
        }
        features.sort_unstable();
        features.dedup();
        let total: i64 = features.iter().map(|&f| self.weights[f as usize] as i64).sum();
        total as f64 * self.qscale as f64 + self.bias as f64
    }
}

pub struct SearchCost {
    scale: f32,
    /// 정렬된 (bucket, weight) 병렬 배열.
    buckets: Vec<u32>,
    weights: Vec<f32>,
    /// 버킷 존재 비트맵 — 대부분의 feature는 가중치가 없어 이진 탐색을 건너뛴다.
    present: Vec<u64>,
    router: Option<Router>,
}

impl SearchCost {
    /// v1: `[GSCW\x01][dim_log2 u8][scale f32][count u32][qscale f32]`
    ///     `[bucket delta LEB128 × count][weight i16 × count]`
    /// v2: v1 본문(magic만 `GSCW\x02`) 뒤에
    ///     `[router dim_log2 u8][qscale f32][bias f32][threshold f32][weight i8 × 2^dim]`
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < HEADER_LEN || (&data[..5] != MAGIC_V1 && &data[..5] != MAGIC_V2) {
            return Err("Unsupported search cost section".into());
        }
        let gated = &data[..5] == MAGIC_V2;
        if data[5] != DIM_LOG2 {
            return Err("Search cost dimension mismatch".into());
        }
        let f32_at = |at: usize| f32::from_le_bytes(data[at..at + 4].try_into().unwrap());
        let scale = f32_at(6);
        let count = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
        let qscale = f32_at(14);
        if !scale.is_finite() || !qscale.is_finite() {
            return Err("Invalid search cost scale".into());
        }

        let mut pos = HEADER_LEN;
        let mut buckets = Vec::with_capacity(count);
        let mut acc: u64 = 0;
        for _ in 0..count {
            let mut delta: u64 = 0;
            let mut shift = 0u32;
            loop {
                if pos >= data.len() || shift > 35 {
                    return Err("Search cost section truncated".into());
                }
                let b = data[pos];
                pos += 1;
                delta |= ((b & 0x7F) as u64) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            if delta == 0 && !buckets.is_empty() {
                return Err("Search cost buckets not strictly sorted".into());
            }
            acc += delta;
            if acc >= (1u64 << DIM_LOG2) {
                return Err("Search cost bucket out of range".into());
            }
            buckets.push(acc as u32);
        }
        let weights_end = pos + count * 2;
        if data.len() < weights_end {
            return Err("Search cost section truncated".into());
        }
        let weights = data[pos..weights_end].chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 * qscale)
            .collect();
        let router = if gated { Some(Self::router(&data[weights_end..])?) } else { None };
        if !gated && data.len() != weights_end {
            return Err("Search cost section size mismatch".into());
        }
        let mut present = vec![0u64; (1usize << DIM_LOG2) / 64];
        for &b in &buckets {
            present[b as usize / 64] |= 1 << (b % 64);
        }
        Ok(SearchCost { scale, buckets, weights, present, router })
    }

    fn router(data: &[u8]) -> Result<Router, String> {
        if data.len() < ROUTER_HEADER_LEN || data[0] > MAX_ROUTER_DIM_LOG2 {
            return Err("Invalid search router".into());
        }
        let f32_at = |at: usize| f32::from_le_bytes(data[at..at + 4].try_into().unwrap());
        let (qscale, bias, threshold) = (f32_at(1), f32_at(5), f32_at(9));
        if ![qscale, bias, threshold].iter().all(|v| v.is_finite()) {
            return Err("Invalid search router scale".into());
        }
        let dim = 1usize << data[0];
        if data.len() != ROUTER_HEADER_LEN + dim {
            return Err("Search router size mismatch".into());
        }
        let weights = data[ROUTER_HEADER_LEN..].iter().map(|&b| b as i8).collect();
        Ok(Router { mask: dim as u64 - 1, qscale, bias, threshold, weights })
    }

    /// 판별기가 없거나 입력을 구어로 판별하면 true.
    pub fn applies(&self, text: &str) -> bool {
        self.router.as_ref().map_or(true, |r| r.score(text) > r.threshold as f64)
    }

    pub fn delta(&self, features: &[u32]) -> f32 {
        let sum: f32 = features.iter()
            .filter(|&&f| self.present[f as usize / 64] >> (f % 64) & 1 == 1)
            .filter_map(|f| self.buckets.binary_search(f).ok().map(|i| self.weights[i]))
            .sum();
        self.scale * sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(scale: f32, entries: &[(u32, i16)], qscale: f32) -> Vec<u8> {
        let mut out = MAGIC_V1.to_vec();
        out.push(DIM_LOG2);
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        out.extend_from_slice(&qscale.to_le_bytes());
        let mut prev = 0u32;
        for &(b, _) in entries {
            let mut d = b - prev;
            prev = b;
            loop {
                let byte = (d & 0x7F) as u8;
                d >>= 7;
                if d == 0 { out.push(byte); break; }
                out.push(byte | 0x80);
            }
        }
        for &(_, w) in entries {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    #[test]
    fn features_are_deterministic_and_lexical() {
        let m = vec![("하늘".to_string(), Pos::NNG), ("을".to_string(), Pos::JKO)];
        let (mut a, mut b) = (Vec::new(), Vec::new());
        arc_features(&m, 3, true, true, &mut a);
        arc_features(&m, 3, true, true, &mut b);
        assert_eq!(a, b);
        assert_eq!(a.len(), 6);
        let mut c = Vec::new();
        arc_features(&m, 3, true, false, &mut c);
        assert_eq!(a[..2], c[..2]);
        assert_ne!(a[2], c[2], "어절 경계 feature가 달라야 함");
        assert!(a.iter().all(|&f| f < (1 << DIM_LOG2)));
    }

    #[test]
    fn delta_sums_matching_weights() {
        let cost = SearchCost::from_bytes(&blob(0.5, &[(3, 100), (70_000, -50)], 0.01)).unwrap();
        assert!((cost.delta(&[3, 70_000, 9]) - 0.5 * (1.0 - 0.5)).abs() < 1e-6);
        assert_eq!(cost.delta(&[9]), 0.0);
    }

    fn gated(base: &[u8], dim_log2: u8, weights: &[i8], qscale: f32, bias: f32, threshold: f32) -> Vec<u8> {
        let mut out = base.to_vec();
        out[..5].copy_from_slice(MAGIC_V2);
        out.push(dim_log2);
        for v in [qscale, bias, threshold] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend(weights.iter().map(|&w| w as u8));
        out
    }

    #[test]
    fn router_matches_python_hashing() {
        // training/search/router.py score("가나", w, 0.5, -1.0, 4), w[i] = i - 8 → -8.0
        // (features [0, 1, 2, 6, 7, 8, 9, 12, 13]: (58 - 72) × 0.5 - 1)
        let weights: Vec<i8> = (0..16).map(|i| i as i8 - 8).collect();
        let base = blob(1.0, &[(3, 1)], 0.1);
        let cost = SearchCost::from_bytes(&gated(&base, 4, &weights, 0.5, -1.0, 0.0)).unwrap();
        let router = cost.router.as_ref().unwrap();
        assert_eq!(router.score("가나"), -8.0);
        assert_eq!(router.score("  가나 "), -8.0, "앞뒤 공백은 무시");
        assert!(!cost.applies("가나"));
        let lenient = SearchCost::from_bytes(&gated(&base, 4, &weights, 0.5, -1.0, -9.0)).unwrap();
        assert!(lenient.applies("가나"));
        assert!(SearchCost::from_bytes(&base).unwrap().applies("가나"), "v1은 항상 적용");
    }

    #[test]
    fn rejects_malformed_router() {
        let base = blob(1.0, &[(3, 1)], 0.1);
        let weights = vec![0i8; 16];
        let good = gated(&base, 4, &weights, 0.5, 0.0, 0.0);
        assert!(SearchCost::from_bytes(&good[..good.len() - 1]).is_err());
        assert!(SearchCost::from_bytes(&gated(&base, 4, &weights[..8], 0.5, 0.0, 0.0)).is_err());
        assert!(SearchCost::from_bytes(&gated(&base, 4, &weights, f32::NAN, 0.0, 0.0)).is_err());
        let mut trailing = base.clone();
        trailing.push(0);
        assert!(SearchCost::from_bytes(&trailing).is_err());
    }

    #[test]
    fn rejects_malformed_sections() {
        let good = blob(1.0, &[(3, 1), (5, 2)], 0.1);
        assert!(SearchCost::from_bytes(&good[..good.len() - 1]).is_err());
        assert!(SearchCost::from_bytes(&blob(1.0, &[(3, 1), (3, 2)], 0.1)).is_err());
        let mut wrong_dim = good.clone();
        wrong_dim[5] = 19;
        assert!(SearchCost::from_bytes(&wrong_dim).is_err());
        assert!(SearchCost::from_bytes(b"GSCW\x02").is_err());
    }
}
