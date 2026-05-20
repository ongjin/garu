//! Lightweight 2-layer 1D CNN for syllable-level POS tagging.
//!
//! Architecture: Embedding → Conv1D×3 (k=3,5,9) → ReLU → Conv1D×2 (k=3,7) → ReLU → FC
//! All weights are int8 quantized with per-tensor symmetric scaling.
//! BatchNorm is fused into conv1 weights at export time.

use std::collections::HashMap;

/// Decompress brotli-compressed bytes.
pub(crate) fn decompress_brotli(data: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut decoder = brotli_decompressor::Decompressor::new(data, 4096);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Minimal JSON string-array parser for `["a","b",...]` payloads.
/// Supports `\"`, `\\`, `\/` escapes. No unicode escapes (Korean is stored as UTF-8).
fn parse_json_string_array(s: &str) -> Result<Vec<String>, String> {
    let bytes = s.as_bytes();
    let mut i = 0;
    // Skip leading whitespace
    while i < bytes.len() && bytes[i].is_ascii_whitespace() { i += 1; }
    if i >= bytes.len() || bytes[i] != b'[' {
        return Err("expected '['".into());
    }
    i += 1;
    let mut out = Vec::new();
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() { i += 1; }
        if i >= bytes.len() { return Err("unexpected EOF".into()); }
        if bytes[i] == b']' { return Ok(out); }
        if bytes[i] != b'"' { return Err(format!("expected '\"' at {}", i)); }
        i += 1;
        let mut buf = Vec::new();
        while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] == b'\\' {
                i += 1;
                if i >= bytes.len() { return Err("bad escape".into()); }
                match bytes[i] {
                    b'"' => buf.push(b'"'),
                    b'\\' => buf.push(b'\\'),
                    b'/' => buf.push(b'/'),
                    c => return Err(format!("unsupported escape \\{}", c as char)),
                }
                i += 1;
            } else {
                buf.push(bytes[i]);
                i += 1;
            }
        }
        if i >= bytes.len() { return Err("unterminated string".into()); }
        i += 1; // skip closing "
        out.push(String::from_utf8(buf).map_err(|_| "bad UTF-8")?);
        while i < bytes.len() && bytes[i].is_ascii_whitespace() { i += 1; }
        if i >= bytes.len() { return Err("unexpected EOF".into()); }
        match bytes[i] {
            b',' => { i += 1; }
            b']' => return Ok(out),
            c => return Err(format!("expected ',' or ']' at {} got {}", i, c as char)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_json_string_array;
    #[test] fn empty() { assert_eq!(parse_json_string_array("[]").unwrap(), Vec::<String>::new()); }
    #[test] fn one() { assert_eq!(parse_json_string_array("[\"가\"]").unwrap(), vec!["가"]); }
    #[test] fn many() {
        assert_eq!(parse_json_string_array("[\"가\",\"나\",\"다\"]").unwrap(), vec!["가","나","다"]);
    }
    #[test] fn escapes() {
        assert_eq!(parse_json_string_array("[\"\\\"\",\"\\\\\",\"\\/\"]").unwrap(), vec!["\"","\\","/"]);
    }
    #[test] fn whitespace() {
        assert_eq!(parse_json_string_array(" [ \"a\" , \"b\" ] ").unwrap(), vec!["a","b"]);
    }
}

/// Int8 quantized tensor with scale factor (used only during loading).
struct QTensor {
    data: Vec<i8>,
    scale: f32,
}

impl QTensor {
    /// Dequantize as flat f32 buffer (preserves original layout).
    fn dequant_flat(&self) -> Vec<f32> {
        self.data.iter().map(|&v| v as f32 * self.scale).collect()
    }

    /// Dequantize conv weight `[out_ch, in_ch, kernel_size]` and transpose to
    /// `[out_ch, kernel_size, in_ch]` so that the inner `in_ch` dimension is
    /// contiguous — required for SIMD load of consecutive lanes.
    fn dequant_conv_transposed(
        &self,
        out_ch: usize,
        in_ch: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; out_ch * kernel_size * in_ch];
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                for k in 0..kernel_size {
                    let src = oc * in_ch * kernel_size + ic * kernel_size + k;
                    let dst = oc * kernel_size * in_ch + k * in_ch + ic;
                    out[dst] = self.data[src] as f32 * self.scale;
                }
            }
        }
        out
    }
}

/// CNN2 model for syllable-level BIO+POS tagging.
///
/// At load time all int8 weights are dequantized to f32, and conv weights are
/// transposed from `[out_ch, in_ch, kernel_size]` to `[out_ch, kernel_size, in_ch]`
/// so the inner channel dimension is contiguous (SIMD-friendly).
pub struct Cnn2 {
    embed_dim: usize,
    hidden: usize,
    num_labels: usize,
    vocab_size: usize,
    // Embedding [vocab_size, embed_dim]
    emb_w: Vec<f32>,
    // Layer 1 conv weights [oc, ks, ic] + biases
    c3_w: Vec<f32>,
    c3_bias: Vec<f32>,
    c5_w: Vec<f32>,
    c5_bias: Vec<f32>,
    c9_w: Vec<f32>,
    c9_bias: Vec<f32>,
    // Layer 2 conv weights [oc, ks, ic] + biases
    c2a_w: Vec<f32>,
    c2a_bias: Vec<f32>,
    c2b_w: Vec<f32>,
    c2b_bias: Vec<f32>,
    // FC [num_labels, in_features]
    fc_w: Vec<f32>,
    fc_bias: Vec<f32>,
    // Vocabularies
    syl_to_idx: HashMap<char, usize>,
    labels: Vec<String>,
}

impl Cnn2 {
    /// Parse CNN2 binary format. Accepts raw or brotli-compressed input.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let data = if data.len() >= 4 && &data[0..4] == b"CNN2" {
            data.to_vec()
        } else {
            decompress_brotli(data).map_err(|e| format!("CNN brotli decompress failed: {}", e))?
        };

        if data.len() < 13 || &data[0..4] != b"CNN2" {
            return Err("Invalid CNN2 magic".into());
        }
        let _version = data[4];
        let embed_dim = u16::from_le_bytes(data[5..7].try_into().unwrap()) as usize;
        let hidden = u16::from_le_bytes(data[7..9].try_into().unwrap()) as usize;
        let num_labels = u16::from_le_bytes(data[9..11].try_into().unwrap()) as usize;
        let vocab_size = u16::from_le_bytes(data[11..13].try_into().unwrap()) as usize;

        let mut pos = 13;

        // Helper to read f32
        let read_f32 = |p: &mut usize| -> f32 {
            let v = f32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
            *p += 4;
            v
        };

        // Helper to read int8 tensor
        let read_qtensor = |p: &mut usize, shape: Vec<usize>| -> QTensor {
            let scale = read_f32(p);
            let n: usize = shape.iter().product();
            let raw = &data[*p..*p + n];
            let int8_data: Vec<i8> = raw.iter().map(|&b| b as i8).collect();
            *p += n;
            QTensor { data: int8_data, scale }
        };

        // Helper to read f32 bias
        let read_bias = |p: &mut usize, n: usize| -> Vec<f32> {
            let mut bias = Vec::with_capacity(n);
            for _ in 0..n {
                bias.push(read_f32(p));
            }
            bias
        };

        // 1. Embedding
        let emb = read_qtensor(&mut pos, vec![vocab_size, embed_dim]);

        // 2. Conv layers
        let c3 = read_qtensor(&mut pos, vec![hidden, embed_dim, 3]);
        let c3_bias = read_bias(&mut pos, hidden);
        let c5 = read_qtensor(&mut pos, vec![hidden, embed_dim, 5]);
        let c5_bias = read_bias(&mut pos, hidden);
        let c9 = read_qtensor(&mut pos, vec![hidden, embed_dim, 9]);
        let c9_bias = read_bias(&mut pos, hidden);

        let c2a = read_qtensor(&mut pos, vec![hidden, hidden * 3, 3]);
        let c2a_bias = read_bias(&mut pos, hidden);
        let c2b = read_qtensor(&mut pos, vec![hidden, hidden * 3, 7]);
        let c2b_bias = read_bias(&mut pos, hidden);

        // 3. FC
        let fc = read_qtensor(&mut pos, vec![num_labels, hidden * 2]);
        let fc_bias = read_bias(&mut pos, num_labels);

        // 4. Vocabularies
        let syl_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let syl_json = std::str::from_utf8(&data[pos..pos + syl_len])
            .map_err(|_| "Bad UTF-8 in syllable vocab")?;
        pos += syl_len;

        let lab_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let lab_json = std::str::from_utf8(&data[pos..pos + lab_len])
            .map_err(|_| "Bad UTF-8 in label vocab")?;

        // Parse syllable vocab → char→index map
        let syls = parse_json_string_array(syl_json)
            .map_err(|e| format!("Bad syllable vocab JSON: {}", e))?;
        let mut syl_to_idx = HashMap::with_capacity(syls.len());
        for (i, s) in syls.iter().enumerate() {
            if let Some(ch) = s.chars().next() {
                syl_to_idx.insert(ch, i);
            }
        }

        let labels = parse_json_string_array(lab_json)
            .map_err(|e| format!("Bad label vocab JSON: {}", e))?;

        // Dequantize all weights to f32 once, dropping int8 buffers.
        // Conv weights are transposed to [oc, ks, ic] for SIMD-friendly access.
        let h3 = hidden * 3;
        let emb_w = emb.dequant_flat();
        let c3_w = c3.dequant_conv_transposed(hidden, embed_dim, 3);
        let c5_w = c5.dequant_conv_transposed(hidden, embed_dim, 5);
        let c9_w = c9.dequant_conv_transposed(hidden, embed_dim, 9);
        let c2a_w = c2a.dequant_conv_transposed(hidden, h3, 3);
        let c2b_w = c2b.dequant_conv_transposed(hidden, h3, 7);
        let fc_w = fc.dequant_flat();

        Ok(Cnn2 {
            embed_dim, hidden, num_labels, vocab_size,
            emb_w,
            c3_w, c3_bias, c5_w, c5_bias, c9_w, c9_bias,
            c2a_w, c2a_bias, c2b_w, c2b_bias,
            fc_w, fc_bias,
            syl_to_idx, labels,
        })
    }

    /// Run inference on a text string. Returns Vec<(char, label_str, confidence)>.
    pub fn predict(&self, text: &str) -> Vec<(char, &str, f32)> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return vec![];
        }

        // Map chars to indices
        let indices: Vec<usize> = chars.iter()
            .map(|&ch| *self.syl_to_idx.get(&ch).unwrap_or(&1)) // 1 = <UNK>
            .collect();

        // 1. Embedding lookup (pre-dequantized at load time)
        let mut emb_out = vec![0.0f32; n * self.embed_dim];
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.vocab_size {
                let base = idx * self.embed_dim;
                emb_out[i * self.embed_dim..(i + 1) * self.embed_dim]
                    .copy_from_slice(&self.emb_w[base..base + self.embed_dim]);
            }
        }

        // 2. Layer 1: three parallel conv1d + ReLU
        let h = self.hidden;
        let mut l1_out = vec![0.0f32; n * h * 3];

        conv1d_relu(&emb_out, n, self.embed_dim,
                    &self.c3_w, &self.c3_bias, 3, &mut l1_out, 0, h);
        conv1d_relu(&emb_out, n, self.embed_dim,
                    &self.c5_w, &self.c5_bias, 5, &mut l1_out, h, h);
        conv1d_relu(&emb_out, n, self.embed_dim,
                    &self.c9_w, &self.c9_bias, 9, &mut l1_out, h * 2, h);

        // 3. Layer 2: two parallel conv1d + ReLU
        let mut l2_out = vec![0.0f32; n * h * 2];

        conv1d_relu(&l1_out, n, h * 3,
                    &self.c2a_w, &self.c2a_bias, 3, &mut l2_out, 0, h);
        conv1d_relu(&l1_out, n, h * 3,
                    &self.c2b_w, &self.c2b_bias, 7, &mut l2_out, h, h);

        // 4. FC layer
        let mut logits = vec![0.0f32; n * self.num_labels];
        fc_forward(&l2_out, n, h * 2, &self.fc_w, &self.fc_bias, self.num_labels, &mut logits);

        // 5. Argmax + softmax confidence
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * self.num_labels;

            // Find argmax
            let mut best_j = 0;
            let mut best_v = logits[base];
            for j in 1..self.num_labels {
                if logits[base + j] > best_v {
                    best_v = logits[base + j];
                    best_j = j;
                }
            }

            // Softmax confidence for best label
            let max_logit = best_v;
            let mut sum_exp = 0.0f32;
            for j in 0..self.num_labels {
                sum_exp += (logits[base + j] - max_logit).exp();
            }
            let confidence = 1.0 / sum_exp; // exp(0) / sum = 1/sum

            let label = if best_j < self.labels.len() {
                &self.labels[best_j]
            } else {
                "O"
            };
            result.push((chars[i], label, confidence));
        }

        result
    }

    /// Get label string by index.
    pub fn label(&self, idx: usize) -> &str {
        self.labels.get(idx).map(|s| s.as_str()).unwrap_or("O")
    }

    /// Get number of labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
}

// -----------------------------------------------------------------------------
// Conv1D / FC kernels — f32 weights, ic-contiguous layout, SIMD on wasm32
// -----------------------------------------------------------------------------

/// Dot product of two equal-length f32 slices. SIMD-accelerated on wasm32.
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use std::arch::wasm32::*;
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut acc = f32x4_splat(0.0);
        let mut i = 0;
        while i + 4 <= n {
            let av = v128_load(a.as_ptr().add(i) as *const v128);
            let bv = v128_load(b.as_ptr().add(i) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(av, bv));
            i += 4;
        }
        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Conv1D with f32 weights `[out_ch, kernel_size, in_ch]` (ic-contiguous) + bias + ReLU.
fn conv1d_relu(
    input: &[f32],
    seq_len: usize,
    in_ch: usize,
    weight: &[f32],
    bias: &[f32],
    kernel_size: usize,
    output: &mut [f32],
    out_offset: usize,
    out_ch: usize,
) {
    let pad = kernel_size / 2;
    let total_out_ch = output.len() / seq_len;
    let w_stride_oc = kernel_size * in_ch;

    for t in 0..seq_len {
        for oc in 0..out_ch {
            let mut sum = bias[oc];
            let w_base = oc * w_stride_oc;
            for k in 0..kernel_size {
                let t_in = t as isize + k as isize - pad as isize;
                if t_in < 0 || t_in >= seq_len as isize {
                    continue;
                }
                let t_in = t_in as usize;
                let w_slice = &weight[w_base + k * in_ch..w_base + (k + 1) * in_ch];
                let in_slice = &input[t_in * in_ch..(t_in + 1) * in_ch];
                sum += dot_f32(w_slice, in_slice);
            }
            output[t * total_out_ch + out_offset + oc] = sum.max(0.0);
        }
    }
}

/// FC layer with f32 weights `[num_labels, in_features]` + bias.
fn fc_forward(
    input: &[f32],
    seq_len: usize,
    in_features: usize,
    weight: &[f32],
    bias: &[f32],
    num_labels: usize,
    output: &mut [f32],
) {
    for t in 0..seq_len {
        let in_slice = &input[t * in_features..(t + 1) * in_features];
        for j in 0..num_labels {
            let w_slice = &weight[j * in_features..(j + 1) * in_features];
            output[t * num_labels + j] = bias[j] + dot_f32(w_slice, in_slice);
        }
    }
}
