//! Lightweight 2-layer 1D CNN for syllable-level POS tagging.
//!
//! Architecture: Embedding → Conv1D×3 (k=3,5,9) → ReLU → Conv1D×2 (k=3,7) → ReLU → FC
//! All weights are int8 quantized with per-tensor symmetric scaling.
//! BatchNorm is fused into conv1 weights at export time.

use std::collections::HashMap;

/// Int8 quantized tensor with scale factor.
struct QTensor {
    data: Vec<i8>,
    scale: f32,
    shape: Vec<usize>, // [out_ch, in_ch, kernel_size] for conv, [rows, cols] for fc
}

/// CNN2 model for syllable-level BIO+POS tagging.
pub struct Cnn2 {
    embed_dim: usize,
    hidden: usize,
    num_labels: usize,
    vocab_size: usize,
    // Embedding (int8)
    emb: QTensor,
    // Layer 1: 3 parallel convolutions (BN fused)
    c3: QTensor,
    c3_bias: Vec<f32>,
    c5: QTensor,
    c5_bias: Vec<f32>,
    c9: QTensor,
    c9_bias: Vec<f32>,
    // Layer 2: 2 convolutions
    c2a: QTensor,
    c2a_bias: Vec<f32>,
    c2b: QTensor,
    c2b_bias: Vec<f32>,
    // FC output
    fc: QTensor,
    fc_bias: Vec<f32>,
    // Vocabularies
    syl_to_idx: HashMap<char, usize>,
    labels: Vec<String>,
}

impl Cnn2 {
    /// Parse CNN2 binary format (supports gzip).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let data = if data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b {
            use std::io::Read;
            let mut decoder = flate2::read::GzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)
                .map_err(|e| format!("CNN gzip decompress failed: {}", e))?;
            decompressed
        } else {
            data.to_vec()
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
            QTensor { data: int8_data, scale, shape }
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
        let syls: Vec<String> = serde_json::from_str(syl_json)
            .map_err(|e| format!("Bad syllable vocab JSON: {}", e))?;
        let mut syl_to_idx = HashMap::with_capacity(syls.len());
        for (i, s) in syls.iter().enumerate() {
            if let Some(ch) = s.chars().next() {
                syl_to_idx.insert(ch, i);
            }
        }

        let labels: Vec<String> = serde_json::from_str(lab_json)
            .map_err(|e| format!("Bad label vocab JSON: {}", e))?;

        Ok(Cnn2 {
            embed_dim, hidden, num_labels, vocab_size,
            emb, c3, c3_bias, c5, c5_bias, c9, c9_bias,
            c2a, c2a_bias, c2b, c2b_bias, fc, fc_bias,
            syl_to_idx, labels,
        })
    }

    /// Run inference on a text string. Returns Vec<(char, label_str)>.
    pub fn predict(&self, text: &str) -> Vec<(char, &str)> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return vec![];
        }

        // Map chars to indices
        let indices: Vec<usize> = chars.iter()
            .map(|&ch| *self.syl_to_idx.get(&ch).unwrap_or(&1)) // 1 = <UNK>
            .collect();

        // 1. Embedding lookup (dequantize to f32)
        // emb_out: [n, embed_dim]
        let mut emb_out = vec![0.0f32; n * self.embed_dim];
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.vocab_size {
                let base = idx * self.embed_dim;
                for j in 0..self.embed_dim {
                    emb_out[i * self.embed_dim + j] =
                        self.emb.data[base + j] as f32 * self.emb.scale;
                }
            }
        }

        // 2. Layer 1: three parallel conv1d + ReLU
        let h = self.hidden; // 96
        let mut l1_out = vec![0.0f32; n * h * 3]; // [n, 288]

        self.conv1d_relu(&emb_out, n, self.embed_dim,
                         &self.c3, &self.c3_bias, 3, &mut l1_out, 0, h);
        self.conv1d_relu(&emb_out, n, self.embed_dim,
                         &self.c5, &self.c5_bias, 5, &mut l1_out, h, h);
        self.conv1d_relu(&emb_out, n, self.embed_dim,
                         &self.c9, &self.c9_bias, 9, &mut l1_out, h * 2, h);

        // 3. Layer 2: two parallel conv1d + ReLU
        let mut l2_out = vec![0.0f32; n * h * 2]; // [n, 192]

        self.conv1d_relu(&l1_out, n, h * 3,
                         &self.c2a, &self.c2a_bias, 3, &mut l2_out, 0, h);
        self.conv1d_relu(&l1_out, n, h * 3,
                         &self.c2b, &self.c2b_bias, 7, &mut l2_out, h, h);

        // 4. FC layer
        let mut logits = vec![0.0f32; n * self.num_labels];
        self.fc_forward(&l2_out, n, h * 2, &mut logits);

        // 5. Argmax
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * self.num_labels;
            let mut best_j = 0;
            let mut best_v = logits[base];
            for j in 1..self.num_labels {
                if logits[base + j] > best_v {
                    best_v = logits[base + j];
                    best_j = j;
                }
            }
            let label = if best_j < self.labels.len() {
                &self.labels[best_j]
            } else {
                "O"
            };
            result.push((chars[i], label));
        }

        result
    }

    /// Conv1D with int8 weights + f32 bias + ReLU.
    /// Input: [seq_len, in_ch], Output written to out[seq_pos, out_offset..out_offset+out_ch].
    fn conv1d_relu(
        &self,
        input: &[f32],      // [seq_len, in_ch]
        seq_len: usize,
        in_ch: usize,
        weight: &QTensor,   // [out_ch, in_ch, kernel_size]
        bias: &[f32],       // [out_ch]
        kernel_size: usize,
        output: &mut [f32], // [seq_len, total_out_ch]
        out_offset: usize,  // channel offset in output
        out_ch: usize,
    ) {
        let pad = kernel_size / 2;
        let total_out_ch = output.len() / seq_len;

        for t in 0..seq_len {
            for oc in 0..out_ch {
                let mut sum = bias[oc];
                for k in 0..kernel_size {
                    let t_in = t as isize + k as isize - pad as isize;
                    if t_in < 0 || t_in >= seq_len as isize {
                        continue;
                    }
                    let t_in = t_in as usize;
                    for ic in 0..in_ch {
                        let w_idx = oc * in_ch * kernel_size + ic * kernel_size + k;
                        let w_val = weight.data[w_idx] as f32 * weight.scale;
                        let in_val = input[t_in * in_ch + ic];
                        sum += w_val * in_val;
                    }
                }
                // ReLU
                output[t * total_out_ch + out_offset + oc] = sum.max(0.0);
            }
        }
    }

    /// FC layer with int8 weights + f32 bias.
    fn fc_forward(
        &self,
        input: &[f32],    // [seq_len, in_features]
        seq_len: usize,
        in_features: usize,
        output: &mut [f32], // [seq_len, num_labels]
    ) {
        for t in 0..seq_len {
            for j in 0..self.num_labels {
                let mut sum = self.fc_bias[j];
                for k in 0..in_features {
                    let w_idx = j * in_features + k;
                    let w_val = self.fc.data[w_idx] as f32 * self.fc.scale;
                    sum += w_val * input[t * in_features + k];
                }
                output[t * self.num_labels + j] = sum;
            }
        }
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
