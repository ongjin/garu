/// BiLSTM forward pass with INT8 quantized weights.

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn tanh_f(x: f32) -> f32 {
    x.tanh()
}

fn tanh_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| tanh_f(x)).collect()
}

/// Element-wise addition of three slices of the same length.
fn add3(a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .zip(c.iter())
        .map(|((&a, &b), &c)| a + b + c)
        .collect()
}

// ---------------------------------------------------------------------------
// QuantizedMatrix
// ---------------------------------------------------------------------------

/// A matrix stored in INT8 with a single scalar scale factor.
/// Effective float value = data[i] as f32 * scale.
pub struct QuantizedMatrix {
    pub data: Vec<i8>,
    pub rows: usize,
    pub cols: usize,
    pub scale: f32,
}

impl QuantizedMatrix {
    pub fn new(data: Vec<i8>, rows: usize, cols: usize, scale: f32) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must equal rows * cols");
        Self {
            data,
            rows,
            cols,
            scale,
        }
    }

    /// Matrix-vector product with on-the-fly dequantization.
    /// output[r] = sum_c(data[r,c] * scale * input[c])
    pub fn matvec(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.cols,
            "input length must equal number of columns"
        );
        (0..self.rows)
            .map(|r| {
                let row_start = r * self.cols;
                let mut acc: f32 = 0.0;
                for c in 0..self.cols {
                    acc += self.data[row_start + c] as f32 * input[c];
                }
                acc * self.scale
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Simple embedding lookup table.
pub struct Embedding {
    pub weights: Vec<f32>,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl Embedding {
    pub fn new(weights: Vec<f32>, vocab_size: usize, embed_dim: usize) -> Self {
        assert_eq!(
            weights.len(),
            vocab_size * embed_dim,
            "weights length must equal vocab_size * embed_dim"
        );
        Self {
            weights,
            vocab_size,
            embed_dim,
        }
    }

    /// Look up the embedding for `id`. Falls back to ID 0 if out of bounds.
    pub fn lookup(&self, id: u16) -> &[f32] {
        let idx = if (id as usize) < self.vocab_size {
            id as usize
        } else {
            0
        };
        let start = idx * self.embed_dim;
        &self.weights[start..start + self.embed_dim]
    }
}

// ---------------------------------------------------------------------------
// LstmLayer (single direction)
// ---------------------------------------------------------------------------

/// A single-direction LSTM layer with 4 gates (i, f, g, o).
/// Each gate has input weights (w_i), hidden weights (w_h), and bias.
pub struct LstmLayer {
    /// Input weights for gates [i, f, g, o], each: hidden_size x input_size
    pub w_i: [QuantizedMatrix; 4],
    /// Hidden weights for gates [i, f, g, o], each: hidden_size x hidden_size
    pub w_h: [QuantizedMatrix; 4],
    /// Biases for gates [i, f, g, o], each: hidden_size
    pub bias: [Vec<f32>; 4],
    pub hidden_size: usize,
}

impl LstmLayer {
    /// One LSTM time step.
    /// Returns (new_h, new_c).
    pub fn step(&self, input: &[f32], h: &[f32], c: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let hs = self.hidden_size;
        debug_assert_eq!(h.len(), hs);
        debug_assert_eq!(c.len(), hs);

        // For each gate: gate = act(w_i * input + w_h * h + bias)
        let gate_raw: Vec<Vec<f32>> = (0..4)
            .map(|g| add3(&self.w_i[g].matvec(input), &self.w_h[g].matvec(h), &self.bias[g]))
            .collect();

        // i = sigmoid, f = sigmoid, g = tanh, o = sigmoid
        let i_gate: Vec<f32> = gate_raw[0].iter().map(|&x| sigmoid(x)).collect();
        let f_gate: Vec<f32> = gate_raw[1].iter().map(|&x| sigmoid(x)).collect();
        let g_gate: Vec<f32> = tanh_vec(&gate_raw[2]);
        let o_gate: Vec<f32> = gate_raw[3].iter().map(|&x| sigmoid(x)).collect();

        // c_new = f * c + i * g
        let new_c: Vec<f32> = (0..hs)
            .map(|j| f_gate[j] * c[j] + i_gate[j] * g_gate[j])
            .collect();

        // h_new = o * tanh(c_new)
        let tanh_c = tanh_vec(&new_c);
        let new_h: Vec<f32> = (0..hs).map(|j| o_gate[j] * tanh_c[j]).collect();

        (new_h, new_c)
    }
}

// ---------------------------------------------------------------------------
// BiLstm
// ---------------------------------------------------------------------------

/// Bidirectional LSTM with multiple stacked layers.
pub struct BiLstm {
    /// Each element is (forward_layer, backward_layer).
    pub layers: Vec<(LstmLayer, LstmLayer)>,
    pub hidden_size: usize,
}

impl BiLstm {
    /// Run the full bidirectional LSTM over a sequence of embeddings.
    /// Returns Vec of length seq_len, each element of size hidden_size * 2.
    pub fn forward(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = embeddings.len();
        if seq_len == 0 {
            return vec![];
        }

        let mut current: Vec<Vec<f32>> = embeddings.to_vec();

        for (fwd_layer, bwd_layer) in &self.layers {
            let hs = self.hidden_size;

            // Forward pass
            let mut fwd_outputs = Vec::with_capacity(seq_len);
            let mut h = vec![0.0f32; hs];
            let mut c = vec![0.0f32; hs];
            for t in 0..seq_len {
                let (new_h, new_c) = fwd_layer.step(&current[t], &h, &c);
                h = new_h.clone();
                c = new_c;
                fwd_outputs.push(new_h);
            }

            // Backward pass
            let mut bwd_outputs = vec![vec![0.0f32; hs]; seq_len];
            let mut h = vec![0.0f32; hs];
            let mut c = vec![0.0f32; hs];
            for t in (0..seq_len).rev() {
                let (new_h, new_c) = bwd_layer.step(&current[t], &h, &c);
                h = new_h.clone();
                c = new_c;
                bwd_outputs[t] = new_h;
            }

            // Concatenate fwd + bwd
            current = (0..seq_len)
                .map(|t| {
                    let mut out = fwd_outputs[t].clone();
                    out.extend_from_slice(&bwd_outputs[t]);
                    out
                })
                .collect();
        }

        current
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_matvec() {
        // [[10, 20], [30, 40]] * scale=0.1 * [1, 2]
        // row0: (10*1 + 20*2) * 0.1 = 50 * 0.1 = 5.0
        // row1: (30*1 + 40*2) * 0.1 = 110 * 0.1 = 11.0
        let mat = QuantizedMatrix::new(vec![10, 20, 30, 40], 2, 2, 0.1);
        let input = vec![1.0, 2.0];
        let result = mat.matvec(&input);
        assert!((result[0] - 5.0).abs() < 1e-5, "expected 5.0, got {}", result[0]);
        assert!(
            (result[1] - 11.0).abs() < 1e-5,
            "expected 11.0, got {}",
            result[1]
        );
    }

    #[test]
    fn test_embedding_lookup_normal() {
        // vocab_size=3, embed_dim=2
        let emb = Embedding::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        assert_eq!(emb.lookup(0), &[1.0, 2.0]);
        assert_eq!(emb.lookup(1), &[3.0, 4.0]);
        assert_eq!(emb.lookup(2), &[5.0, 6.0]);
    }

    #[test]
    fn test_embedding_lookup_out_of_bounds() {
        let emb = Embedding::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        // Out of bounds falls back to ID 0
        assert_eq!(emb.lookup(99), &[1.0, 2.0]);
        assert_eq!(emb.lookup(u16::MAX), &[1.0, 2.0]);
    }

    /// Helper: build an LstmLayer with all-zero quantized weights and biases.
    fn make_zero_lstm(input_size: usize, hidden_size: usize) -> LstmLayer {
        let make_mat = |r, c| QuantizedMatrix::new(vec![0i8; r * c], r, c, 0.01);
        LstmLayer {
            w_i: [
                make_mat(hidden_size, input_size),
                make_mat(hidden_size, input_size),
                make_mat(hidden_size, input_size),
                make_mat(hidden_size, input_size),
            ],
            w_h: [
                make_mat(hidden_size, hidden_size),
                make_mat(hidden_size, hidden_size),
                make_mat(hidden_size, hidden_size),
                make_mat(hidden_size, hidden_size),
            ],
            bias: [
                vec![0.0; hidden_size],
                vec![0.0; hidden_size],
                vec![0.0; hidden_size],
                vec![0.0; hidden_size],
            ],
            hidden_size,
        }
    }

    #[test]
    fn test_lstm_step_output_shape() {
        let hidden_size = 8;
        let input_size = 4;
        let layer = make_zero_lstm(input_size, hidden_size);
        let input = vec![1.0; input_size];
        let h = vec![0.0; hidden_size];
        let c = vec![0.0; hidden_size];
        let (new_h, new_c) = layer.step(&input, &h, &c);
        assert_eq!(new_h.len(), hidden_size);
        assert_eq!(new_c.len(), hidden_size);
    }

    #[test]
    fn test_bilstm_forward_output_shape() {
        let hidden_size = 8;
        let input_size = 4;
        let seq_len = 5;

        let bilstm = BiLstm {
            layers: vec![(
                make_zero_lstm(input_size, hidden_size),
                make_zero_lstm(input_size, hidden_size),
            )],
            hidden_size,
        };

        let embeddings: Vec<Vec<f32>> = (0..seq_len).map(|_| vec![1.0; input_size]).collect();
        let output = bilstm.forward(&embeddings);

        assert_eq!(output.len(), seq_len);
        for t in 0..seq_len {
            assert_eq!(
                output[t].len(),
                hidden_size * 2,
                "output at t={} should be hidden_size*2={}",
                t,
                hidden_size * 2
            );
        }
    }
}
