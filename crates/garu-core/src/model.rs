//! Model loader and analyzer orchestration.
//!
//! Parses the GMDL binary format, constructs the full pipeline
//! (embedding → BiLSTM → output projection → CRF), and converts
//! BIO tag sequences back into morphological tokens.

use crate::crf::Crf;
use crate::jamo;
use crate::lstm::{BiLstm, Embedding, LstmLayer, QuantizedMatrix};
use crate::trie::Dict;
use crate::types::{AnalyzeResult, Pos, Token};

// ---------------------------------------------------------------------------
// TagSet
// ---------------------------------------------------------------------------

/// BIO tag type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BioTag {
    B,
    I,
    O,
}

/// Maps tag IDs (from CRF output) to (BIO, optional POS) pairs.
pub struct TagSet {
    pub labels: Vec<(BioTag, Option<Pos>)>,
}

impl TagSet {
    /// Convert a tag ID to its BIO tag and optional POS.
    pub fn tag_to_bio_pos(&self, tag_id: usize) -> (BioTag, Option<Pos>) {
        if tag_id < self.labels.len() {
            self.labels[tag_id]
        } else {
            (BioTag::O, None)
        }
    }

    /// Number of tags in this tagset.
    pub fn num_tags(&self) -> usize {
        self.labels.len()
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"GMDL";
const VERSION: u32 = 1;

/// Full model holding all components needed for analysis.
pub struct Model {
    pub embedding: Embedding,
    pub bilstm: BiLstm,
    pub output_weights: QuantizedMatrix,
    pub output_bias: Vec<f32>,
    pub crf: Crf,
    pub tagset: TagSet,
    pub dict: Option<Dict>,
}

/// Helper to read a little-endian u32 from a byte slice at a given offset.
fn read_u32(data: &[u8], pos: usize) -> Result<u32, String> {
    if pos + 4 > data.len() {
        return Err(format!("Unexpected end of data at offset {}", pos));
    }
    Ok(u32::from_le_bytes(
        data[pos..pos + 4]
            .try_into()
            .map_err(|_| "Bad u32 bytes")?,
    ))
}

/// Helper to read a little-endian f32 from a byte slice at a given offset.
fn read_f32(data: &[u8], pos: usize) -> Result<f32, String> {
    if pos + 4 > data.len() {
        return Err(format!("Unexpected end of data at offset {}", pos));
    }
    Ok(f32::from_le_bytes(
        data[pos..pos + 4]
            .try_into()
            .map_err(|_| "Bad f32 bytes")?,
    ))
}

/// Parse a quantized matrix from section data starting at `pos`.
/// Returns (QuantizedMatrix, bytes_consumed).
fn parse_quantized_matrix(data: &[u8], pos: usize) -> Result<(QuantizedMatrix, usize), String> {
    let rows = read_u32(data, pos)? as usize;
    let cols = read_u32(data, pos + 4)? as usize;
    let scale = read_f32(data, pos + 8)?;
    let start = pos + 12;
    let len = rows * cols;
    if start + len > data.len() {
        return Err("Quantized matrix data truncated".into());
    }
    // Reinterpret &[u8] as &[i8] safely.
    let i8_data: Vec<i8> = data[start..start + len]
        .iter()
        .map(|&b| b as i8)
        .collect();
    Ok((QuantizedMatrix::new(i8_data, rows, cols, scale), 12 + len))
}

impl Model {
    /// Deserialize a model from the GMDL binary format.
    ///
    /// Format: `[GMDL 4bytes][version u32][sections...]`
    /// Each section: `[type u8][len u32][data...]`
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 8 {
            return Err("Data too short for GMDL header".into());
        }
        if &data[0..4] != MAGIC {
            return Err("Invalid magic bytes (expected GMDL)".into());
        }
        let version = read_u32(data, 4)?;
        if version != VERSION {
            return Err(format!("Unsupported GMDL version: {}", version));
        }

        let mut pos = 8;

        let mut embedding: Option<Embedding> = None;
        let mut bilstm: Option<BiLstm> = None;
        let mut output_weights: Option<QuantizedMatrix> = None;
        let mut output_bias: Option<Vec<f32>> = None;
        let mut crf: Option<Crf> = None;
        let mut tagset: Option<TagSet> = None;
        let mut dict: Option<Dict> = None;

        while pos < data.len() {
            if pos + 5 > data.len() {
                return Err("Truncated section header".into());
            }
            let section_type = data[pos];
            let section_len = read_u32(data, pos + 1)? as usize;
            pos += 5;

            if pos + section_len > data.len() {
                return Err(format!(
                    "Section type {} claims {} bytes but only {} remain",
                    section_type,
                    section_len,
                    data.len() - pos
                ));
            }

            let section_data = &data[pos..pos + section_len];

            match section_type {
                0 => {
                    // Embedding: u32 vocab_size + u32 embed_dim + f32 weights
                    let vocab_size = read_u32(section_data, 0)? as usize;
                    let embed_dim = read_u32(section_data, 4)? as usize;
                    let num_floats = vocab_size * embed_dim;
                    let weights_start = 8;
                    if weights_start + num_floats * 4 > section_data.len() {
                        return Err("Embedding weights truncated".into());
                    }
                    let mut weights = Vec::with_capacity(num_floats);
                    for i in 0..num_floats {
                        weights.push(read_f32(section_data, weights_start + i * 4)?);
                    }
                    embedding = Some(Embedding::new(weights, vocab_size, embed_dim));
                }
                1 => {
                    // BiLSTM: u32 num_layers + u32 hidden_size + per layer data
                    let num_layers = read_u32(section_data, 0)? as usize;
                    let hidden_size = read_u32(section_data, 4)? as usize;
                    let mut spos = 8;
                    let mut layers = Vec::with_capacity(num_layers);

                    for _ in 0..num_layers {
                        let mut layer_pair: Vec<LstmLayer> = Vec::new();
                        // fwd + bwd
                        for _ in 0..2 {
                            let mut w_i: Vec<QuantizedMatrix> = Vec::new();
                            let mut w_h: Vec<QuantizedMatrix> = Vec::new();
                            let mut biases: Vec<Vec<f32>> = Vec::new();

                            // 4 gates
                            for _ in 0..4 {
                                let (mat, consumed) =
                                    parse_quantized_matrix(section_data, spos)?;
                                w_i.push(mat);
                                spos += consumed;

                                let (mat, consumed) =
                                    parse_quantized_matrix(section_data, spos)?;
                                w_h.push(mat);
                                spos += consumed;

                                // bias: hidden_size f32 values
                                let mut bias = Vec::with_capacity(hidden_size);
                                for j in 0..hidden_size {
                                    bias.push(read_f32(section_data, spos + j * 4)?);
                                }
                                spos += hidden_size * 4;
                                biases.push(bias);
                            }

                            layer_pair.push(LstmLayer {
                                w_i: [
                                    w_i.remove(0),
                                    w_i.remove(0),
                                    w_i.remove(0),
                                    w_i.remove(0),
                                ],
                                w_h: [
                                    w_h.remove(0),
                                    w_h.remove(0),
                                    w_h.remove(0),
                                    w_h.remove(0),
                                ],
                                bias: [
                                    biases.remove(0),
                                    biases.remove(0),
                                    biases.remove(0),
                                    biases.remove(0),
                                ],
                                hidden_size,
                            });
                        }

                        layers.push((layer_pair.remove(0), layer_pair.remove(0)));
                    }

                    bilstm = Some(BiLstm {
                        layers,
                        hidden_size,
                    });
                }
                2 => {
                    // Output weights: quantized matrix
                    let (mat, _) = parse_quantized_matrix(section_data, 0)?;
                    output_weights = Some(mat);
                }
                3 => {
                    // Output bias: f32 array
                    let num_floats = section_data.len() / 4;
                    let mut bias = Vec::with_capacity(num_floats);
                    for i in 0..num_floats {
                        bias.push(read_f32(section_data, i * 4)?);
                    }
                    output_bias = Some(bias);
                }
                4 => {
                    // CRF: u32 num_tags + f32 transitions flattened
                    let num_tags = read_u32(section_data, 0)? as usize;
                    let mut transitions = vec![vec![0.0f32; num_tags]; num_tags];
                    let mut spos = 4;
                    for row in transitions.iter_mut().take(num_tags) {
                        for val in row.iter_mut().take(num_tags) {
                            *val = read_f32(section_data, spos)?;
                            spos += 4;
                        }
                    }
                    crf = Some(Crf::new(transitions, num_tags));
                }
                5 => {
                    // TagSet: u32 num_labels + per label: u8 bio + u8 pos_byte
                    let num_labels = read_u32(section_data, 0)? as usize;
                    let mut labels = Vec::with_capacity(num_labels);
                    let mut spos = 4;
                    for _ in 0..num_labels {
                        if spos + 2 > section_data.len() {
                            return Err("TagSet data truncated".into());
                        }
                        let bio_byte = section_data[spos];
                        let pos_byte = section_data[spos + 1];
                        spos += 2;

                        let bio = match bio_byte {
                            0 => BioTag::B,
                            1 => BioTag::I,
                            2 => BioTag::O,
                            _ => return Err(format!("Invalid BIO byte: {}", bio_byte)),
                        };

                        let pos_tag = if bio == BioTag::O {
                            None
                        } else if pos_byte <= 41 {
                            // Safety: pos_byte validated in range 0..=41
                            // matching the 42 variants of Pos (#[repr(u8)]).
                            Some(unsafe { std::mem::transmute::<u8, Pos>(pos_byte) })
                        } else {
                            return Err(format!("Invalid POS byte: {}", pos_byte));
                        };

                        labels.push((bio, pos_tag));
                    }
                    tagset = Some(TagSet { labels });
                }
                6 => {
                    // Dictionary: raw bytes for Dict::from_bytes
                    dict = Some(Dict::from_bytes(section_data)?);
                }
                _ => {
                    // Unknown section type: skip silently for forward compat
                }
            }

            pos += section_len;
        }

        Ok(Model {
            embedding: embedding.ok_or("Missing embedding section (type 0)")?,
            bilstm: bilstm.ok_or("Missing BiLSTM section (type 1)")?,
            output_weights: output_weights.ok_or("Missing output weights section (type 2)")?,
            output_bias: output_bias.ok_or("Missing output bias section (type 3)")?,
            crf: crf.ok_or("Missing CRF section (type 4)")?,
            tagset: tagset.ok_or("Missing tagset section (type 5)")?,
            dict,
        })
    }
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Maximum input length before chunking.
const MAX_INPUT_CHARS: usize = 100_000;

/// High-level analyzer wrapping a loaded Model.
pub struct Analyzer {
    pub model: Model,
}

impl Analyzer {
    /// Create an analyzer from a loaded model.
    pub fn new(model: Model) -> Self {
        Self { model }
    }

    /// Build a mapping from Jamo index to (char_index, byte_offset).
    /// Each Hangul syllable produces 2-3 Jamo; all other chars produce 1.
    fn build_jamo_to_char_map(text: &str) -> Vec<(usize, usize)> {
        let mut mapping = Vec::new();
        for (char_idx, (byte_offset, ch)) in text.char_indices().enumerate() {
            if let Some((_, _, tail)) = jamo::decompose(ch) {
                // lead + vowel
                mapping.push((char_idx, byte_offset));
                mapping.push((char_idx, byte_offset));
                // optional tail
                if tail.is_some() {
                    mapping.push((char_idx, byte_offset));
                }
            } else {
                mapping.push((char_idx, byte_offset));
            }
        }
        mapping
    }

    /// Run the full pipeline on a single chunk of text.
    fn analyze_chunk(&self, text: &str) -> AnalyzeResult {
        if text.is_empty() {
            return AnalyzeResult {
                tokens: vec![],
                score: 0.0,
                elapsed_ms: 0.0,
            };
        }

        let start_time = std::time::Instant::now();

        // 1. Encode to Jamo IDs
        let ids = jamo::encode(text);
        if ids.is_empty() {
            return AnalyzeResult {
                tokens: vec![],
                score: 0.0,
                elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            };
        }

        // 2. Embedding lookup
        let embeddings: Vec<Vec<f32>> = ids
            .iter()
            .map(|&id| self.model.embedding.lookup(id).to_vec())
            .collect();

        // 3. BiLSTM forward
        let lstm_out = self.model.bilstm.forward(&embeddings);

        // 4. Output projection: output_weights * h + output_bias
        let emissions: Vec<Vec<f32>> = lstm_out
            .iter()
            .map(|h| {
                let proj = self.model.output_weights.matvec(h);
                proj.iter()
                    .zip(self.model.output_bias.iter())
                    .map(|(&p, &b)| p + b)
                    .collect()
            })
            .collect();

        // 5. CRF decode
        let (tag_ids, crf_score) = self.model.crf.decode(&emissions);

        // 6. Convert BIO tags to Token spans
        let jamo_to_char = Self::build_jamo_to_char_map(text);
        let tokens = self.bio_to_tokens(text, &tag_ids, &jamo_to_char);

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        AnalyzeResult {
            tokens,
            score: crf_score,
            elapsed_ms,
        }
    }

    /// Convert BIO tag sequence into Token spans.
    fn bio_to_tokens(
        &self,
        text: &str,
        tag_ids: &[usize],
        jamo_to_char: &[(usize, usize)],
    ) -> Vec<Token> {
        let mut tokens = Vec::new();
        let char_byte_offsets: Vec<(usize, usize)> = text
            .char_indices()
            .map(|(byte_off, ch)| (byte_off, ch.len_utf8()))
            .collect();
        let text_byte_len = text.len();

        // Track current token being built
        let mut current_start_char: Option<usize> = None;
        let mut current_pos: Option<Pos> = None;

        for (jamo_idx, &tag_id) in tag_ids.iter().enumerate() {
            let (bio, pos_opt) = self.model.tagset.tag_to_bio_pos(tag_id);

            match bio {
                BioTag::B => {
                    // Close previous token if any
                    if let Some(start_char) = current_start_char {
                        let char_idx = if jamo_idx < jamo_to_char.len() {
                            jamo_to_char[jamo_idx].0
                        } else {
                            char_byte_offsets.len()
                        };
                        self.emit_token(
                            text,
                            &char_byte_offsets,
                            text_byte_len,
                            start_char,
                            char_idx,
                            current_pos.unwrap_or(Pos::SW),
                            &mut tokens,
                        );
                    }
                    // Start new token
                    current_start_char = if jamo_idx < jamo_to_char.len() {
                        Some(jamo_to_char[jamo_idx].0)
                    } else {
                        None
                    };
                    current_pos = pos_opt;
                }
                BioTag::I => {
                    // Continue current token; if no current token, start one
                    if current_start_char.is_none() {
                        current_start_char = if jamo_idx < jamo_to_char.len() {
                            Some(jamo_to_char[jamo_idx].0)
                        } else {
                            None
                        };
                        current_pos = pos_opt;
                    }
                }
                BioTag::O => {
                    // Close current token if any
                    if let Some(start_char) = current_start_char {
                        let char_idx = if jamo_idx < jamo_to_char.len() {
                            jamo_to_char[jamo_idx].0
                        } else {
                            char_byte_offsets.len()
                        };
                        self.emit_token(
                            text,
                            &char_byte_offsets,
                            text_byte_len,
                            start_char,
                            char_idx,
                            current_pos.unwrap_or(Pos::SW),
                            &mut tokens,
                        );
                        current_start_char = None;
                        current_pos = None;
                    }
                }
            }
        }

        // Close any remaining token
        if let Some(start_char) = current_start_char {
            self.emit_token(
                text,
                &char_byte_offsets,
                text_byte_len,
                start_char,
                char_byte_offsets.len(),
                current_pos.unwrap_or(Pos::SW),
                &mut tokens,
            );
        }

        tokens
    }

    /// Helper to emit a token from char index range.
    #[allow(clippy::too_many_arguments)]
    fn emit_token(
        &self,
        text: &str,
        char_byte_offsets: &[(usize, usize)],
        text_byte_len: usize,
        start_char: usize,
        end_char: usize,
        pos: Pos,
        tokens: &mut Vec<Token>,
    ) {
        if start_char >= end_char || start_char >= char_byte_offsets.len() {
            return;
        }
        let byte_start = char_byte_offsets[start_char].0;
        let byte_end = if end_char < char_byte_offsets.len() {
            char_byte_offsets[end_char].0
        } else {
            text_byte_len
        };
        if byte_start >= byte_end {
            return;
        }
        let surface = &text[byte_start..byte_end];
        // Skip whitespace-only tokens
        if surface.trim().is_empty() {
            return;
        }
        tokens.push(Token {
            text: surface.to_string(),
            pos,
            start: start_char,
            end: end_char,
            score: None,
        });
    }

    /// Analyze text, returning morphological tokens.
    ///
    /// - Empty string returns empty result.
    /// - Input >100K chars is chunked at spaces and results combined.
    pub fn analyze(&self, text: &str) -> AnalyzeResult {
        if text.is_empty() {
            return AnalyzeResult {
                tokens: vec![],
                score: 0.0,
                elapsed_ms: 0.0,
            };
        }

        if text.chars().count() <= MAX_INPUT_CHARS {
            return self.analyze_chunk(text);
        }

        // Chunk at spaces for long input
        let start_time = std::time::Instant::now();
        let mut all_tokens = Vec::new();
        let mut total_score = 0.0;
        let mut char_offset = 0;

        for chunk in Self::chunk_at_spaces(text, MAX_INPUT_CHARS) {
            let result = self.analyze_chunk(chunk);
            for mut token in result.tokens {
                token.start += char_offset;
                token.end += char_offset;
                all_tokens.push(token);
            }
            total_score += result.score;
            char_offset += chunk.chars().count();
        }

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        AnalyzeResult {
            tokens: all_tokens,
            score: total_score,
            elapsed_ms,
        }
    }

    /// Analyze returning top-N results via CRF beam search.
    pub fn analyze_topn(&self, text: &str, n: usize) -> Vec<AnalyzeResult> {
        if text.is_empty() || n == 0 {
            return vec![];
        }

        let start_time = std::time::Instant::now();

        // Encode and run through pipeline
        let ids = jamo::encode(text);
        if ids.is_empty() {
            return vec![AnalyzeResult {
                tokens: vec![],
                score: 0.0,
                elapsed_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            }];
        }

        let embeddings: Vec<Vec<f32>> = ids
            .iter()
            .map(|&id| self.model.embedding.lookup(id).to_vec())
            .collect();

        let lstm_out = self.model.bilstm.forward(&embeddings);

        let emissions: Vec<Vec<f32>> = lstm_out
            .iter()
            .map(|h| {
                let proj = self.model.output_weights.matvec(h);
                proj.iter()
                    .zip(self.model.output_bias.iter())
                    .map(|(&p, &b)| p + b)
                    .collect()
            })
            .collect();

        let jamo_to_char = Self::build_jamo_to_char_map(text);
        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        self.model
            .crf
            .decode_topn(&emissions, n)
            .into_iter()
            .map(|(tag_ids, score)| {
                let tokens = self.bio_to_tokens(text, &tag_ids, &jamo_to_char);
                AnalyzeResult {
                    tokens,
                    score,
                    elapsed_ms,
                }
            })
            .collect()
    }

    /// Extract surface forms from analysis.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.analyze(text)
            .tokens
            .into_iter()
            .map(|t| t.text)
            .collect()
    }

    /// Split text into chunks at space boundaries, each chunk at most
    /// `max_chars` characters long.
    fn chunk_at_spaces(text: &str, max_chars: usize) -> Vec<&str> {
        let mut chunks = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let char_count = remaining.chars().count();
            if char_count <= max_chars {
                chunks.push(remaining);
                break;
            }

            // Find the last space within max_chars
            let mut split_byte = 0;
            let mut last_space_byte = None;

            for (char_idx, (byte_off, ch)) in remaining.char_indices().enumerate() {
                if char_idx >= max_chars {
                    break;
                }
                if ch == ' ' {
                    last_space_byte = Some(byte_off + 1); // include the space
                }
                split_byte = byte_off + ch.len_utf8();
            }

            let cut = last_space_byte.unwrap_or(split_byte);
            if cut == 0 {
                // Degenerate: no space found and no progress; take all
                chunks.push(remaining);
                break;
            }

            chunks.push(&remaining[..cut]);
            remaining = &remaining[cut..];
        }

        chunks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tagset_basics() {
        let tagset = TagSet {
            labels: vec![
                (BioTag::B, Some(Pos::NNG)),
                (BioTag::I, Some(Pos::NNG)),
                (BioTag::O, None),
                (BioTag::B, Some(Pos::JKS)),
            ],
        };

        assert_eq!(tagset.num_tags(), 4);

        assert_eq!(tagset.tag_to_bio_pos(0), (BioTag::B, Some(Pos::NNG)));
        assert_eq!(tagset.tag_to_bio_pos(1), (BioTag::I, Some(Pos::NNG)));
        assert_eq!(tagset.tag_to_bio_pos(2), (BioTag::O, None));
        assert_eq!(tagset.tag_to_bio_pos(3), (BioTag::B, Some(Pos::JKS)));

        // Out of bounds falls back to O
        assert_eq!(tagset.tag_to_bio_pos(999), (BioTag::O, None));
    }

    #[test]
    fn test_from_bytes_invalid_magic() {
        let bad_data = b"BAAD\x01\x00\x00\x00";
        let result = Model::from_bytes(bad_data);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("Invalid magic"), "got: {}", err);
    }

    #[test]
    fn test_from_bytes_too_short() {
        let result = Model::from_bytes(b"GM");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_bytes_bad_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GMDL");
        data.extend_from_slice(&99u32.to_le_bytes());
        let result = Model::from_bytes(&data);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("version"), "got: {}", err);
    }

    #[test]
    fn test_from_bytes_missing_sections() {
        // Valid header but no sections
        let mut data = Vec::new();
        data.extend_from_slice(b"GMDL");
        data.extend_from_slice(&1u32.to_le_bytes());
        let result = Model::from_bytes(&data);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("Missing"), "got: {}", err);
    }

    #[test]
    fn test_jamo_to_char_map() {
        // "가" = 2 jamo (lead + vowel), "나" = 2 jamo
        let map = Analyzer::build_jamo_to_char_map("가나");
        // 가 → 2 jamo entries pointing to char 0
        // 나 → 2 jamo entries pointing to char 1
        assert_eq!(map.len(), 4);
        assert_eq!(map[0].0, 0); // char index 0
        assert_eq!(map[1].0, 0);
        assert_eq!(map[2].0, 1); // char index 1
        assert_eq!(map[3].0, 1);

        // "한A" → 한 = 3 jamo (lead + vowel + tail), A = 1
        let map = Analyzer::build_jamo_to_char_map("한A");
        assert_eq!(map.len(), 4);
        assert_eq!(map[0].0, 0);
        assert_eq!(map[1].0, 0);
        assert_eq!(map[2].0, 0);
        assert_eq!(map[3].0, 1); // 'A'
    }

    #[test]
    fn test_chunk_at_spaces() {
        let text = "hello world foo bar baz";
        let chunks = Analyzer::chunk_at_spaces(text, 11);
        // "hello world" = 11 chars, "foo bar baz" = 11 chars
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 12); // allow slight overshoot at boundaries
        }

        // Short text should not be chunked
        let chunks = Analyzer::chunk_at_spaces("short", 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "short");

        // Empty text
        let chunks = Analyzer::chunk_at_spaces("", 100);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_bio_tag_enum() {
        assert_ne!(BioTag::B, BioTag::I);
        assert_ne!(BioTag::I, BioTag::O);
        assert_eq!(BioTag::B, BioTag::B);
    }
}
