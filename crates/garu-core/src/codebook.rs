//! Codebook-based morphological analyzer (lattice + Viterbi).

use std::collections::HashMap;
use crate::trie::Dict;
use crate::types::{AnalyzeResult, Pos, Token};

const NUM_POS: usize = 42;
const BOS: u8 = 255;

// ---------------------------------------------------------------------------
// Content / Functional POS classification
// ---------------------------------------------------------------------------

fn is_content_pos(pos: Pos) -> bool {
    matches!(
        pos,
        Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP
            | Pos::VV | Pos::VA | Pos::VX | Pos::VCP | Pos::VCN
            | Pos::MAG | Pos::MAJ | Pos::MM | Pos::IC | Pos::XR
    )
}

fn is_functional_pos(pos: Pos) -> bool {
    matches!(
        pos,
        Pos::JKS | Pos::JKC | Pos::JKG | Pos::JKO | Pos::JKB
            | Pos::JKV | Pos::JKQ | Pos::JX | Pos::JC
            | Pos::EP | Pos::EF | Pos::EC | Pos::ETN | Pos::ETM
            | Pos::XPN | Pos::XSN | Pos::XSV | Pos::XSA
    )
}

fn classify_oov_char(ch: char) -> Pos {
    match ch {
        '.' | '!' | '?' => Pos::SF,
        ',' | ';' | ':' | '/' | '\u{00B7}' => Pos::SP, // · / = SP
        '\u{2026}' => Pos::SE, // … (ellipsis) = SE
        '~' | '-' | '\u{2013}' | '\u{2212}' => Pos::SO, // ~, -, –, − = SO
        // SS: brackets, quotes, parentheses (Sejong tagset standard)
        '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'' |
        '\u{2018}' | '\u{2019}' | '\u{201C}' | '\u{201D}' |
        '<' | '>' | '\u{2015}' | '\u{2014}' |
        '\u{300C}' | '\u{300D}' | '\u{300E}' | '\u{300F}' |
        '\u{3008}' | '\u{3009}' | '\u{300A}' | '\u{300B}' => Pos::SS,
        c if c.is_ascii_alphabetic() => Pos::SL,
        c if c.is_ascii_digit() => Pos::SN,
        c if ('\u{AC00}'..='\u{D7A3}').contains(&c) => Pos::NNG,
        // CJK Ideographs: treat as NNG (Kiwi convention — Hanja in Korean text is usually NNG)
        c if ('\u{4E00}'..='\u{9FFF}').contains(&c) ||
             ('\u{3400}'..='\u{4DBF}').contains(&c) ||
             ('\u{F900}'..='\u{FAFF}').contains(&c) => Pos::NNG,
        _ => Pos::SW,
    }
}

// ---------------------------------------------------------------------------
// Jamo normalization
// ---------------------------------------------------------------------------

/// Normalize Hangul jongseong (U+11A8-U+11C2) to compatibility jamo (U+3131-U+314E).
/// This ensures output matches the convention used by standard Korean morpheme benchmarks.
fn normalize_jamo(s: &str) -> String {
    s.chars().map(|c| match c {
        '\u{11A8}' => '\u{3131}', // ᄀ → ㄱ
        '\u{11A9}' => '\u{3132}', // ᄁ → ㄲ
        '\u{11AB}' => '\u{3134}', // ᆫ → ㄴ
        '\u{11AE}' => '\u{3137}', // ᆮ → ㄷ
        '\u{11AF}' => '\u{3139}', // ᆯ → ㄹ
        '\u{11B7}' => '\u{3141}', // ᆷ → ㅁ
        '\u{11B8}' => '\u{3142}', // ᆸ → ㅂ
        '\u{11BA}' => '\u{3145}', // ᆺ → ㅅ
        '\u{11BB}' => '\u{3146}', // ᆻ → ㅆ
        '\u{11BC}' => '\u{3147}', // ᆼ → ㅇ
        '\u{11BD}' => '\u{3148}', // ᆽ → ㅈ
        '\u{11BE}' => '\u{314A}', // ᆾ → ㅊ
        '\u{11BF}' => '\u{314B}', // ᆿ → ㅋ
        '\u{11C0}' => '\u{314C}', // ᇀ → ㅌ
        '\u{11C1}' => '\u{314D}', // ᇁ → ㅍ
        '\u{11C2}' => '\u{314E}', // ᇂ → ㅎ
        other => other,
    }).collect()
}

// ---------------------------------------------------------------------------
// Suffix codebook types
// ---------------------------------------------------------------------------

/// A single morpheme in a suffix analysis.
#[derive(Debug, Clone)]
pub struct SuffixMorpheme {
    pub form: String,
    pub pos: Pos,
}

/// One possible analysis of a suffix pattern.
#[derive(Debug, Clone)]
pub struct SuffixAnalysis {
    pub morphemes: Vec<SuffixMorpheme>,
    pub freq: u32,
}

/// A suffix codebook entry.
#[derive(Debug, Clone)]
pub struct SuffixEntry {
    pub surface: String,
    pub analyses: Vec<SuffixAnalysis>,
}

// ---------------------------------------------------------------------------
// Lattice arc
// ---------------------------------------------------------------------------

struct LatticeArc {
    start: usize,      // char position
    end: usize,        // char position
    morphemes: Vec<(String, Pos)>,
    cost: f32,
}

// ---------------------------------------------------------------------------
// Viterbi backpointer
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Backpointer {
    prev_pos: usize,        // position in text
    prev_state: (u8, u8),   // (prev_pos_tag, prev_prev_pos_tag)
    arc_idx: Option<usize>, // index into arcs list, None for space pass-through
}

// ---------------------------------------------------------------------------
// CodebookAnalyzer
// ---------------------------------------------------------------------------

fn now_ms() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::Instant;
        static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let origin = ORIGIN.get_or_init(Instant::now);
        origin.elapsed().as_secs_f64() * 1000.0
    }
    #[cfg(target_arch = "wasm32")]
    {
        0.0
    }
}

/// Codebook-based analyzer using lattice + Viterbi decoding.
pub struct CodebookAnalyzer {
    suffix_entries: Vec<SuffixEntry>,
    suffix_map: HashMap<String, usize>, // surface -> index in suffix_entries
    content_dict: Dict,
    trigram_costs: Vec<f32>,    // flat [42*42*42]
    bigram_costs: Vec<f32>,     // flat [42*42]
    default_cost: f32,
    max_freq: f32,
    max_suffix_freq: f32,
    morpheme_penalty: f32,
    oov_penalty: f32,
    length_bonus: f32,
    single_char_content_penalty: f32,
    #[allow(dead_code)]
    max_word_len: usize,        // max char length of content words in dict
    max_suffix_len: usize,      // max char length of suffix patterns
    /// Word ambiguity table: word → Vec<(Pos, score)>
    ambiguity: HashMap<String, Vec<(Pos, f32)>>,
    /// Word-bigram cost bonuses: (word, prev_pos) → (target_pos, bonus)
    word_bigrams: HashMap<String, Vec<(u8, u8, f32)>>,
    /// Smart eojeol cache: pre-analyzed eojeol → morpheme sequence
    eojeol_cache: HashMap<String, Vec<(String, Pos)>>,
}

impl CodebookAnalyzer {
    /// Parse a GMDL v3 file from raw bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 8 {
            return Err("Data too short for GMDL header".into());
        }
        if &data[0..4] != b"GMDL" {
            return Err("Invalid magic bytes (expected GMDL)".into());
        }
        let version = u32::from_le_bytes(
            data[4..8].try_into().map_err(|_| "Bad version bytes")?,
        );
        if version != 3 {
            return Err(format!("Expected GMDL v3, got v{}", version));
        }

        let mut pos = 8;
        let mut section6: Option<&[u8]> = None;
        let mut section7: Option<&[u8]> = None;
        let mut section8: Option<&[u8]> = None;
        let mut section9: Option<&[u8]> = None;
        let mut section10: Option<&[u8]> = None;
        let mut section11: Option<&[u8]> = None;
        let mut section12: Option<&[u8]> = None;
        let mut section13: Option<&[u8]> = None;

        while pos < data.len() {
            if pos + 5 > data.len() {
                return Err("Truncated section header".into());
            }
            let section_type = data[pos];
            let section_len = u32::from_le_bytes(
                data[pos + 1..pos + 5].try_into().map_err(|_| "Bad section len")?,
            ) as usize;
            pos += 5;

            if pos + section_len > data.len() {
                return Err(format!(
                    "Section {} claims {} bytes but only {} remain",
                    section_type, section_len, data.len() - pos
                ));
            }

            let section_data = &data[pos..pos + section_len];
            match section_type {
                6 => section6 = Some(section_data),
                7 => section7 = Some(section_data),
                8 => section8 = Some(section_data),
                9 => section9 = Some(section_data),
                10 => section10 = Some(section_data),
                11 => section11 = Some(section_data),
                12 => section12 = Some(section_data),
                13 => section13 = Some(section_data),
                _ => {} // skip unknown sections
            }
            pos += section_len;
        }

        Self::from_sections(
            section6.ok_or("Missing content dict section (6)")?,
            section7.ok_or("Missing suffix codebook section (7)")?,
            section8.ok_or("Missing trigram costs section (8)")?,
            section9.ok_or("Missing frequencies section (9)")?,
            section10.ok_or("Missing parameters section (10)")?,
            section11,
            section12,
            section13,
        )
    }

    /// Build from parsed section data.
    fn from_sections(
        dict_data: &[u8],
        suffix_data: &[u8],
        trigram_data: &[u8],
        freq_data: &[u8],
        param_data: &[u8],
        ambig_data: Option<&[u8]>,
        wbigram_data: Option<&[u8]>,
        ecache_data: Option<&[u8]>,
    ) -> Result<Self, String> {
        // Section 6: Content dict
        let content_dict = Dict::from_bytes(dict_data)?;

        // Compute max_word_len from dict (we'll approximate by scanning suffix too)
        // We can't easily iterate all dict entries, so set a reasonable max
        let max_word_len = 15; // Korean words rarely exceed this

        // Section 7: Suffix codebook
        let (suffix_entries, suffix_map, max_suffix_len) = Self::parse_suffix_codebook(suffix_data)?;

        // Section 8: Trigram costs
        let (trigram_costs, bigram_costs, default_cost) = Self::parse_trigram_costs(trigram_data)?;

        // Section 9: Frequencies
        if freq_data.len() < 8 {
            return Err("Frequency section too short".into());
        }
        let max_freq = u32::from_le_bytes(
            freq_data[0..4].try_into().map_err(|_| "Bad max_freq")?,
        ) as f32;
        let max_suffix_freq = u32::from_le_bytes(
            freq_data[4..8].try_into().map_err(|_| "Bad max_suffix_freq")?,
        ) as f32;

        // Section 10: Parameters
        if param_data.len() < 16 {
            return Err("Parameters section too short".into());
        }
        let morpheme_penalty = f32::from_le_bytes(
            param_data[0..4].try_into().map_err(|_| "Bad morpheme_penalty")?,
        );
        let oov_penalty = f32::from_le_bytes(
            param_data[4..8].try_into().map_err(|_| "Bad oov_penalty")?,
        );
        let length_bonus = f32::from_le_bytes(
            param_data[8..12].try_into().map_err(|_| "Bad length_bonus")?,
        );
        let single_char_content_penalty = f32::from_le_bytes(
            param_data[12..16].try_into().map_err(|_| "Bad single_char_content_penalty")?,
        );

        // Parse ambiguity table (Section 11, optional)
        let ambiguity = Self::parse_ambiguity_table(ambig_data, max_freq)?;

        // Parse word-bigram table (Section 12, optional)
        let word_bigrams = Self::parse_word_bigrams(wbigram_data)?;

        // Parse eojeol cache (Section 13, optional)
        let eojeol_cache = Self::parse_eojeol_cache(ecache_data)?;

        Ok(CodebookAnalyzer {
            suffix_entries,
            suffix_map,
            content_dict,
            trigram_costs,
            bigram_costs,
            default_cost,
            max_freq,
            max_suffix_freq,
            morpheme_penalty,
            oov_penalty,
            length_bonus,
            single_char_content_penalty,
            max_word_len,
            max_suffix_len,
            ambiguity,
            word_bigrams,
            eojeol_cache,
        })
    }

    fn parse_suffix_codebook(data: &[u8]) -> Result<(Vec<SuffixEntry>, HashMap<String, usize>, usize), String> {
        if data.len() < 4 {
            return Err("Suffix codebook too short".into());
        }
        let num_entries = u32::from_le_bytes(
            data[0..4].try_into().map_err(|_| "Bad num_entries")?,
        ) as usize;

        let mut entries = Vec::with_capacity(num_entries);
        let mut map = HashMap::with_capacity(num_entries);
        let mut max_suffix_len: usize = 0;
        let mut pos = 4;

        for i in 0..num_entries {
            if pos + 2 > data.len() {
                return Err(format!("Suffix entry {} truncated at surface_len", i));
            }
            let surface_len = u16::from_le_bytes(
                data[pos..pos + 2].try_into().map_err(|_| "Bad surface_len")?,
            ) as usize;
            pos += 2;

            if pos + surface_len > data.len() {
                return Err(format!("Suffix entry {} surface truncated", i));
            }
            let surface = std::str::from_utf8(&data[pos..pos + surface_len])
                .map_err(|e| format!("Invalid UTF-8 in suffix surface: {}", e))?
                .to_string();
            pos += surface_len;

            let char_len = surface.chars().count();
            if char_len > max_suffix_len {
                max_suffix_len = char_len;
            }

            if pos + 2 > data.len() {
                return Err(format!("Suffix entry {} truncated at num_analyses", i));
            }
            let num_analyses = u16::from_le_bytes(
                data[pos..pos + 2].try_into().map_err(|_| "Bad num_analyses")?,
            ) as usize;
            pos += 2;

            let mut analyses = Vec::with_capacity(num_analyses);
            for _ in 0..num_analyses {
                if pos + 5 > data.len() {
                    return Err("Suffix analysis truncated".into());
                }
                let freq = u32::from_le_bytes(
                    data[pos..pos + 4].try_into().map_err(|_| "Bad freq")?,
                );
                pos += 4;

                let num_morphemes = data[pos] as usize;
                pos += 1;

                let mut morphemes = Vec::with_capacity(num_morphemes);
                for _ in 0..num_morphemes {
                    if pos + 2 > data.len() {
                        return Err("Suffix morpheme truncated at form_len".into());
                    }
                    let form_len = u16::from_le_bytes(
                        data[pos..pos + 2].try_into().map_err(|_| "Bad form_len")?,
                    ) as usize;
                    pos += 2;

                    if pos + form_len > data.len() {
                        return Err("Suffix morpheme form truncated".into());
                    }
                    let form = normalize_jamo(
                        std::str::from_utf8(&data[pos..pos + form_len])
                            .map_err(|e| format!("Invalid UTF-8 in suffix morpheme: {}", e))?
                    );
                    pos += form_len;

                    if pos >= data.len() {
                        return Err("Suffix morpheme truncated at pos_byte".into());
                    }
                    let pos_byte = data[pos];
                    pos += 1;

                    if pos_byte > 41 {
                        return Err(format!("Invalid POS byte in suffix: {}", pos_byte));
                    }
                    let pos_tag: Pos = unsafe { std::mem::transmute(pos_byte) };
                    morphemes.push(SuffixMorpheme { form, pos: pos_tag });
                }

                analyses.push(SuffixAnalysis { morphemes, freq });
            }

            map.insert(surface.clone(), i);
            entries.push(SuffixEntry { surface, analyses });
        }

        Ok((entries, map, max_suffix_len))
    }

    fn parse_trigram_costs(data: &[u8]) -> Result<(Vec<f32>, Vec<f32>, f32), String> {
        if data.len() < 8 {
            return Err("Trigram costs section too short".into());
        }
        let num_pos = u32::from_le_bytes(
            data[0..4].try_into().map_err(|_| "Bad num_pos")?,
        ) as usize;
        if num_pos != NUM_POS {
            return Err(format!("Expected {} POS tags, got {}", NUM_POS, num_pos));
        }
        let default_cost = f32::from_le_bytes(
            data[4..8].try_into().map_err(|_| "Bad default_cost")?,
        );

        let trigram_count = NUM_POS * NUM_POS * NUM_POS;
        let bigram_count = NUM_POS * NUM_POS;

        // Detect format: sparse (v2) has 24-byte header, dense (v1) has 8-byte header
        let expected_dense = 8 + (trigram_count + bigram_count) * 4;
        if data.len() == expected_dense {
            // Dense format (v1)
            let mut trigram_costs = Vec::with_capacity(trigram_count);
            let mut pos = 8;
            for _ in 0..trigram_count {
                trigram_costs.push(f32::from_le_bytes(
                    data[pos..pos + 4].try_into().map_err(|_| "Bad trigram f32")?,
                ));
                pos += 4;
            }
            let mut bigram_costs = Vec::with_capacity(bigram_count);
            for _ in 0..bigram_count {
                bigram_costs.push(f32::from_le_bytes(
                    data[pos..pos + 4].try_into().map_err(|_| "Bad bigram f32")?,
                ));
                pos += 4;
            }
            return Ok((trigram_costs, bigram_costs, default_cost));
        }

        // Sparse bitmap + u8 quantized format (v2)
        if data.len() < 24 {
            return Err("Sparse trigram section too short for header".into());
        }
        let tg_min = f32::from_le_bytes(data[8..12].try_into().map_err(|_| "Bad tg_min")?);
        let tg_max = f32::from_le_bytes(data[12..16].try_into().map_err(|_| "Bad tg_max")?);
        let bg_min = f32::from_le_bytes(data[16..20].try_into().map_err(|_| "Bad bg_min")?);
        let bg_max = f32::from_le_bytes(data[20..24].try_into().map_err(|_| "Bad bg_max")?);

        let tg_bitmap_len = (trigram_count + 7) / 8;
        let bg_bitmap_len = (bigram_count + 7) / 8;

        let mut pos = 24;
        // Read trigram bitmap
        let tg_bitmap = &data[pos..pos + tg_bitmap_len];
        pos += tg_bitmap_len;
        // Count set bits for trigram values
        let tg_nz: usize = tg_bitmap.iter().map(|b| b.count_ones() as usize).sum();
        let tg_values = &data[pos..pos + tg_nz];
        pos += tg_nz;
        // Read bigram bitmap
        let bg_bitmap = &data[pos..pos + bg_bitmap_len];
        pos += bg_bitmap_len;
        let bg_nz: usize = bg_bitmap.iter().map(|b| b.count_ones() as usize).sum();
        let bg_values = &data[pos..pos + bg_nz];

        // Expand trigrams
        let tg_range = tg_max - tg_min;
        let mut trigram_costs = vec![0.0f32; trigram_count];
        let mut vi = 0;
        for idx in 0..trigram_count {
            if tg_bitmap[idx / 8] & (1 << (idx % 8)) != 0 {
                trigram_costs[idx] = tg_min + (tg_values[vi] as f32 / 255.0) * tg_range;
                vi += 1;
            }
        }

        // Expand bigrams
        let bg_range = bg_max - bg_min;
        let mut bigram_costs = vec![0.0f32; bigram_count];
        vi = 0;
        for idx in 0..bigram_count {
            if bg_bitmap[idx / 8] & (1 << (idx % 8)) != 0 {
                bigram_costs[idx] = bg_min + (bg_values[vi] as f32 / 255.0) * bg_range;
                vi += 1;
            }
        }

        Ok((trigram_costs, bigram_costs, default_cost))
    }

    // -----------------------------------------------------------------------
    // Cost functions
    // -----------------------------------------------------------------------

    fn parse_ambiguity_table(data: Option<&[u8]>, max_freq: f32) -> Result<HashMap<String, Vec<(Pos, f32)>>, String> {
        let mut map = HashMap::new();
        let data = match data {
            Some(d) => d,
            None => return Ok(map),
        };
        if data.len() < 4 { return Ok(map); }
        let num_entries = u32::from_le_bytes(data[0..4].try_into().map_err(|_| "Bad ambig count")?) as usize;
        let mut pos = 4;
        for _ in 0..num_entries {
            if pos + 2 > data.len() { break; }
            let wlen = u16::from_le_bytes(data[pos..pos+2].try_into().map_err(|_| "Bad wlen")?) as usize;
            pos += 2;
            if pos + wlen > data.len() { break; }
            let word = std::str::from_utf8(&data[pos..pos+wlen])
                .map_err(|_| "Bad UTF-8 in ambig word")?.to_string();
            pos += wlen;
            if pos >= data.len() { break; }
            let num_alts = data[pos] as usize;
            pos += 1;
            let mut alts = Vec::new();
            for _ in 0..num_alts {
                if pos + 3 > data.len() { break; }
                let pos_byte = data[pos];
                let qfreq = u16::from_le_bytes(data[pos+1..pos+3].try_into().map_err(|_| "Bad qfreq")?);
                pos += 3;
                if pos_byte < NUM_POS as u8 {
                    let p: Pos = unsafe { std::mem::transmute(pos_byte) };
                    // Dequantize: score = -ln(freq/max_freq), freq = max_freq * (qfreq/65535)
                    let freq = max_freq * (qfreq as f32 / 65535.0);
                    let score = (max_freq / freq.max(1.0)).ln();
                    alts.push((p, score));
                }
            }
            if !alts.is_empty() {
                map.insert(word, alts);
            }
        }
        Ok(map)
    }

    fn parse_word_bigrams(data: Option<&[u8]>) -> Result<HashMap<String, Vec<(u8, u8, f32)>>, String> {
        let mut map: HashMap<String, Vec<(u8, u8, f32)>> = HashMap::new();
        let data = match data {
            Some(d) if d.len() >= 4 => d,
            _ => return Ok(map),
        };
        let num = u32::from_le_bytes(data[0..4].try_into().map_err(|_| "Bad wbigram count")?) as usize;
        let mut pos = 4;
        for _ in 0..num {
            if pos + 2 > data.len() { break; }
            let wlen = u16::from_le_bytes(data[pos..pos+2].try_into().map_err(|_| "Bad wlen")?) as usize;
            pos += 2;
            if pos + wlen + 3 > data.len() { break; }
            let word = std::str::from_utf8(&data[pos..pos+wlen])
                .map_err(|_| "Bad UTF-8 in wbigram")?.to_string();
            pos += wlen;
            let prev_pos = data[pos];
            let target_pos = data[pos + 1];
            let bonus_q = data[pos + 2] as i8;
            pos += 3;
            let bonus = bonus_q as f32 / 25.0;  // dequantize (scale=25)
            map.entry(word).or_default().push((prev_pos, target_pos, bonus));
        }
        Ok(map)
    }

    /// Get word-bigram cost bonus for (word, prev_pos, target_pos).
    /// Returns a negative bonus (cheaper) if context strongly favors target_pos.
    fn get_word_bigram_bonus(&self, word: &str, prev_pos: u8, target_pos: u8) -> f32 {
        if let Some(entries) = self.word_bigrams.get(word) {
            for &(pp, tp, bonus) in entries {
                if pp == prev_pos && tp == target_pos {
                    return bonus;
                }
            }
        }
        0.0
    }

    fn parse_eojeol_cache(data: Option<&[u8]>) -> Result<HashMap<String, Vec<(String, Pos)>>, String> {
        let mut map = HashMap::new();
        let data = match data {
            Some(d) if d.len() >= 4 => d,
            _ => return Ok(map),
        };
        let num = u32::from_le_bytes(data[0..4].try_into().map_err(|_| "Bad ecache count")?) as usize;
        let mut pos = 4;
        for _ in 0..num {
            if pos + 2 > data.len() { break; }
            let elen = u16::from_le_bytes(data[pos..pos+2].try_into().map_err(|_| "Bad elen")?) as usize;
            pos += 2;
            if pos + elen > data.len() { break; }
            let eojeol = std::str::from_utf8(&data[pos..pos+elen])
                .map_err(|_| "Bad UTF-8 in ecache")?.to_string();
            pos += elen;
            if pos >= data.len() { break; }
            let nm = data[pos] as usize;
            pos += 1;
            let mut morphs = Vec::with_capacity(nm);
            for _ in 0..nm {
                if pos + 2 > data.len() { break; }
                let flen = u16::from_le_bytes(data[pos..pos+2].try_into().map_err(|_| "Bad flen")?) as usize;
                pos += 2;
                if pos + flen + 1 > data.len() { break; }
                let form = normalize_jamo(
                    std::str::from_utf8(&data[pos..pos+flen]).map_err(|_| "Bad UTF-8 in ecache form")?
                );
                pos += flen;
                let pos_byte = data[pos];
                pos += 1;
                if pos_byte <= 41 {
                    let p: Pos = unsafe { std::mem::transmute(pos_byte) };
                    morphs.push((form, p));
                }
            }
            if !morphs.is_empty() {
                map.insert(eojeol, morphs);
            }
        }
        Ok(map)
    }

    fn get_trigram_cost(&self, p1: u8, p2: u8, p3: u8) -> f32 {
        // Any tag outside POS range (including BOS=255) → default cost or bigram backoff
        if p1 as usize >= NUM_POS || p2 as usize >= NUM_POS || p3 as usize >= NUM_POS {
            // Try bigram if p2 and p3 are valid
            if (p2 as usize) < NUM_POS && (p3 as usize) < NUM_POS {
                let bg_idx = (p2 as usize) * NUM_POS + (p3 as usize);
                let bg = self.bigram_costs[bg_idx];
                return if bg != 0.0 { bg + 2.0 } else { self.default_cost + 2.0 };
            }
            return self.default_cost;
        }
        let idx = (p1 as usize) * NUM_POS * NUM_POS + (p2 as usize) * NUM_POS + (p3 as usize);
        let cost = self.trigram_costs[idx];
        if cost != 0.0 {
            return cost;
        }
        // Backoff to bigram
        let bg_idx = (p2 as usize) * NUM_POS + (p3 as usize);
        let bg = self.bigram_costs[bg_idx];
        if bg != 0.0 { bg + 2.0 } else { self.default_cost + 2.0 }
    }

    fn get_word_cost(&self, surface: &str, pos: Pos, freq: u32) -> f32 {
        if freq > 0 {
            let cost = (self.max_freq / freq as f32).ln();
            let char_len = surface.chars().count();
            if char_len > 1 {
                cost - self.length_bonus * (char_len - 1) as f32
            } else if is_content_pos(pos) {
                cost + self.single_char_content_penalty
            } else {
                cost
            }
        } else {
            self.oov_penalty
        }
    }

    fn get_suffix_cost(&self, surface: &str, analysis: &SuffixAnalysis) -> f32 {
        let cost = (self.max_suffix_freq / analysis.freq.max(1) as f32).ln();
        let cost = cost + self.morpheme_penalty * analysis.morphemes.len() as f32;
        let char_len = surface.chars().count();
        if char_len > 1 {
            cost - self.length_bonus * (char_len - 1) as f32
        } else {
            cost
        }
    }

    /// Get frequency from a dict entry score. score = -ln(freq/max_freq), so freq = max_freq * exp(-score).
    fn freq_from_score(&self, score: f32) -> u32 {
        (self.max_freq * (-score).exp()) as u32
    }

    // -----------------------------------------------------------------------
    // ASCII run detection
    // -----------------------------------------------------------------------

    /// Find ASCII runs in text. Returns (char_start, char_end, pos) tuples.
    fn find_ascii_runs(chars: &[char]) -> Vec<(usize, usize, Pos)> {
        let mut runs = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].is_ascii_alphabetic() {
                // SL run: [A-Za-z][A-Za-z.]*[A-Za-z] or single [A-Za-z]
                let start = i;
                i += 1;
                while i < chars.len() && (chars[i].is_ascii_alphabetic() || chars[i] == '.') {
                    i += 1;
                }
                // Trim trailing dots
                while i > start + 1 && chars[i - 1] == '.' {
                    i -= 1;
                }
                runs.push((start, i, Pos::SL));
            } else if chars[i].is_ascii_digit() {
                let start = i;
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                runs.push((start, i, Pos::SN));
            } else {
                i += 1;
            }
        }
        runs
    }

    // -----------------------------------------------------------------------
    // Check if all morphemes are functional
    // -----------------------------------------------------------------------

    fn is_pure_functional(morphemes: &[SuffixMorpheme]) -> bool {
        morphemes.iter().all(|m| is_functional_pos(m.pos))
    }

    // -----------------------------------------------------------------------
    // Lattice construction
    // -----------------------------------------------------------------------

    fn build_lattice(&self, text: &str) -> Vec<LatticeArc> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }

        let mut arcs: Vec<LatticeArc> = Vec::new();
        let mut covered = vec![false; n];

        // Pre-process ASCII runs — always create SL/SN arc, optionally NNP from dict
        let ascii_runs = Self::find_ascii_runs(&chars);
        for &(start, end, default_pos) in &ascii_runs {
            let surface: String = chars[start..end].iter().collect();

            // Always create default SL/SN arc
            arcs.push(LatticeArc {
                start,
                end,
                morphemes: vec![(surface.clone(), default_pos)],
                cost: self.get_word_cost(&surface, default_pos, 1000),
            });

            // Also check dict for NNP — create competing arc if found
            let entries = self.content_dict.lookup(&surface);
            if let Some(entry) = entries.first() {
                let p = entry.morphemes.first().map(|m| m.pos).unwrap_or(default_pos);
                if p != default_pos {
                    let f = self.freq_from_score(entry.score).max(100);
                    arcs.push(LatticeArc {
                        start,
                        end,
                        morphemes: vec![(surface.clone(), p)],
                        cost: self.get_word_cost(&surface, p, f),
                    });
                }
            }
            for j in start..end {
                covered[j] = true;
            }
        }

        // Build byte-to-char and char-to-byte maps
        let mut char_to_byte = Vec::with_capacity(n + 1);
        let mut byte_offset = 0;
        for ch in &chars {
            char_to_byte.push(byte_offset);
            byte_offset += ch.len_utf8();
        }
        char_to_byte.push(byte_offset); // sentinel for end

        // For each starting position, try strategies A, B, C
        for i in 0..n {
            if chars[i].is_whitespace() {
                continue; // spaces handled in Viterbi
            }

            // Strategy A: Content word + optional suffix
            // Use common_prefix_search on the remaining text from position i
            let remaining: String = chars[i..].iter().collect();
            let prefix_matches = self.content_dict.common_prefix_search(&remaining);

            for (byte_len, entries) in &prefix_matches {
                let match_str = &remaining[..*byte_len];
                let match_char_len = match_str.chars().count();
                let content_end = i + match_char_len;

                for entry in entries {
                    if entry.morphemes.is_empty() {
                        continue;
                    }
                    let first_pos = entry.morphemes[0].pos;

                    // Only use as content word if first morpheme is content POS
                    if !is_content_pos(first_pos) {
                        continue;
                    }

                    let freq = self.freq_from_score(entry.score);

                    // A1: Content word alone (primary POS from FST)
                    let morphemes: Vec<(String, Pos)> = entry.morphemes.iter()
                        .map(|m| (m.text.clone(), m.pos))
                        .collect();
                    let mut word_cost = self.get_word_cost(match_str, first_pos, freq);

                    arcs.push(LatticeArc {
                        start: i,
                        end: content_end,
                        morphemes: morphemes.clone(),
                        cost: word_cost,
                    });

                    // A1b: Alternative POS from ambiguity table
                    if let Some(alts) = self.ambiguity.get(match_str) {
                        for &(alt_pos, alt_score) in alts {
                            if alt_pos == first_pos { continue; }
                            let alt_freq = self.freq_from_score(alt_score);
                            let alt_cost = self.get_word_cost(match_str, alt_pos, alt_freq);
                            arcs.push(LatticeArc {
                                start: i,
                                end: content_end,
                                morphemes: vec![(match_str.to_string(), alt_pos)],
                                cost: alt_cost,
                            });
                        }
                    }

                    // A2: Content word + suffix
                    if content_end < n {
                        let max_suf = self.max_suffix_len.min(n - content_end);
                        for suf_len in 1..=max_suf {
                            let suf_surface: String = chars[content_end..content_end + suf_len].iter().collect();
                            if let Some(&suf_idx) = self.suffix_map.get(&suf_surface) {
                                let suf_entry = &self.suffix_entries[suf_idx];
                                for analysis in &suf_entry.analyses {
                                    let suf_cost = self.get_suffix_cost(&suf_surface, analysis);
                                    let mut combined = morphemes.clone();
                                    for m in &analysis.morphemes {
                                        combined.push((m.form.clone(), m.pos));
                                    }
                                    arcs.push(LatticeArc {
                                        start: i,
                                        end: content_end + suf_len,
                                        morphemes: combined,
                                        cost: word_cost + suf_cost,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Strategy B: Pure functional suffix standalone
            let max_suf = self.max_suffix_len.min(n - i);
            for suf_len in 1..=max_suf {
                let suf_surface: String = chars[i..i + suf_len].iter().collect();
                if let Some(&suf_idx) = self.suffix_map.get(&suf_surface) {
                    let suf_entry = &self.suffix_entries[suf_idx];
                    for analysis in &suf_entry.analyses {
                        if Self::is_pure_functional(&analysis.morphemes) {
                            let suf_cost = self.get_suffix_cost(&suf_surface, analysis);
                            let morphemes: Vec<(String, Pos)> = analysis.morphemes.iter()
                                .map(|m| (m.form.clone(), m.pos))
                                .collect();
                            arcs.push(LatticeArc {
                                start: i,
                                end: i + suf_len,
                                morphemes,
                                cost: suf_cost,
                            });
                        }
                    }
                }
            }

            // Strategy C: Contracted forms — content+functional from codebook
            // (codebook entries that start with content morpheme followed by functional)
            for suf_len in 1..=max_suf {
                let suf_surface: String = chars[i..i + suf_len].iter().collect();
                if let Some(&suf_idx) = self.suffix_map.get(&suf_surface) {
                    let suf_entry = &self.suffix_entries[suf_idx];
                    for analysis in &suf_entry.analyses {
                        if analysis.morphemes.len() >= 2
                            && !Self::is_pure_functional(&analysis.morphemes)
                            && !analysis.morphemes.iter().all(|m| is_content_pos(m.pos))
                        {
                            // Mixed content+functional
                            let suf_cost = self.get_suffix_cost(&suf_surface, analysis);
                            let morphemes: Vec<(String, Pos)> = analysis.morphemes.iter()
                                .map(|m| (m.form.clone(), m.pos))
                                .collect();
                            arcs.push(LatticeArc {
                                start: i,
                                end: i + suf_len,
                                morphemes,
                                cost: suf_cost,
                            });
                        }
                    }
                }
            }
        }

        // OOV fallback: single-character arcs for uncovered positions
        // Check which positions have at least one arc starting there
        let mut has_arc = vec![false; n];
        for arc in &arcs {
            has_arc[arc.start] = true;
        }
        for i in 0..n {
            if !has_arc[i] && !chars[i].is_whitespace() {
                let pos = classify_oov_char(chars[i]);
                let surface = chars[i].to_string();
                arcs.push(LatticeArc {
                    start: i,
                    end: i + 1,
                    morphemes: vec![(surface, pos)],
                    cost: self.oov_penalty,
                });
            }
        }

        arcs
    }

    // -----------------------------------------------------------------------
    // Viterbi decoding
    // -----------------------------------------------------------------------

    fn viterbi(&self, text: &str, arcs: &[LatticeArc]) -> (Vec<Token>, f32) {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return (Vec::new(), 0.0);
        }

        // Group arcs by their end position for efficient lookup
        let mut arcs_ending_at: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        for (idx, arc) in arcs.iter().enumerate() {
            arcs_ending_at[arc.end].push(idx);
        }

        // DP states: dp[position] maps (prev_pos, prev_prev_pos) -> (cost, backpointer)
        let mut dp: Vec<HashMap<(u8, u8), (f32, Option<Backpointer>)>> = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            dp.push(HashMap::new());
        }

        // Initial state at position 0
        dp[0].insert((BOS, BOS), (0.0, None));

        // Forward pass
        for pos in 0..=n {
            // Process all arcs ending at this position FIRST
            // (arcs may end at a space position, e.g., "나는" ends where space is)
            for &arc_idx in &arcs_ending_at[pos] {
                let arc = &arcs[arc_idx];
                // Last morpheme POS of this arc determines new state
                let last_pos = arc.morphemes.last().map(|(_, p)| *p as u8).unwrap_or(BOS);

                // Look at all states at arc.start
                let start_states: Vec<((u8, u8), f32)> = dp[arc.start].iter()
                    .map(|(&state, &(cost, _))| (state, cost))
                    .collect();

                for ((prev_pos, prev_prev_pos), prev_cost) in start_states {
                    // Transition cost: trigram(prev_prev_pos, prev_pos, first_morpheme_pos)
                    let first_pos = arc.morphemes.first().map(|(_, p)| *p as u8).unwrap_or(BOS);
                    let transition_cost = self.get_trigram_cost(prev_prev_pos, prev_pos, first_pos);

                    // Word-bigram bonus: context-dependent cost adjustment
                    let first_form = &arc.morphemes[0].0;
                    let wb_bonus = self.get_word_bigram_bonus(first_form, prev_pos, first_pos);

                    // Internal morpheme transitions (if arc has multiple morphemes)
                    let mut internal_cost = 0.0;
                    if arc.morphemes.len() > 1 {
                        let mut pp = prev_prev_pos;
                        let mut p = prev_pos;
                        for (mi, (_, mpos)) in arc.morphemes.iter().enumerate() {
                            if mi == 0 {
                                pp = prev_pos;
                                p = *mpos as u8;
                            } else {
                                internal_cost += self.get_trigram_cost(pp, p, *mpos as u8);
                                pp = p;
                                p = *mpos as u8;
                            }
                        }
                    }

                    let total_cost = prev_cost + arc.cost + transition_cost + internal_cost + wb_bonus;

                    // New state: (last_pos, second_to_last_pos)
                    let new_prev_prev = if arc.morphemes.len() >= 2 {
                        arc.morphemes[arc.morphemes.len() - 2].1 as u8
                    } else {
                        prev_pos
                    };
                    let new_state = (last_pos, new_prev_prev);

                    let entry = dp[pos].entry(new_state).or_insert((f32::INFINITY, None));
                    if total_cost < entry.0 {
                        *entry = (total_cost, Some(Backpointer {
                            prev_pos: arc.start,
                            prev_state: (prev_pos, prev_prev_pos),
                            arc_idx: Some(arc_idx),
                        }));
                    }
                }
            }

            // Space pass-through: propagate states across whitespace
            if pos < n && chars[pos].is_whitespace() {
                let states: Vec<((u8, u8), f32)> = dp[pos].iter()
                    .map(|(&state, &(cost, _))| (state, cost))
                    .collect();
                for (state, cost) in states {
                    let entry = dp[pos + 1].entry(state).or_insert((f32::INFINITY, None));
                    if cost < entry.0 {
                        *entry = (cost, Some(Backpointer {
                            prev_pos: pos,
                            prev_state: state,
                            arc_idx: None,
                        }));
                    }
                }
            }
        }

        // Find best final state
        let final_states = &dp[n];
        if final_states.is_empty() {
            // Fallback: try to find best state at any position
            return self.fallback_tokenize(text);
        }

        let mut best_cost = f32::INFINITY;
        let mut best_state: (u8, u8) = (BOS, BOS);
        for (&state, &(cost, _)) in final_states {
            // Add EOS transition cost
            let eos_cost = self.get_trigram_cost(state.1, state.0, BOS);
            let total = cost + eos_cost;
            if total < best_cost {
                best_cost = total;
                best_state = state;
            }
        }

        // Backtrack
        let mut path_arcs: Vec<usize> = Vec::new();
        let mut cur_pos = n;
        let mut cur_state = best_state;

        loop {
            let bp = match dp[cur_pos].get(&cur_state) {
                Some((_, Some(bp))) => bp.clone(),
                _ => break,
            };
            if let Some(arc_idx) = bp.arc_idx {
                path_arcs.push(arc_idx);
            }
            cur_pos = bp.prev_pos;
            cur_state = bp.prev_state;
            if cur_pos == 0 && cur_state == (BOS, BOS) {
                break;
            }
        }

        path_arcs.reverse();

        // Build tokens from path
        let mut tokens = Vec::new();
        // Build char-to-byte offset map
        let mut char_to_byte = Vec::with_capacity(n + 1);
        let mut byte_off = 0;
        for ch in &chars {
            char_to_byte.push(byte_off);
            byte_off += ch.len_utf8();
        }
        char_to_byte.push(byte_off);

        for &arc_idx in &path_arcs {
            let arc = &arcs[arc_idx];
            for (form, pos) in &arc.morphemes {
                tokens.push(Token {
                    text: normalize_jamo(form),
                    pos: *pos,
                    start: arc.start,
                    end: arc.end,
                    score: None,
                });
            }
        }

        // Post-process: merge consecutive single-char SL/SN tokens
        let mut tokens = Self::merge_sl_sn_tokens(tokens);

        // Post-process: convert 었→았 based on preceding vowel (ㅏ=0, ㅗ=8)
        for i in 1..tokens.len() {
            if tokens[i].pos == Pos::EP && (tokens[i].text == "었" || tokens[i].text == "었었") {
                let prev_text = &tokens[i - 1].text;
                if let Some(last_char) = prev_text.chars().last() {
                    let code = last_char as u32;
                    if code >= 0xAC00 && code <= 0xD7A3 {
                        let vowel = ((code - 0xAC00) % (21 * 28)) / 28;
                        if vowel == 0 || vowel == 8 {
                            if tokens[i].text == "었" {
                                tokens[i].text = "았".to_string();
                            } else {
                                tokens[i].text = "았었".to_string();
                            }
                        }
                    }
                }
            }

            // Also convert 어→아 for EC/EF endings based on preceding vowel harmony
            if (tokens[i].pos == Pos::EC || tokens[i].pos == Pos::EF)
                && tokens[i].text.starts_with("어")
            {
                let prev_text = &tokens[i - 1].text;
                if let Some(last_char) = prev_text.chars().last() {
                    let code = last_char as u32;
                    if code >= 0xAC00 && code <= 0xD7A3 {
                        let vowel = ((code - 0xAC00) % (21 * 28)) / 28;
                        if vowel == 0 || vowel == 8 {
                            tokens[i].text = tokens[i].text.replacen("어", "아", 1);
                        }
                    }
                }
            }
        }

        // Post-process: minimal — NIKL trigram handles most disambiguation
        Self::fix_xsv_xsa(&mut tokens);

        (tokens, best_cost)
    }

    /// Fix JKB → JC for 과/와 connecting nouns.
    fn fix_jc_jkb(tokens: &mut [Token]) {
        if tokens.len() < 2 {
            return;
        }
        for i in 1..tokens.len() {
            if tokens[i].pos != Pos::JKB {
                continue;
            }
            let form = tokens[i].text.as_str();
            if form != "과" && form != "와" && form != "이랑" && form != "랑" && form != "하고" {
                continue;
            }
            // Check if preceded by a noun-like POS
            let prev_pos = tokens[i - 1].pos;
            let preceded_by_noun = matches!(
                prev_pos,
                Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP | Pos::SN | Pos::SL
            );
            if preceded_by_noun {
                // Check if followed by a noun-like POS (or end of sentence / JX)
                let followed_by_noun = if i + 1 < tokens.len() {
                    matches!(
                        tokens[i + 1].pos,
                        Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP
                            | Pos::SN | Pos::SL | Pos::MM | Pos::MAG
                    )
                } else {
                    false
                };
                if followed_by_noun {
                    tokens[i].pos = Pos::JC;
                }
            }
        }
    }

    /// Fix XSV/XSA: 하/되 after NNG → XSV; 하/VA after NNG → XSA.
    /// Only apply when NNG is directly attached (no JKO/JKS intervening).
    fn fix_xsv_xsa(tokens: &mut [Token]) {
        if tokens.len() < 2 {
            return;
        }
        for i in 1..tokens.len() {
            let prev_pos = tokens[i - 1].pos;
            // Only apply after NNG (not NNP/NNB — those are rarely 하다 verb stems)
            if prev_pos != Pos::NNG {
                continue;
            }
            // Don't apply if there's a case marker between NNG and 하/되
            // (e.g., "일을 하다" → 하 stays VV)
            if i >= 2 {
                let pp = tokens[i - 1].pos;
                if matches!(pp, Pos::JKO | Pos::JKS | Pos::JKB) {
                    continue;
                }
            }
            let form = tokens[i].text.as_str();
            if tokens[i].pos == Pos::VV && (form == "하" || form == "되" || form == "시키") {
                tokens[i].pos = Pos::XSV;
            } else if tokens[i].pos == Pos::VA && form == "하" {
                tokens[i].pos = Pos::XSA;
            }
        }
    }

    /// Fix VV → VX for auxiliary verbs after EC (connective endings).
    fn fix_vx(tokens: &mut [Token]) {
        // 있/VV → 있/VX after 고/EC or 어/아/EC (progressive/state)
        // 하/VV → 하/VX after 고/EC (repeated action)
        // 보/VV → 보/VX after 어/아/EC (try)
        // 주/VV → 주/VX after 어/아/EC (for someone)
        // 지/VV → 지/VX after 어/아/EC (continuation)
        for i in 1..tokens.len() {
            if tokens[i].pos != Pos::VV {
                continue;
            }
            if tokens[i - 1].pos != Pos::EC {
                continue;
            }
            let form = tokens[i].text.as_str();
            let prev_form = tokens[i - 1].text.as_str();
            match form {
                "있" | "없" => {
                    // 고 있다, 어 있다
                    if prev_form == "고" || prev_form == "어" || prev_form == "아" {
                        tokens[i].pos = Pos::VX;
                    }
                }
                "하" => {
                    // 지 않다 / ~고 하다 pattern (하 as VX)
                    if prev_form == "지" {
                        tokens[i].pos = Pos::VX;
                    }
                }
                "보" | "주" | "지" | "오" | "가" | "내" | "나" | "버리" | "놓" | "두" => {
                    // 어/아 + auxiliary verb pattern
                    if prev_form == "어" || prev_form == "아" {
                        tokens[i].pos = Pos::VX;
                    }
                }
                _ => {}
            }
        }
    }

    /// Fix NNG → NNB for common dependency nouns after ETM.
    fn fix_nnb(tokens: &mut [Token]) {
        // Conservative list: only clearly NNB items after ETM
        // Removed: 때(often NNG), 일(often NNG), 가지(often VX), 곳(often NNG)
        const NNB_AFTER_ETM: &[&str] = &[
            "수", "것", "데", "바", "번", "개", "군", "줄",
            "뿐", "채", "척", "듯", "리", "셈", "나름", "탓", "만큼",
        ];
        for i in 1..tokens.len() {
            if tokens[i].pos != Pos::NNG && tokens[i].pos != Pos::VV {
                continue;
            }
            if tokens[i - 1].pos == Pos::ETM {
                let form = tokens[i].text.as_str();
                if NNB_AFTER_ETM.iter().any(|&w| w == form) {
                    tokens[i].pos = Pos::NNB;
                }
            }
        }
    }

    /// Fix NNG → XSN/XPN for common suffix/prefix morphemes.
    fn fix_xsn_xpn(tokens: &mut [Token]) {
        // Conservative: only clear suffixes. Removed: 제(NP 제), 상/장(NNG)
        const XSN_FORMS: &[&str] = &[
            "성", "형", "적", "식", "계", "권", "화", "률", "율", "급",
        ];
        const XPN_FORMS: &[&str] = &[
            "비", "재", "초", "무", "탈", "반", "불",
        ];
        for i in 0..tokens.len() {
            if tokens[i].pos != Pos::NNG {
                continue;
            }
            let form = tokens[i].text.as_str();
            // XSN: single-char after NNG
            if form.chars().count() == 1 && i > 0 && tokens[i - 1].pos == Pos::NNG {
                if XSN_FORMS.iter().any(|&w| w == form) {
                    tokens[i].pos = Pos::XSN;
                }
            }
            // XPN: single-char before NNG
            if form.chars().count() == 1 && i + 1 < tokens.len() && tokens[i + 1].pos == Pos::NNG {
                if XPN_FORMS.iter().any(|&w| w == form) {
                    tokens[i].pos = Pos::XPN;
                }
            }
        }
    }

    /// Fix NP/JKS → MM for demonstratives before nouns.
    fn fix_mm(tokens: &mut [Token]) {
        for i in 0..tokens.len() {
            let form = tokens[i].text.as_str();
            let cur_pos = tokens[i].pos;

            // Must be followed by a noun
            let followed_by_noun = i + 1 < tokens.len() && matches!(
                tokens[i + 1].pos,
                Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::XR
            );
            if !followed_by_noun {
                continue;
            }

            // 이/저: only convert if NOT preceded by a noun (avoid JKS→MM mistake)
            // Note: "그" excluded — too ambiguous with IC/NP in spoken Korean
            if (form == "이" || form == "저") && (cur_pos == Pos::NP || cur_pos == Pos::JKS) {
                let preceded_by_noun = i > 0 && matches!(
                    tokens[i - 1].pos,
                    Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP
                        | Pos::SN | Pos::SL | Pos::XSN | Pos::XSA | Pos::XSV
                );
                if !preceded_by_noun {
                    tokens[i].pos = Pos::MM;
                }
                continue;
            }

            // 새/각/온: convert from NNG to MM before nouns
            // Removed: 현/전 — too ambiguous with NNG
            if cur_pos == Pos::NNG && (form == "새" || form == "각" || form == "온") {
                tokens[i].pos = Pos::MM;
            }
        }
    }

    /// Fix JKS → JKC for 가/이 before 되다/아니다.
    fn fix_jkc(tokens: &mut [Token]) {
        for i in 0..tokens.len() {
            if tokens[i].pos != Pos::JKS {
                continue;
            }
            let form = tokens[i].text.as_str();
            if form != "가" && form != "이" {
                continue;
            }
            // Check if followed by 되/아니
            if i + 1 < tokens.len() {
                let next_form = tokens[i + 1].text.as_str();
                if next_form == "되" || next_form == "아니" {
                    tokens[i].pos = Pos::JKC;
                }
            }
        }
    }

    /// Merge consecutive single-char SL or SN tokens that are adjacent.
    fn merge_sl_sn_tokens(tokens: Vec<Token>) -> Vec<Token> {
        if tokens.is_empty() {
            return tokens;
        }
        let mut merged: Vec<Token> = Vec::new();
        for token in tokens {
            if let Some(last) = merged.last_mut() {
                if last.pos == token.pos
                    && (last.pos == Pos::SL || last.pos == Pos::SN)
                    && last.end == token.start
                {
                    last.text.push_str(&token.text);
                    last.end = token.end;
                    continue;
                }
            }
            merged.push(token);
        }
        merged
    }

    /// Fallback tokenization when Viterbi fails to reach end.
    fn fallback_tokenize(&self, text: &str) -> (Vec<Token>, f32) {
        let chars: Vec<char> = text.chars().collect();
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }
            let pos = classify_oov_char(chars[i]);
            tokens.push(Token {
                text: normalize_jamo(&chars[i].to_string()),
                pos,
                start: i,
                end: i + 1,
                score: None,
            });
            i += 1;
        }
        let tokens = Self::merge_sl_sn_tokens(tokens);
        (tokens, self.oov_penalty * chars.len() as f32)
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Analyze text, returning morphological tokens.
    pub fn analyze(&self, text: &str) -> AnalyzeResult {
        if text.is_empty() {
            return AnalyzeResult {
                tokens: vec![],
                score: 0.0,
                elapsed_ms: 0.0,
            };
        }

        let t0 = now_ms();

        // Hybrid: eojeol cache + Viterbi fallback
        if !self.eojeol_cache.is_empty() {
            let mut tokens = Vec::new();
            let mut total_score = 0.0f32;
            let mut char_offset = 0usize;

            // Find eojeol boundaries in original text (split by whitespace)
            for eojeol in text.split_whitespace() {
                // Find this eojeol's position in the original text
                let eojeol_start = text[char_offset..].find(eojeol)
                    .map(|p| char_offset + p)
                    .unwrap_or(char_offset);
                let eojeol_char_start = text[..eojeol_start].chars().count();
                let eojeol_char_end = eojeol_char_start + eojeol.chars().count();

                if let Some(cached_morphs) = self.eojeol_cache.get(eojeol) {
                    for (form, pos) in cached_morphs {
                        tokens.push(Token {
                            text: form.clone(),
                            pos: *pos,
                            start: eojeol_char_start,
                            end: eojeol_char_end,
                            score: None,
                        });
                    }
                } else {
                    let arcs = self.build_lattice(eojeol);
                    let (eojeol_tokens, score) = self.viterbi(eojeol, &arcs);
                    for token in eojeol_tokens {
                        tokens.push(Token {
                            start: eojeol_char_start,
                            end: eojeol_char_end,
                            ..token
                        });
                    }
                    total_score += score;
                }

                char_offset = eojeol_start + eojeol.len();
            }

            return AnalyzeResult {
                tokens,
                score: total_score,
                elapsed_ms: now_ms() - t0,
            };
        }

        // No cache: full Viterbi with eojeol-level spans
        let arcs = self.build_lattice(text);
        let (raw_tokens, score) = self.viterbi(text, &arcs);

        // Assign eojeol-level spans to each token
        let mut tokens = Vec::with_capacity(raw_tokens.len());
        let mut eojeol_boundaries: Vec<(usize, usize)> = Vec::new();
        let mut char_offset = 0;
        for eojeol in text.split_whitespace() {
            let byte_start = text[char_offset..].find(eojeol)
                .map(|p| char_offset + p).unwrap_or(char_offset);
            let cs = text[..byte_start].chars().count();
            let ce = cs + eojeol.chars().count();
            eojeol_boundaries.push((cs, ce));
            char_offset = byte_start + eojeol.len();
        }

        for token in raw_tokens {
            // Find which eojeol this token belongs to
            let (es, ee) = eojeol_boundaries.iter()
                .find(|&&(s, e)| token.start >= s && token.start < e)
                .copied()
                .unwrap_or((token.start, token.end));
            tokens.push(Token { start: es, end: ee, ..token });
        }

        AnalyzeResult {
            tokens,
            score,
            elapsed_ms: now_ms() - t0,
        }
    }

    /// Analyze returning top-N results (currently returns single best).
    pub fn analyze_topn(&self, text: &str, _n: usize) -> Vec<AnalyzeResult> {
        vec![self.analyze(text)]
    }

    /// Extract surface forms from analysis.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.analyze(text)
            .tokens
            .into_iter()
            .map(|t| t.text)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_oov() {
        assert_eq!(classify_oov_char('A'), Pos::SL);
        assert_eq!(classify_oov_char('z'), Pos::SL);
        assert_eq!(classify_oov_char('3'), Pos::SN);
        assert_eq!(classify_oov_char('.'), Pos::SF);
        assert_eq!(classify_oov_char(','), Pos::SP);
        assert_eq!(classify_oov_char('('), Pos::SS);
        assert_eq!(classify_oov_char('가'), Pos::NNG);
        assert_eq!(classify_oov_char('@'), Pos::SW);
    }

    #[test]
    fn test_is_content_pos() {
        assert!(is_content_pos(Pos::NNG));
        assert!(is_content_pos(Pos::VV));
        assert!(!is_content_pos(Pos::JKS));
        assert!(!is_content_pos(Pos::EF));
    }

    #[test]
    fn test_is_functional_pos() {
        assert!(is_functional_pos(Pos::JKS));
        assert!(is_functional_pos(Pos::EF));
        assert!(!is_functional_pos(Pos::NNG));
        assert!(!is_functional_pos(Pos::VV));
    }

    #[test]
    fn test_ascii_runs() {
        let chars: Vec<char> = "Hello 123 world".chars().collect();
        let runs = CodebookAnalyzer::find_ascii_runs(&chars);
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0], (0, 5, Pos::SL));
        assert_eq!(runs[1], (6, 9, Pos::SN));
        assert_eq!(runs[2], (10, 15, Pos::SL));
    }

    #[test]
    fn test_merge_sl_sn() {
        let tokens = vec![
            Token { text: "H".into(), pos: Pos::SL, start: 0, end: 1, score: None },
            Token { text: "i".into(), pos: Pos::SL, start: 1, end: 2, score: None },
            Token { text: "3".into(), pos: Pos::SN, start: 3, end: 4, score: None },
        ];
        let merged = CodebookAnalyzer::merge_sl_sn_tokens(tokens);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].text, "Hi");
        assert_eq!(merged[1].text, "3");
    }
}
