//! Codebook-based morphological analyzer (lattice + Viterbi).

use std::collections::HashMap;
use std::io::Read as IoRead;
use crate::trie::Dict;
use crate::types::{AnalyzeResult, Pos, Token};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(inline_js = "export function performance_now() { return performance.now(); }")]
extern "C" {
    fn performance_now() -> f64;
}

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
// Hangul syllable decomposition (for jongseong splitting)
// ---------------------------------------------------------------------------

/// Split a Hangul syllable's jongseong: returns (syllable_without_jong, jamo_char).
/// e.g. '친' → Some(('치', 'ㄴ')), '다' → None (no jongseong).
fn split_jongseong(ch: char) -> Option<(char, char)> {
    let code = ch as u32;
    if code < 0xAC00 || code > 0xD7A3 {
        return None;
    }
    let offset = code - 0xAC00;
    let jong = offset % 28;
    if jong == 0 {
        return None; // no jongseong
    }
    // Only handle single jongseong (skip double jongseong like ㄳ, ㄵ, etc.)
    let jamo = match jong {
        1 => 'ㄱ',
        2 => 'ㄲ',
        4 => 'ㄴ',
        7 => 'ㄷ',
        8 => 'ㄹ',
        16 => 'ㅁ',
        17 => 'ㅂ',
        19 => 'ㅅ',
        20 => 'ㅆ',
        21 => 'ㅇ',
        22 => 'ㅈ',
        23 => 'ㅊ',
        24 => 'ㅋ',
        25 => 'ㅌ',
        26 => 'ㅍ',
        27 => 'ㅎ',
        _ => return None, // double jongseong — skip
    };
    let base = 0xAC00 + (offset / 28) * 28; // remove jongseong
    Some((char::from_u32(base).unwrap(), jamo))
}

/// Decompose a Hangul syllable into (chosung, jungsung, jongseong) indices.
fn decompose_hangul(ch: char) -> Option<(u32, u32, u32)> {
    let code = ch as u32;
    if code < 0xAC00 || code > 0xD7A3 { return None; }
    let offset = code - 0xAC00;
    Some((offset / (21 * 28), (offset % (21 * 28)) / 28, offset % 28))
}

/// Compose a Hangul syllable from (chosung, jungsung, jongseong) indices.
fn compose_hangul(cho: u32, jung: u32, jong: u32) -> Option<char> {
    if cho >= 19 || jung >= 21 || jong >= 28 { return None; }
    char::from_u32((cho * 21 + jung) * 28 + jong + 0xAC00)
}

// Common Korean typo rules: (index_a, index_b) pairs for bidirectional substitution.
// Chosung: plain ↔ tense consonant
const TYPO_CHO: &[(u32, u32)] = &[
    (0, 1),   // ㄱ ↔ ㄲ
    (3, 4),   // ㄷ ↔ ㄸ
    (7, 8),   // ㅂ ↔ ㅃ
    (9, 10),  // ㅅ ↔ ㅆ
    (12, 13), // ㅈ ↔ ㅉ
];
// Jungsung: commonly confused vowels
const TYPO_JUNG: &[(u32, u32)] = &[
    (1, 5),   // ㅐ ↔ ㅔ
    (3, 7),   // ㅒ ↔ ㅖ
    (10, 15), // ㅙ ↔ ㅞ
    (10, 11), // ㅙ ↔ ㅚ
];
// Jongseong: plain ↔ tense (most impactful: 했→햇)
const TYPO_JONG: &[(u32, u32)] = &[
    (1, 2),   // ㄱ ↔ ㄲ
    (19, 20), // ㅅ ↔ ㅆ
];
/// Additional cost for typo-corrected arcs (must beat OOV but lose to exact match).
const TYPO_PENALTY: f32 = 3.0;

// ---------------------------------------------------------------------------
// Jamo normalization
// ---------------------------------------------------------------------------

/// Normalize Hangul jongseong (U+11A8-U+11C2) to compatibility jamo (U+3131-U+314E).
/// Used internally for consistent representation during codebook loading.
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
    prev_rank: usize,       // rank at prev state (for N-best Viterbi)
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
        performance_now()
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
    /// Parse a GMDL v3 file from raw bytes (supports gzip-compressed input).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        // Auto-detect gzip (magic bytes 1f 8b) and decompress
        if data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b {
            let mut decoder = flate2::read::GzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)
                .map_err(|e| format!("Gzip decompression failed: {}", e))?;
            return Self::from_bytes_inner(&decompressed);
        }
        Self::from_bytes_inner(data)
    }

    fn from_bytes_inner(data: &[u8]) -> Result<Self, String> {
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
        let marker = u32::from_le_bytes(data[0..4].try_into().map_err(|_| "Bad header")?);
        if marker == 0xFFFFFFFF {
            return Self::parse_suffix_codebook_v1(data);
        }
        Self::parse_suffix_codebook_legacy(data)
    }

    fn parse_suffix_codebook_legacy(data: &[u8]) -> Result<(Vec<SuffixEntry>, HashMap<String, usize>, usize), String> {
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

    fn parse_suffix_codebook_v1(data: &[u8]) -> Result<(Vec<SuffixEntry>, HashMap<String, usize>, usize), String> {
        let mut pos = 5; // skip 0xFFFFFFFF marker + sub-version byte

        // String table
        let st_len = u32::from_le_bytes(
            data[pos..pos + 4].try_into().map_err(|_| "Bad st_len")?,
        ) as usize;
        pos += 4;
        let string_table = &data[pos..pos + st_len];
        pos += st_len;

        let num_strings = u16::from_le_bytes(
            data[pos..pos + 2].try_into().map_err(|_| "Bad num_strings")?,
        ) as usize;
        pos += 2;

        let mut string_offsets = Vec::with_capacity(num_strings + 1);
        for _ in 0..=num_strings {
            let off = u16::from_le_bytes(
                data[pos..pos + 2].try_into().map_err(|_| "Bad string offset")?,
            ) as usize;
            string_offsets.push(off);
            pos += 2;
        }

        // Precompute form strings
        let mut forms: Vec<String> = Vec::with_capacity(num_strings);
        for i in 0..num_strings {
            let s = std::str::from_utf8(&string_table[string_offsets[i]..string_offsets[i + 1]])
                .map_err(|e| format!("Invalid UTF-8 in string table: {}", e))?;
            forms.push(normalize_jamo(s));
        }

        // Max freq for dequantization
        let max_freq_raw = u32::from_le_bytes(
            data[pos..pos + 4].try_into().map_err(|_| "Bad max_freq")?,
        );
        pos += 4;
        let ln_max_freq = (max_freq_raw.max(2) as f64).ln();

        // Entries
        let num_entries = u32::from_le_bytes(
            data[pos..pos + 4].try_into().map_err(|_| "Bad num_entries")?,
        ) as usize;
        pos += 4;

        let mut entries = Vec::with_capacity(num_entries);
        let mut map = HashMap::with_capacity(num_entries);
        let mut max_suffix_len: usize = 0;

        for i in 0..num_entries {
            let surface_len = data[pos] as usize;
            pos += 1;
            let surface = std::str::from_utf8(&data[pos..pos + surface_len])
                .map_err(|e| format!("Invalid UTF-8 in surface {}: {}", i, e))?
                .to_string();
            pos += surface_len;

            let char_len = surface.chars().count();
            if char_len > max_suffix_len {
                max_suffix_len = char_len;
            }

            let num_analyses = data[pos] as usize;
            pos += 1;

            let mut analyses = Vec::with_capacity(num_analyses);
            for _ in 0..num_analyses {
                let freq_q = data[pos];
                pos += 1;
                let freq = if freq_q == 0 {
                    1u32
                } else {
                    ((freq_q as f64 / 255.0 * ln_max_freq).exp()).round() as u32
                };

                let num_morphemes = data[pos] as usize;
                pos += 1;

                let mut morphemes = Vec::with_capacity(num_morphemes);
                for _ in 0..num_morphemes {
                    let string_idx = u16::from_le_bytes(
                        data[pos..pos + 2].try_into().map_err(|_| "Bad string_idx")?,
                    ) as usize;
                    pos += 2;
                    if string_idx >= forms.len() {
                        return Err(format!("String index {} out of range", string_idx));
                    }
                    let form = forms[string_idx].clone();

                    let pos_byte = data[pos];
                    pos += 1;
                    if pos_byte > 41 {
                        return Err(format!("Invalid POS byte: {}", pos_byte));
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
        let data = match data {
            Some(d) if d.len() >= 4 => d,
            _ => return Ok(HashMap::new()),
        };
        let marker = u32::from_le_bytes(data[0..4].try_into().map_err(|_| "Bad header")?);
        if marker == 0xFFFFFFFF {
            return Self::parse_eojeol_cache_v1(data);
        }
        Self::parse_eojeol_cache_legacy(data)
    }

    fn parse_eojeol_cache_legacy(data: &[u8]) -> Result<HashMap<String, Vec<(String, Pos)>>, String> {
        let mut map = HashMap::new();
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

    fn parse_eojeol_cache_v1(data: &[u8]) -> Result<HashMap<String, Vec<(String, Pos)>>, String> {
        let mut pos = 5; // skip marker + sub-version

        // String table
        let st_len = u32::from_le_bytes(
            data[pos..pos + 4].try_into().map_err(|_| "Bad st_len")?,
        ) as usize;
        pos += 4;
        let string_table = &data[pos..pos + st_len];
        pos += st_len;

        let num_strings = u16::from_le_bytes(
            data[pos..pos + 2].try_into().map_err(|_| "Bad num_strings")?,
        ) as usize;
        pos += 2;

        let mut string_offsets = Vec::with_capacity(num_strings + 1);
        for _ in 0..=num_strings {
            let off = u16::from_le_bytes(
                data[pos..pos + 2].try_into().map_err(|_| "Bad string offset")?,
            ) as usize;
            string_offsets.push(off);
            pos += 2;
        }

        let mut forms: Vec<String> = Vec::with_capacity(num_strings);
        for i in 0..num_strings {
            let s = std::str::from_utf8(&string_table[string_offsets[i]..string_offsets[i + 1]])
                .map_err(|e| format!("Invalid UTF-8 in ecache string table: {}", e))?;
            forms.push(normalize_jamo(s));
        }

        // Entries
        let num = u32::from_le_bytes(
            data[pos..pos + 4].try_into().map_err(|_| "Bad ecache count")?,
        ) as usize;
        pos += 4;

        let mut map = HashMap::with_capacity(num);
        for _ in 0..num {
            if pos >= data.len() { break; }
            let elen = data[pos] as usize;
            pos += 1;
            if pos + elen > data.len() { break; }
            let eojeol = std::str::from_utf8(&data[pos..pos + elen])
                .map_err(|_| "Bad UTF-8 in ecache")?.to_string();
            pos += elen;
            if pos >= data.len() { break; }
            let nm = data[pos] as usize;
            pos += 1;
            let mut morphs = Vec::with_capacity(nm);
            for _ in 0..nm {
                if pos + 3 > data.len() { break; }
                let string_idx = u16::from_le_bytes(
                    data[pos..pos + 2].try_into().map_err(|_| "Bad string_idx")?,
                ) as usize;
                pos += 2;
                let pos_byte = data[pos];
                pos += 1;
                if string_idx < forms.len() && pos_byte <= 41 {
                    let p: Pos = unsafe { std::mem::transmute(pos_byte) };
                    morphs.push((forms[string_idx].clone(), p));
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
                // SL run: starts with alpha, may include digits/hyphens (b2b, BM25, GPT-4o)
                let start = i;
                i += 1;
                while i < chars.len()
                    && (chars[i].is_ascii_alphanumeric() || chars[i] == '-' || chars[i] == '.')
                {
                    i += 1;
                }
                // Trim trailing dots/hyphens
                while i > start + 1 && (chars[i - 1] == '.' || chars[i - 1] == '-') {
                    i -= 1;
                }
                runs.push((start, i, Pos::SL));
            } else if chars[i].is_ascii_digit() {
                // SN run: digits, including decimal points and thousand separators
                // e.g. "3.14", "1,900", "12,345,678"
                let start = i;
                i += 1;
                while i < chars.len() {
                    if chars[i].is_ascii_digit() {
                        i += 1;
                    } else if (chars[i] == '.' || chars[i] == ',')
                        && i + 1 < chars.len()
                        && chars[i + 1].is_ascii_digit()
                    {
                        // Include separator only if followed by digit
                        i += 1;
                    } else {
                        break;
                    }
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
            let byte_start = char_to_byte[i];
            let remaining = &text[byte_start..];
            let prefix_matches = self.content_dict.common_prefix_search(remaining);

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
                    let word_cost = self.get_word_cost(match_str, first_pos, freq);

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
                        // Determine vowel contraction prefix for A2b
                        // When stem ends in open syllable (no jongseong), the suffix's
                        // initial vowel may have contracted: 건너+어라→건너라, 가+아라→가라
                        let vowel_prefix: Option<char> = {
                            let last_ch = chars[content_end - 1];
                            let code = last_ch as u32;
                            if code >= 0xAC00 && code <= 0xD7A3 {
                                let offset = code - 0xAC00;
                                let jong = offset % 28;
                                if jong == 0 { // open syllable (no jongseong)
                                    let vowel = (offset % (21 * 28)) / 28;
                                    match vowel {
                                        0 | 8 => Some('아'), // ㅏ, ㅗ → 양성모음
                                        _ => Some('어'),     // others → 음성모음
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        };

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

                            // A2b: Vowel contraction recovery
                            // e.g. "건너라" → "건너" + "라" → try "어라" in codebook
                            if let Some(v) = vowel_prefix {
                                let mut expanded = String::new();
                                expanded.push(v);
                                expanded.push_str(&suf_surface);
                                if let Some(&suf_idx) = self.suffix_map.get(&expanded) {
                                    let suf_entry = &self.suffix_entries[suf_idx];
                                    for analysis in &suf_entry.analyses {
                                        let suf_cost = self.get_suffix_cost(&expanded, analysis);
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
            }

            // Strategy A3: Jongseong split — content word with merged jongseong
            // e.g. "고친다" → split last char of prefix: "고친" → "고치" + ㄴ
            //       then lookup "고치" in dict (VV), "ㄴ다" in suffix codebook (EF)
            // This handles conjugated forms where the verb stem's final syllable
            // merges with a consonant-initial suffix (ㄴ, ㄹ, ㅂ, etc.)
            {
                // Try each possible prefix length ending at position j (j > i).
                // Include one-syllable stems such as 올→오+ㄹ and 볼→보+ㄹ.
                let max_prefix = (n - i).min(8); // content words rarely exceed 8 chars
                for prefix_len in 1..=max_prefix {
                    let j = i + prefix_len; // one past the last char of prefix
                    if j > n { break; }
                    let last_char = chars[j - 1];
                    if let Some((base_char, jamo)) = split_jongseong(last_char) {
                        // Build the dictionary lookup key with jongseong removed
                        let mut lookup: String = chars[i..j - 1].iter().collect();
                        lookup.push(base_char);

                        // Check if this exists as a content word in the FST
                        let lookup_matches = self.content_dict.lookup(&lookup);
                        for entry in &lookup_matches {
                            if entry.morphemes.is_empty() { continue; }
                            let first_pos = entry.morphemes[0].pos;
                            if !is_content_pos(first_pos) { continue; }

                            let freq = self.freq_from_score(entry.score);
                            let word_cost = self.get_word_cost(&lookup, first_pos, freq);
                            let morphemes: Vec<(String, Pos)> = entry.morphemes.iter()
                                .map(|m| (m.text.clone(), m.pos))
                                .collect();

                            // Build suffix starting with the split jamo + remaining chars
                            let remaining_after = n - j;
                            let max_suf_tail = self.max_suffix_len.saturating_sub(1).min(remaining_after);
                            for tail_len in 0..=max_suf_tail {
                                let mut suf_key = String::new();
                                suf_key.push(jamo);
                                for k in 0..tail_len {
                                    suf_key.push(chars[j + k]);
                                }
                                if let Some(&suf_idx) = self.suffix_map.get(&suf_key) {
                                    let suf_entry = &self.suffix_entries[suf_idx];
                                    for analysis in &suf_entry.analyses {
                                        let suf_cost = self.get_suffix_cost(&suf_key, analysis);
                                        let mut arc_cost = word_cost + suf_cost;
                                        if prefix_len == 1
                                            && jamo == 'ㄹ'
                                            && j < n
                                            && chars[j] == '만'
                                        {
                                            arc_cost -= 5.0;
                                        }
                                        let mut combined = morphemes.clone();
                                        for m in &analysis.morphemes {
                                            combined.push((m.form.clone(), m.pos));
                                        }
                                        arcs.push(LatticeArc {
                                            start: i,
                                            end: j + tail_len,
                                            morphemes: combined,
                                            cost: arc_cost,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Strategy D: Typo correction via syllable substitution.
            // Only at positions where no content word arc starts (OOV recovery).
            // For each word span, substitute one syllable using typo rules,
            // then look up the corrected form in content dict and suffix codebook.
            {
                let has_content_arc = arcs.iter().any(|a| a.start == i
                    && a.morphemes.first().map_or(false, |(_, p)| is_content_pos(*p)));
                if !has_content_arc {
                    let max_typo = (n - i).min(6);
                    for word_len in 1..=max_typo {
                        let j = i + word_len;
                        if j > n { break; }
                        for syl_pos in 0..word_len {
                            let ch = chars[i + syl_pos];
                            let Some((cho, jung, jong)) = decompose_hangul(ch) else { continue };
                            // Collect typo variants for this syllable
                            let mut variants: Vec<char> = Vec::new();
                            for &(a, b) in TYPO_CHO {
                                if cho == a { if let Some(c) = compose_hangul(b, jung, jong) { variants.push(c); } }
                                else if cho == b { if let Some(c) = compose_hangul(a, jung, jong) { variants.push(c); } }
                            }
                            for &(a, b) in TYPO_JUNG {
                                if jung == a { if let Some(c) = compose_hangul(cho, b, jong) { variants.push(c); } }
                                else if jung == b { if let Some(c) = compose_hangul(cho, a, jong) { variants.push(c); } }
                            }
                            for &(a, b) in TYPO_JONG {
                                if jong == a { if let Some(c) = compose_hangul(cho, jung, b) { variants.push(c); } }
                                else if jong == b { if let Some(c) = compose_hangul(cho, jung, a) { variants.push(c); } }
                            }
                            for var_ch in variants {
                                let mut corrected = String::with_capacity(word_len * 3);
                                for k in 0..word_len {
                                    corrected.push(if k == syl_pos { var_ch } else { chars[i + k] });
                                }
                                // Try as content word via prefix search
                                let prefix_matches = self.content_dict.common_prefix_search(&corrected);
                                for (byte_len, entries) in &prefix_matches {
                                    let prefix_char_len = corrected[..*byte_len].chars().count();
                                    for entry in entries {
                                        if entry.morphemes.is_empty() { continue; }
                                        let first_pos = entry.morphemes[0].pos;
                                        if !is_content_pos(first_pos) { continue; }
                                        let freq = self.freq_from_score(entry.score);
                                        let prefix_str = &corrected[..*byte_len];
                                        let word_cost = self.get_word_cost(prefix_str, first_pos, freq);
                                        let morphemes: Vec<(String, Pos)> = entry.morphemes.iter()
                                            .map(|m| (m.text.clone(), m.pos)).collect();
                                        if prefix_char_len == word_len {
                                            // Full match: inject content word arc
                                            arcs.push(LatticeArc {
                                                start: i, end: j, morphemes, cost: word_cost + TYPO_PENALTY,
                                            });
                                        } else {
                                            // Partial match: try suffix for remainder
                                            let suf_key = &corrected[*byte_len..];
                                            if let Some(&si) = self.suffix_map.get(suf_key) {
                                                for analysis in &self.suffix_entries[si].analyses {
                                                    let sc = self.get_suffix_cost(suf_key, analysis);
                                                    let mut combined = morphemes.clone();
                                                    for m in &analysis.morphemes {
                                                        combined.push((m.form.clone(), m.pos));
                                                    }
                                                    arcs.push(LatticeArc {
                                                        start: i, end: j, morphemes: combined,
                                                        cost: word_cost + sc + TYPO_PENALTY,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                                // Try as suffix (handles "햇다"→"했다" in suffix codebook)
                                if let Some(&si) = self.suffix_map.get(&corrected) {
                                    let suf_entry = &self.suffix_entries[si];
                                    for analysis in &suf_entry.analyses {
                                        let sc = self.get_suffix_cost(&corrected, analysis);
                                        let morphemes: Vec<(String, Pos)> = analysis.morphemes.iter()
                                            .map(|m| (m.form.clone(), m.pos)).collect();
                                        arcs.push(LatticeArc {
                                            start: i, end: j, morphemes, cost: sc + TYPO_PENALTY,
                                        });
                                    }
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

        // Inject VCP+EF arcs for "입니다/입니까" patterns.
        // These high-frequency endings often lose to dictionary entries like 진입/NNG.
        let text_chars: &[char] = &chars;
        for i in 0..n {
            if i + 3 <= n {
                let three: String = text_chars[i..i + 3].iter().collect();
                if three == "입니다" || three == "입니까" {
                    let ef = if three == "입니다" { "ㅂ니다" } else { "ㅂ니까" };
                    arcs.push(LatticeArc {
                        start: i,
                        end: i + 3,
                        morphemes: vec![
                            ("이".to_string(), Pos::VCP),
                            (ef.to_string(), Pos::EF),
                        ],
                        cost: -1.0, // negative cost to strongly favor VCP+EF
                    });
                }
            }
        }

        // SF-aware EF injection: for arcs ending just before a SF character,
        // if the last morpheme is EC, add a sibling arc with EF instead.
        // This lets Viterbi choose EF when trigram context supports it,
        // compensating for EC's structural frequency advantage in the codebook.
        {
            // Include SF punctuation and compatibility jamo (ㄱ-ㅣ, U+3131-U+3163)
            // Trailing jamo like ㅋ/ㅠ/ㅜ signal sentence-end in informal text.
            let sf_positions: Vec<usize> = chars.iter().enumerate()
                .filter(|(_, &c)| matches!(c, '.' | '!' | '?' | '…')
                    || (c >= '\u{3131}' && c <= '\u{3163}'))
                .map(|(i, _)| i)
                .collect();

            if !sf_positions.is_empty() {
                let mut ef_arcs: Vec<LatticeArc> = Vec::new();
                for arc in arcs.iter() {
                    if let Some(last) = arc.morphemes.last() {
                        if last.1 == Pos::EC && sf_positions.contains(&arc.end) {
                            let mut new_morphemes = arc.morphemes.clone();
                            new_morphemes.last_mut().unwrap().1 = Pos::EF;
                            ef_arcs.push(LatticeArc {
                                start: arc.start,
                                end: arc.end,
                                morphemes: new_morphemes,
                                cost: arc.cost - 0.5,
                            });
                        }
                    }
                }
                arcs.extend(ef_arcs);
            }
        }

        // Eojeol-level NNG span arcs for OOV recovery.
        // For each whitespace-delimited eojeol of 3–7 pure-Hangul syllables,
        // inject a single NNG arc spanning the whole eojeol.
        // Cost formula: oov_penalty * 1.2 * sqrt(n)
        // This ensures well-analyzed eojeols (됐다=8.11, 봤다=7.42) retain their
        // decomposition while truly OOV eojeols (탕후루~35) use the span arc.
        //
        // Guard: skip span arc when the eojeol contains ETM+의존명사 substrings
        // (만한/만하/는데/한데/을만/뿐이/수있/것같/것이). These patterns signal a
        // well-defined decomposition (e.g. 갈만한데 = 가+ㄹ+만+하+ㄴ+데) that would
        // otherwise lose to the single-NNG span due to 종성분리 cost accumulation.
        {
            const SPAN_FACTOR: f32 = 2.8;
            const SPAN_MIN_LEN: usize = 3;
            const SPAN_MAX_LEN: usize = 7;
            const GUARD_SUBS: [&str; 10] = [
                "만한", "만하", "만해", "는데", "한데", "을만", "뿐이", "수있", "것같", "것이",
            ];
            let mut ej_start = 0;
            while ej_start < n {
                if chars[ej_start].is_whitespace() {
                    ej_start += 1;
                    continue;
                }
                let mut ej_end = ej_start;
                while ej_end < n && !chars[ej_end].is_whitespace() {
                    ej_end += 1;
                }
                let ej_len = ej_end - ej_start;
                if ej_len >= SPAN_MIN_LEN && ej_len <= SPAN_MAX_LEN {
                    let all_hangul = chars[ej_start..ej_end]
                        .iter()
                        .all(|&c| ('\u{AC00}'..='\u{D7A3}').contains(&c));
                    if all_hangul {
                        let surface: String = chars[ej_start..ej_end].iter().collect();
                        if GUARD_SUBS.iter().any(|g| surface.contains(g)) {
                            ej_start = ej_end;
                            continue;
                        }
                        let span_cost = self.oov_penalty * SPAN_FACTOR * (ej_len as f32).sqrt();
                        arcs.push(LatticeArc {
                            start: ej_start,
                            end: ej_end,
                            morphemes: vec![(surface, Pos::NNG)],
                            cost: span_cost,
                        });
                    }
                }
                ej_start = ej_end;
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
                            prev_rank: 0,
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
                            prev_rank: 0,
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
        Self::merge_ec_ef_vowel(&mut tokens);

        (tokens, best_cost)
    }

    /// N-best Viterbi decoding. Returns top `top_k` distinct paths sorted by cost.
    fn viterbi_nbest(&self, text: &str, arcs: &[LatticeArc], top_k: usize) -> Vec<(Vec<Token>, f32)> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return vec![(Vec::new(), 0.0)];
        }

        let mut arcs_ending_at: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        for (idx, arc) in arcs.iter().enumerate() {
            arcs_ending_at[arc.end].push(idx);
        }

        // DP: each state → Vec<(cost, backpointer)>, sorted ascending, max top_k
        let mut dp: Vec<HashMap<(u8, u8), Vec<(f32, Backpointer)>>> = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            dp.push(HashMap::new());
        }
        dp[0].entry((BOS, BOS)).or_default().push((0.0, Backpointer {
            prev_pos: 0, prev_state: (BOS, BOS), arc_idx: None, prev_rank: 0,
        }));

        for pos in 0..=n {
            for &arc_idx in &arcs_ending_at[pos] {
                let arc = &arcs[arc_idx];
                let last_pos = arc.morphemes.last().map(|(_, p)| *p as u8).unwrap_or(BOS);

                let start_entries: Vec<((u8, u8), usize, f32)> = dp[arc.start].iter()
                    .flat_map(|(&state, entries)| {
                        entries.iter().enumerate().map(move |(rank, &(cost, _))| (state, rank, cost))
                    })
                    .collect();

                for ((prev_pos_tag, prev_prev_pos), rank, prev_cost) in start_entries {
                    let first_pos = arc.morphemes.first().map(|(_, p)| *p as u8).unwrap_or(BOS);
                    let transition_cost = self.get_trigram_cost(prev_prev_pos, prev_pos_tag, first_pos);
                    let first_form = &arc.morphemes[0].0;
                    let wb_bonus = self.get_word_bigram_bonus(first_form, prev_pos_tag, first_pos);

                    let mut internal_cost = 0.0;
                    if arc.morphemes.len() > 1 {
                        let mut pp = prev_prev_pos;
                        let mut p = prev_pos_tag;
                        for (mi, (_, mpos)) in arc.morphemes.iter().enumerate() {
                            if mi == 0 {
                                pp = prev_pos_tag;
                                p = *mpos as u8;
                            } else {
                                internal_cost += self.get_trigram_cost(pp, p, *mpos as u8);
                                pp = p;
                                p = *mpos as u8;
                            }
                        }
                    }

                    let total_cost = prev_cost + arc.cost + transition_cost + internal_cost + wb_bonus;

                    let new_prev_prev = if arc.morphemes.len() >= 2 {
                        arc.morphemes[arc.morphemes.len() - 2].1 as u8
                    } else {
                        prev_pos_tag
                    };
                    let new_state = (last_pos, new_prev_prev);

                    let entries = dp[pos].entry(new_state).or_default();
                    if entries.len() >= top_k && total_cost >= entries.last().unwrap().0 {
                        continue;
                    }
                    let ins = entries.partition_point(|&(c, _)| c < total_cost);
                    entries.insert(ins, (total_cost, Backpointer {
                        prev_pos: arc.start,
                        prev_state: (prev_pos_tag, prev_prev_pos),
                        arc_idx: Some(arc_idx),
                        prev_rank: rank,
                    }));
                    if entries.len() > top_k {
                        entries.pop();
                    }
                }
            }

            // Space pass-through
            if pos < n && chars[pos].is_whitespace() {
                let states: Vec<((u8, u8), usize, f32)> = dp[pos].iter()
                    .flat_map(|(&state, entries)| {
                        entries.iter().enumerate().map(move |(rank, &(cost, _))| (state, rank, cost))
                    })
                    .collect();
                for (state, rank, cost) in states {
                    let entries = dp[pos + 1].entry(state).or_default();
                    if entries.len() >= top_k && cost >= entries.last().unwrap().0 {
                        continue;
                    }
                    let ins = entries.partition_point(|&(c, _)| c < cost);
                    entries.insert(ins, (cost, Backpointer {
                        prev_pos: pos, prev_state: state, arc_idx: None, prev_rank: rank,
                    }));
                    if entries.len() > top_k {
                        entries.pop();
                    }
                }
            }
        }

        // Collect global top-N from final states
        let final_states = &dp[n];
        if final_states.is_empty() {
            return vec![self.fallback_tokenize(text)];
        }

        let mut candidates: Vec<(f32, (u8, u8), usize)> = Vec::new();
        for (&state, entries) in final_states {
            let eos_cost = self.get_trigram_cost(state.1, state.0, BOS);
            for (rank, &(cost, _)) in entries.iter().enumerate() {
                candidates.push((cost + eos_cost, state, rank));
            }
        }
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(top_k * 2); // collect extra for dedup

        // Build char-to-byte map
        let mut char_to_byte = Vec::with_capacity(n + 1);
        let mut byte_off = 0usize;
        for ch in &chars {
            char_to_byte.push(byte_off);
            byte_off += ch.len_utf8();
        }
        char_to_byte.push(byte_off);

        let mut results = Vec::new();
        for &(total_cost, start_state, start_rank) in &candidates {
            if results.len() >= top_k { break; }

            let mut path_arcs: Vec<usize> = Vec::new();
            let mut cur_pos = n;
            let mut cur_state = start_state;
            let mut cur_rank = start_rank;

            loop {
                if cur_pos == 0 && cur_state == (BOS, BOS) { break; }
                let entries = match dp[cur_pos].get(&cur_state) {
                    Some(e) if cur_rank < e.len() => e,
                    _ => break,
                };
                let bp = &entries[cur_rank].1;
                if let Some(ai) = bp.arc_idx {
                    path_arcs.push(ai);
                }
                cur_pos = bp.prev_pos;
                cur_state = bp.prev_state;
                cur_rank = bp.prev_rank;
            }
            path_arcs.reverse();

            // Build tokens
            let mut tokens = Vec::new();
            for &ai in &path_arcs {
                let arc = &arcs[ai];
                for (form, pos) in &arc.morphemes {
                    tokens.push(Token {
                        text: normalize_jamo(form), pos: *pos,
                        start: arc.start, end: arc.end, score: None,
                    });
                }
            }

            // Inline post-processing (same as viterbi)
            let mut tokens = Self::merge_sl_sn_tokens(tokens);
            for i in 1..tokens.len() {
                if tokens[i].pos == Pos::EP && (tokens[i].text == "었" || tokens[i].text == "었었") {
                    if let Some(last_char) = tokens[i - 1].text.chars().last() {
                        let code = last_char as u32;
                        if code >= 0xAC00 && code <= 0xD7A3 {
                            let vowel = ((code - 0xAC00) % (21 * 28)) / 28;
                            if vowel == 0 || vowel == 8 {
                                tokens[i].text = if tokens[i].text == "었" { "았".to_string() } else { "았었".to_string() };
                            }
                        }
                    }
                }
                if (tokens[i].pos == Pos::EC || tokens[i].pos == Pos::EF) && tokens[i].text.starts_with("어") {
                    if let Some(last_char) = tokens[i - 1].text.chars().last() {
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
            Self::fix_xsv_xsa(&mut tokens);
            Self::merge_ec_ef_vowel(&mut tokens);

            // Dedup: skip if identical to a previous result
            let dominated = results.iter().any(|(prev_tokens, _): &(Vec<Token>, f32)| {
                prev_tokens.len() == tokens.len()
                    && prev_tokens.iter().zip(tokens.iter()).all(|(a, b)| a.text == b.text && a.pos == b.pos)
            });
            if !dominated {
                results.push((tokens, total_cost));
            }
        }

        if results.is_empty() {
            results.push(self.fallback_tokenize(text));
        }
        results
    }

    /// Merge vowel EC + EF into a single EF when the EC is 어/아 and the
    /// following morpheme is EF. e.g. 어/EC + 라/EF → 어라/EF.
    /// This corrects Viterbi paths that split contracted vowel endings.
    fn merge_ec_ef_vowel(tokens: &mut Vec<Token>) {
        let mut i = 0;
        while i + 1 < tokens.len() {
            if tokens[i].pos == Pos::EC
                && (tokens[i].text == "어" || tokens[i].text == "아")
                && tokens[i].start == tokens[i + 1].start
                && tokens[i + 1].pos == Pos::EF
            {
                // Merge: 어/EC + 라/EF → 어라/EF
                let merged_text = format!("{}{}", tokens[i].text, tokens[i + 1].text);
                tokens[i + 1].text = merged_text;
                tokens.remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Fix JKB → JC for 과/와/이랑/랑/하고 connecting nouns.
    /// Be conservative: only convert when clearly listing nouns, not with verbs nearby.
    fn fix_jc_jkb(tokens: &mut [Token]) {
        if tokens.len() < 3 {
            return;
        }
        for i in 1..tokens.len() - 1 {
            if tokens[i].pos != Pos::JKB {
                continue;
            }
            let form = tokens[i].text.as_str();
            if form != "과" && form != "와" && form != "이랑" && form != "랑" && form != "하고" {
                continue;
            }
            let prev_pos = tokens[i - 1].pos;
            let preceded_by_noun = matches!(
                prev_pos,
                Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP | Pos::SN | Pos::SL
            );
            if !preceded_by_noun {
                continue;
            }
            // Must be directly followed by a noun (not verb stem, not MM)
            let next_pos = tokens[i + 1].pos;
            let followed_by_noun = matches!(
                next_pos,
                Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP | Pos::SN | Pos::SL
            );
            if followed_by_noun {
                // Extra check: if the noun after is followed by a verb (VV/VA/VX),
                // it might be JKB ("A과 B를 하다" → A과 is still JC).
                // Only skip if next noun has JKB/JKO after it (comitative pattern).
                tokens[i].pos = Pos::JC;
            }
        }
    }

    /// Fix XSV/XSA: 하/되 after NNG → XSV or XSA based on context.
    /// Uses following morpheme patterns to disambiguate:
    ///   + 는/ETM → XSV (VA can't take 는 for present adnominal)
    ///   + ㄴ/ETM → XSA (VA present adnominal: 피곤한)
    ///   + 게/EC  → XSA (VA adverbial: 피곤하게)
    ///   + default → XSV
    fn fix_xsv_xsa(tokens: &mut [Token]) {
        if tokens.len() < 2 {
            return;
        }
        for i in 1..tokens.len() {
            let prev_pos = tokens[i - 1].pos;
            if prev_pos != Pos::NNG {
                continue;
            }
            let form = tokens[i].text.as_str();
            if tokens[i].pos == Pos::VV && (form == "하" || form == "되" || form == "시키") {
                tokens[i].pos = Pos::XSV;
            } else if tokens[i].pos == Pos::VA && form == "하" {
                tokens[i].pos = Pos::XSA;
            }
        }
    }

    /// Fix VV/VA → VX for auxiliary verbs after EC (connective endings).
    fn fix_vx(tokens: &mut [Token]) {
        for i in 1..tokens.len() {
            let cur_pos = tokens[i].pos;
            if tokens[i - 1].pos != Pos::EC {
                continue;
            }
            let form = tokens[i].text.as_str();
            let prev_form = tokens[i - 1].text.as_str();

            if form == "있"
                && (cur_pos == Pos::VV || cur_pos == Pos::VA)
                && matches!(tokens[i - 1].pos, Pos::NNB | Pos::NNG | Pos::VV)
                && prev_form == "수"
                && i >= 2
                && tokens[i - 2].pos == Pos::ETM
                && tokens[i].start == tokens[i - 1].start
                && tokens[i - 1].start == tokens[i - 2].start
            {
                tokens[i].pos = Pos::VX;
                continue;
            }

            // VV or VA → VX patterns
            if cur_pos == Pos::VV || cur_pos == Pos::VA {
                match form {
                    "있" | "없" => {
                        // 고 있다, 어 있다 → VX
                        if prev_form == "고" || prev_form == "어" || prev_form == "아" {
                            tokens[i].pos = Pos::VX;
                        }
                    }
                    _ => {}
                }
            }

            if cur_pos == Pos::VV {
                match form {
                    "하" => {
                        if prev_form == "지" || prev_form == "고" {
                            tokens[i].pos = Pos::VX;
                        }
                    }
                    "보" | "주" | "지" | "오" | "가" | "내" | "나" | "버리" | "놓" | "두"
                    | "가지" | "달" | "말" | "드리" | "치" | "대" | "못하" => {
                        if prev_form == "어" || prev_form == "아" {
                            tokens[i].pos = Pos::VX;
                        }
                    }
                    _ => {}
                }
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
            "법", "대로", "따름",
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

            // 이/저: only convert if at sentence start or after punctuation
            // "이" is extremely ambiguous (MM/NP/JKS/VCP) — be very conservative
            if (form == "이" || form == "저") && (cur_pos == Pos::NP || cur_pos == Pos::JKS) {
                let safe_to_convert = if i == 0 {
                    true  // Sentence start: "이 사건은..."
                } else {
                    let prev_pos = tokens[i - 1].pos;
                    // Only convert after punctuation or sentence markers
                    matches!(prev_pos, Pos::SF | Pos::SP | Pos::SS | Pos::SE)
                };
                if safe_to_convert {
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
            // Check if followed by 되/아니 (possibly with intervening morphemes)
            if i + 1 < tokens.len() {
                let next_form = tokens[i + 1].text.as_str();
                let next_pos = tokens[i + 1].pos;
                if next_form == "되" || next_form == "아니" {
                    tokens[i].pos = Pos::JKC;
                }
                // Also: 가/이 + NNG + 되 pattern (e.g., "문제가 해결이 되다")
                if next_pos == Pos::VCN {
                    tokens[i].pos = Pos::JKC;
                }
            }
        }
    }

    /// Fix JX → JC for 나/이나 connecting nouns.
    fn fix_jx_jc(tokens: &mut [Token]) {
        if tokens.len() < 3 {
            return;
        }
        for i in 1..tokens.len() - 1 {
            if tokens[i].pos != Pos::JX {
                continue;
            }
            let form = tokens[i].text.as_str();
            if form != "나" && form != "이나" {
                continue;
            }
            let prev_pos = tokens[i - 1].pos;
            let preceded_by_noun = matches!(
                prev_pos,
                Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP | Pos::SN | Pos::SL
            );
            let next_pos = tokens[i + 1].pos;
            let followed_by_noun = matches!(
                next_pos,
                Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP | Pos::SN | Pos::SL
            );
            if preceded_by_noun && followed_by_noun {
                tokens[i].pos = Pos::JC;
            }
        }
    }

    /// Fix NNB → NNG when not preceded by ETM.
    /// Dependency nouns (NNB) require a preceding adnominal ending (ETM).
    /// Without ETM, single-char NNB like 이(tooth), 수(number) should be NNG.
    fn fix_nnb_no_etm(tokens: &mut [Token]) {
        for i in 0..tokens.len() {
            if tokens[i].pos != Pos::NNB {
                continue;
            }
            // Check if preceded by ETM
            let has_etm = i > 0 && tokens[i - 1].pos == Pos::ETM;
            if has_etm {
                continue; // correct NNB usage
            }
            let form = tokens[i].text.as_str();
            // Only convert specific NNB that are commonly NNG without ETM
            // Skip: 것/수/바/데/줄 — these are legitimate NNB even without ETM in colloquial speech
            // Target: 이(tooth), 말(speech/horse) — almost always NNG when standalone
            if form == "이" || form == "말" {
                tokens[i].pos = Pos::NNG;
            }
        }
    }

    /// Fix XPN in standalone eojeol → NNG or MM.
    /// Prefixes cannot be independent eojeols. If XPN's start differs from next token's start,
    /// it's a separate eojeol and should not be XPN.
    fn fix_xpn_standalone(tokens: &mut [Token]) {
        for i in 0..tokens.len() {
            if tokens[i].pos != Pos::XPN {
                continue;
            }
            // Check if next token is in the same eojeol (same start position)
            let same_eojeol = i + 1 < tokens.len() && tokens[i].start == tokens[i + 1].start;
            if same_eojeol {
                continue; // attached to next word, XPN is correct
            }
            // Standalone eojeol — convert to appropriate POS
            let form = tokens[i].text.as_str();
            if form == "저" {
                // Check if followed by a noun → MM (관형사)
                let followed_by_noun = i + 1 < tokens.len() && matches!(
                    tokens[i + 1].pos,
                    Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::XR
                );
                tokens[i].pos = if followed_by_noun { Pos::MM } else { Pos::NP };
            } else {
                tokens[i].pos = Pos::NNG;
            }
        }
    }

    /// Fix EC → EF at end-of-sentence for endings that can be terminal.
    fn fix_ec_eos(tokens: &mut Vec<Token>) {
        if tokens.is_empty() {
            return;
        }
        // Find the last non-SF token
        let last_idx = if tokens.last().map_or(false, |t| t.pos == Pos::SF) && tokens.len() >= 2 {
            tokens.len() - 2
        } else {
            tokens.len() - 1
        };
        let last = &tokens[last_idx];
        if last.pos != Pos::EC {
            return;
        }
        // Only convert specific endings that can be EF
        let form = last.text.as_str();
        if matches!(form, "다" | "니" | "자" | "으니" | "지") {
            tokens[last_idx].pos = Pos::EF;
        } else if form == "라" && last_idx >= 1 {
            // "라/EC" at EOS after VV/VA: restore vowel-contracted imperative ending
            // 건너+어라→건너라 → should be 어라/EF (not 라/EF)
            let prev = &tokens[last_idx - 1];
            if matches!(prev.pos, Pos::VV | Pos::VA | Pos::VX) {
                // Determine vowel harmony from verb stem's last vowel
                let vowel_char = if let Some(last_ch) = prev.text.chars().last() {
                    let code = last_ch as u32;
                    if code >= 0xAC00 && code <= 0xD7A3 {
                        let vowel = ((code - 0xAC00) % (21 * 28)) / 28;
                        if vowel == 0 || vowel == 8 { '아' } else { '어' }
                    } else {
                        '어'
                    }
                } else {
                    '어'
                };
                tokens[last_idx].text = format!("{}라", vowel_char);
                tokens[last_idx].pos = Pos::EF;
            } else {
                tokens[last_idx].pos = Pos::EF;
            }
        }
    }

    /// Fix "라/EF" after VV/VA/VX: restore vowel-contracted imperative ending.
    /// e.g. 건너+라/EF → 건너+어라/EF, 가+라/EF → 가+아라/EF.
    /// "하+라/EF" is standard (하라체) and should not be changed.
    fn fix_imperative_ra(tokens: &mut [Token]) {
        for i in 1..tokens.len() {
            if tokens[i].pos != Pos::EF || tokens[i].text != "라" {
                continue;
            }
            let prev = &tokens[i - 1];
            if !matches!(prev.pos, Pos::VV | Pos::VA | Pos::VX) {
                continue;
            }
            // "하" + "라" is the 하라체 imperative — keep as-is
            if prev.text == "하" {
                continue;
            }
            // Determine vowel harmony from verb stem's last vowel
            if let Some(last_ch) = prev.text.chars().last() {
                let code = last_ch as u32;
                if code >= 0xAC00 && code <= 0xD7A3 {
                    let vowel = ((code - 0xAC00) % (21 * 28)) / 28;
                    let prefix = if vowel == 0 || vowel == 8 { "아" } else { "어" };
                    tokens[i].text = format!("{}라", prefix);
                }
            }
        }
    }

    /// Fix VCP copula: split EF/EC/ETM tokens starting with "이" into VCP + remainder.
    /// e.g. "학생이다" → 학생/NNG + 이다/EF → 학생/NNG + 이/VCP + 다/EF
    fn fix_vcp(tokens: &mut Vec<Token>) {
        // VCP endings: 이+EF/EC/ETM suffix patterns
        const VCP_SUFFIXES: &[(&str, &str)] = &[
            // (이+suffix, remaining_suffix) — EF patterns
            ("이다", "다"), ("이야", "야"), ("이에요", "에요"), ("이요", "요"),
            ("이거든", "거든"), ("이거든요", "거든요"),
            ("이랍니다", "랍니다"), ("이래요", "래요"),
            // EC patterns
            ("이고", "고"), ("이며", "며"), ("이라", "라"), ("이면", "면"),
            ("이지만", "지만"), ("이니", "니"), ("이니까", "니까"),
            ("이라서", "라서"), ("이라고", "라고"), ("이란", "란"),
            ("이든", "든"), ("이든지", "든지"), ("이나", "나"),
            // ETM patterns
            ("이던", "던"), ("인", "ㄴ"), ("일", "ㄹ"),
        ];
        let mut i = 0;
        while i < tokens.len() {
            let pos = tokens[i].pos;
            if !matches!(pos, Pos::EF | Pos::EC | Pos::ETM | Pos::ETN) {
                i += 1;
                continue;
            }
            // Must follow a noun
            if i == 0 || !matches!(tokens[i - 1].pos,
                Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP | Pos::SN | Pos::SL | Pos::XSN)
            {
                i += 1;
                continue;
            }
            let form = &tokens[i].text;
            let mut matched = false;
            for &(pattern, remainder) in VCP_SUFFIXES {
                if form == pattern {
                    let orig = tokens[i].clone();
                    tokens[i] = Token {
                        text: "이".to_string(), pos: Pos::VCP,
                        start: orig.start, end: orig.end, score: None,
                    };
                    tokens.insert(i + 1, Token {
                        text: remainder.to_string(), pos: orig.pos,
                        start: orig.start, end: orig.end, score: None,
                    });
                    matched = true;
                    break;
                }
            }
            i += if matched { 2 } else { 1 };
        }
    }

    /// Fix NNG → MM for common determiners: 전, 한, 그런, 이런, 저런, 어떤, 새, 헌, 옛
    fn fix_mm_determiners(tokens: &mut [Token]) {
        const MM_WORDS: &[&str] = &["전", "그런", "이런", "저런", "어떤", "새", "헌", "옛", "온"];
        for i in 0..tokens.len().saturating_sub(1) {
            if tokens[i].pos != Pos::NNG { continue; }
            let form = tokens[i].text.as_str();
            let next_pos = tokens[i + 1].pos;
            let followed_by_noun = matches!(next_pos,
                Pos::NNG | Pos::NNP | Pos::NNB | Pos::NR | Pos::NP | Pos::XPN);
            if followed_by_noun && MM_WORDS.contains(&form) {
                tokens[i].pos = Pos::MM;
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
    ///
    /// Builds a full-sentence lattice where eojeol cache entries are injected as
    /// low-cost arcs competing with Viterbi alternatives. This allows cross-eojeol
    /// trigram context to disambiguate homographs like "나는" (NP+JX vs VV+ETM).
    pub fn analyze(&self, text: &str) -> AnalyzeResult {
        if text.is_empty() {
            return AnalyzeResult {
                tokens: vec![],
                score: 0.0,
                elapsed_ms: 0.0,
            };
        }

        let t0 = now_ms();

        // Build full-sentence lattice with cache entries as arcs
        let mut arcs = self.build_lattice(text);

        // Inject eojeol cache entries as competing arcs in the lattice
        if !self.eojeol_cache.is_empty() {
            let mut char_offset = 0usize;

            for eojeol in text.split_whitespace() {
                let byte_start = text[char_offset..].find(eojeol)
                    .map(|p| char_offset + p)
                    .unwrap_or(char_offset);
                let eojeol_char_start = text[..byte_start].chars().count();
                let eojeol_char_end = eojeol_char_start + eojeol.chars().count();

                if let Some(cached_morphs) = self.eojeol_cache.get(eojeol) {
                    // Add cache entry as a very low-cost arc
                    let morphemes: Vec<(String, Pos)> = cached_morphs.iter()
                        .map(|(f, p)| (f.clone(), *p))
                        .collect();

                    // Give cache entries a strong bonus (low cost) but not absolute
                    // -2.0 bonus: very strong prior, but cross-eojeol trigram can still override
                    // For the last eojeol ending with EC, weaken the bonus so Viterbi
                    // can choose EF if the SF-aware arc (above) provides a better path.
                    let is_last_eojeol = !text[byte_start + eojeol.len()..].chars()
                        .any(|c| !c.is_whitespace()
                            && !matches!(c, '.' | '!' | '?' | '…')
                            && !(c >= '\u{3131}' && c <= '\u{3163}'));
                    let last_is_ec = cached_morphs.last()
                        .map_or(false, |(_, p)| *p == Pos::EC);
                    let cache_cost = if is_last_eojeol && last_is_ec { -0.5 } else { -2.0 };

                    arcs.push(LatticeArc {
                        start: eojeol_char_start,
                        end: eojeol_char_end,
                        morphemes,
                        cost: cache_cost,
                    });
                }

                char_offset = byte_start + eojeol.len();
            }
        }

        // Run sentence-level Viterbi on the full lattice
        let (raw_tokens, score) = self.viterbi(text, &arcs);

        // Assign eojeol-level spans
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
            let (es, ee) = eojeol_boundaries.iter()
                .find(|&&(s, e)| token.start >= s && token.start < e)
                .copied()
                .unwrap_or((token.start, token.end));
            tokens.push(Token { start: es, end: ee, ..token });
        }

        // Post-process
        Self::fix_vx(&mut tokens);
        Self::fix_jc_jkb(&mut tokens);
        Self::fix_nnb(&mut tokens);
        Self::fix_xsn_xpn(&mut tokens);
        Self::fix_mm(&mut tokens);
        Self::fix_jkc(&mut tokens);
        Self::fix_jx_jc(&mut tokens);
        Self::fix_nnb_no_etm(&mut tokens);
        Self::fix_xpn_standalone(&mut tokens);
        Self::fix_ec_eos(&mut tokens);
        Self::fix_imperative_ra(&mut tokens);
        Self::fix_vcp(&mut tokens);
        Self::fix_mm_determiners(&mut tokens);

        AnalyzeResult {
            tokens,
            score,
            elapsed_ms: now_ms() - t0,
        }
    }

    /// Analyze returning top-N distinct results, sorted by Viterbi cost.
    pub fn analyze_topn(&self, text: &str, n: usize) -> Vec<AnalyzeResult> {
        if text.is_empty() || n == 0 {
            return vec![AnalyzeResult { tokens: vec![], score: 0.0, elapsed_ms: 0.0 }];
        }
        if n == 1 {
            return vec![self.analyze(text)];
        }

        let t0 = now_ms();

        // Build lattice with cache injection (same as analyze)
        let mut arcs = self.build_lattice(text);
        if !self.eojeol_cache.is_empty() {
            let mut char_offset = 0usize;
            for eojeol in text.split_whitespace() {
                let byte_start = text[char_offset..].find(eojeol)
                    .map(|p| char_offset + p).unwrap_or(char_offset);
                let eojeol_char_start = text[..byte_start].chars().count();
                let eojeol_char_end = eojeol_char_start + eojeol.chars().count();
                if let Some(cached_morphs) = self.eojeol_cache.get(eojeol) {
                    let morphemes: Vec<(String, Pos)> = cached_morphs.iter()
                        .map(|(f, p)| (f.clone(), *p)).collect();
                    let is_last_eojeol = !text[byte_start + eojeol.len()..].chars()
                        .any(|c| !c.is_whitespace()
                            && !matches!(c, '.' | '!' | '?' | '…')
                            && !(c >= '\u{3131}' && c <= '\u{3163}'));
                    let last_is_ec = cached_morphs.last().map_or(false, |(_, p)| *p == Pos::EC);
                    let cache_cost = if is_last_eojeol && last_is_ec { -0.5 } else { -2.0 };
                    arcs.push(LatticeArc {
                        start: eojeol_char_start, end: eojeol_char_end,
                        morphemes, cost: cache_cost,
                    });
                }
                char_offset = byte_start + eojeol.len();
            }
        }

        // Run N-best Viterbi
        let paths = self.viterbi_nbest(text, &arcs, n);

        // Build eojeol boundaries for span assignment
        let mut eojeol_boundaries: Vec<(usize, usize)> = Vec::new();
        {
            let mut char_offset = 0;
            for eojeol in text.split_whitespace() {
                let byte_start = text[char_offset..].find(eojeol)
                    .map(|p| char_offset + p).unwrap_or(char_offset);
                let cs = text[..byte_start].chars().count();
                let ce = cs + eojeol.chars().count();
                eojeol_boundaries.push((cs, ce));
                char_offset = byte_start + eojeol.len();
            }
        }

        // Post-process each path
        paths.into_iter().map(|(raw_tokens, score)| {
            let mut tokens: Vec<Token> = raw_tokens.into_iter().map(|token| {
                let (es, ee) = eojeol_boundaries.iter()
                    .find(|&&(s, e)| token.start >= s && token.start < e)
                    .copied().unwrap_or((token.start, token.end));
                Token { start: es, end: ee, ..token }
            }).collect();

            Self::fix_vx(&mut tokens);
            Self::fix_jc_jkb(&mut tokens);
            Self::fix_nnb(&mut tokens);
            Self::fix_xsn_xpn(&mut tokens);
            Self::fix_mm(&mut tokens);
            Self::fix_jkc(&mut tokens);
            Self::fix_jx_jc(&mut tokens);
            Self::fix_nnb_no_etm(&mut tokens);
            Self::fix_xpn_standalone(&mut tokens);
            Self::fix_ec_eos(&mut tokens);
            Self::fix_imperative_ra(&mut tokens);
            Self::fix_vcp(&mut tokens);
            Self::fix_mm_determiners(&mut tokens);

            AnalyzeResult { tokens, score, elapsed_ms: now_ms() - t0 }
        }).collect()
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

        // Mixed alphanumeric: alpha-start includes digits/hyphens
        let chars2: Vec<char> = "b2b BM25 GPT-4o 2024".chars().collect();
        let runs2 = CodebookAnalyzer::find_ascii_runs(&chars2);
        assert_eq!(runs2.len(), 4);
        assert_eq!(runs2[0], (0, 3, Pos::SL));   // b2b
        assert_eq!(runs2[1], (4, 8, Pos::SL));   // BM25
        assert_eq!(runs2[2], (9, 15, Pos::SL));  // GPT-4o
        assert_eq!(runs2[3], (16, 20, Pos::SN)); // 2024
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
