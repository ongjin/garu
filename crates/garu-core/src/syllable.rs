//! Syllable-level encoding for GMDL v2 models.
//!
//! Each character maps to exactly one token ID.
//! Hangul syllables (가~힣) get direct IDs; ASCII letters, digits,
//! and common symbols each have their own ID.

const PAD: u16 = 0;
const UNK: u16 = 1;
const SPACE: u16 = 2;
const HANGUL_OFFSET: u16 = 3;
const HANGUL_BASE: u32 = 0xAC00;
const HANGUL_END: u32 = 0xD7A3;
const HANGUL_COUNT: u16 = 11172;

const UPPER_OFFSET: u16 = HANGUL_OFFSET + HANGUL_COUNT; // 11175
const LOWER_OFFSET: u16 = UPPER_OFFSET + 26;            // 11201
const DIGIT_OFFSET: u16 = LOWER_OFFSET + 26;            // 11227

/// Symbol characters — must match Python training code exactly.
/// Note: single quote appears at index 14 and 15 (duplicated in Python list).
const SYMBOL_CHARS: &[char] = &[
    '.', ',', '!', '?', ';', ':', '-', '_', '(', ')',
    '[', ']', '{', '}', '\'', '\'', '"', '@', '#', '$',
    '%', '&', '*', '+', '/', '=', '<', '>', '~', '`',
    '\\', '|', '^',
];

const SYMBOL_OFFSET: u16 = DIGIT_OFFSET + 10; // 11237

/// Encode a single character to its vocab ID.
pub fn char_to_id(ch: char) -> u16 {
    let code = ch as u32;
    if ch == ' ' {
        return SPACE;
    }
    if code >= HANGUL_BASE && code <= HANGUL_END {
        return HANGUL_OFFSET + (code - HANGUL_BASE) as u16;
    }
    if ch.is_ascii_uppercase() {
        return UPPER_OFFSET + (ch as u16 - b'A' as u16);
    }
    if ch.is_ascii_lowercase() {
        return LOWER_OFFSET + (ch as u16 - b'a' as u16);
    }
    if ch.is_ascii_digit() {
        return DIGIT_OFFSET + (ch as u16 - b'0' as u16);
    }
    // Symbol lookup
    for (i, &sym) in SYMBOL_CHARS.iter().enumerate() {
        if ch == sym {
            return SYMBOL_OFFSET + i as u16;
        }
    }
    UNK
}

/// Total vocabulary size — must match training config.
pub const VOCAB_SIZE: usize = SYMBOL_OFFSET as usize + SYMBOL_CHARS.len();

/// Encode a text string into a sequence of vocab IDs.
/// One ID per character (unlike jamo which produces 2-3 per Hangul character).
pub fn encode(text: &str) -> Vec<u16> {
    text.chars().map(char_to_id).collect()
}
