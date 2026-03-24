//! Korean Hangul syllable decomposition into Jamo and vocab ID encoding.

// --- Special token IDs ---
pub const PAD: u16 = 0;
pub const UNK: u16 = 1;
pub const SPACE: u16 = 2;
pub const NUM: u16 = 3;
pub const LATIN: u16 = 4;
pub const PUNCT: u16 = 5;
pub const JAMO_OFFSET: u16 = 6;

/// Jamo compatibility block: 0x3131 ..= 0x3163 (51 characters).
pub const JAMO_COMPAT_START: u32 = 0x3131;
pub const JAMO_COMPAT_END: u32 = 0x3163;
pub const JAMO_COUNT: u16 = 51;
pub const VOCAB_SIZE: u16 = JAMO_OFFSET + JAMO_COUNT; // 57

// Hangul syllable block
const SBase: u32 = 0xAC00;
const LCount: u32 = 19;
const VCount: u32 = 21;
const TCount: u32 = 28;
const NCount: u32 = VCount * TCount; // 588

/// Leading consonants in compatibility Jamo order.
const LEADS: [char; 19] = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
];

/// Vowels in compatibility Jamo order.
const VOWELS: [char; 21] = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
    'ㅣ',
];

/// Trailing consonants (index 0 = no tail).
const TAILS: [Option<char>; 28] = [
    None,
    Some('ㄱ'), Some('ㄲ'), Some('ㄳ'), Some('ㄴ'), Some('ㄵ'), Some('ㄶ'),
    Some('ㄷ'), Some('ㄹ'), Some('ㄺ'), Some('ㄻ'), Some('ㄼ'), Some('ㄽ'),
    Some('ㄾ'), Some('ㄿ'), Some('ㅀ'), Some('ㅁ'), Some('ㅂ'), Some('ㅄ'),
    Some('ㅅ'), Some('ㅆ'), Some('ㅇ'), Some('ㅈ'), Some('ㅊ'), Some('ㅋ'),
    Some('ㅌ'), Some('ㅍ'), Some('ㅎ'),
];

/// Decompose a Hangul syllable (U+AC00..U+D7A3) into (lead, vowel, Option<tail>)
/// using compatibility Jamo characters (U+3131..U+3163).
///
/// Returns `None` if the character is not a precomposed Hangul syllable.
pub fn decompose(ch: char) -> Option<(char, char, Option<char>)> {
    let code = ch as u32;
    if code < SBase || code > 0xD7A3 {
        return None;
    }
    let offset = code - SBase;
    let l_idx = (offset / NCount) as usize;
    let v_idx = ((offset % NCount) / TCount) as usize;
    let t_idx = (offset % TCount) as usize;

    Some((LEADS[l_idx], VOWELS[v_idx], TAILS[t_idx]))
}

/// Map a compatibility Jamo character (U+3131..U+3163) to its vocab ID.
fn jamo_to_id(ch: char) -> u16 {
    let code = ch as u32;
    if code >= JAMO_COMPAT_START && code <= JAMO_COMPAT_END {
        JAMO_OFFSET + (code - JAMO_COMPAT_START) as u16
    } else {
        UNK
    }
}

/// Encode a text string into a sequence of vocab IDs suitable for BiLSTM input.
///
/// - Hangul syllables are decomposed into Jamo and each Jamo gets its own ID.
/// - Standalone compatibility Jamo characters are mapped directly.
/// - Spaces map to `SPACE`, ASCII digits to `NUM`, ASCII letters to `LATIN`,
///   ASCII punctuation to `PUNCT`, and everything else to `UNK`.
pub fn encode(text: &str) -> Vec<u16> {
    let mut ids = Vec::new();
    for ch in text.chars() {
        if let Some((l, v, t)) = decompose(ch) {
            ids.push(jamo_to_id(l));
            ids.push(jamo_to_id(v));
            if let Some(tail) = t {
                ids.push(jamo_to_id(tail));
            }
        } else {
            let code = ch as u32;
            if code >= JAMO_COMPAT_START && code <= JAMO_COMPAT_END {
                ids.push(jamo_to_id(ch));
            } else if ch == ' ' {
                ids.push(SPACE);
            } else if ch.is_ascii_digit() {
                ids.push(NUM);
            } else if ch.is_ascii_alphabetic() {
                ids.push(LATIN);
            } else if ch.is_ascii_punctuation() {
                ids.push(PUNCT);
            } else {
                ids.push(UNK);
            }
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_ga() {
        // 가 = ㄱ + ㅏ, no tail
        let result = decompose('가');
        assert_eq!(result, Some(('ㄱ', 'ㅏ', None)));
    }

    #[test]
    fn test_decompose_han() {
        // 한 = ㅎ + ㅏ + ㄴ
        let result = decompose('한');
        assert_eq!(result, Some(('ㅎ', 'ㅏ', Some('ㄴ'))));
    }

    #[test]
    fn test_decompose_non_hangul() {
        assert_eq!(decompose('A'), None);
        assert_eq!(decompose('1'), None);
        assert_eq!(decompose('!'), None);
    }

    #[test]
    fn test_encode_korean() {
        // 가 → ㄱ ㅏ (2 IDs)
        let ids = encode("가");
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], jamo_to_id('ㄱ'));
        assert_eq!(ids[1], jamo_to_id('ㅏ'));

        // 한 → ㅎ ㅏ ㄴ (3 IDs)
        let ids = encode("한");
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], jamo_to_id('ㅎ'));
        assert_eq!(ids[1], jamo_to_id('ㅏ'));
        assert_eq!(ids[2], jamo_to_id('ㄴ'));
    }

    #[test]
    fn test_encode_mixed() {
        // "가 A1!" → ㄱ ㅏ SPACE LATIN NUM PUNCT
        let ids = encode("가 A1!");
        assert_eq!(ids.len(), 6);
        assert_eq!(ids[0], jamo_to_id('ㄱ'));
        assert_eq!(ids[1], jamo_to_id('ㅏ'));
        assert_eq!(ids[2], SPACE);
        assert_eq!(ids[3], LATIN);
        assert_eq!(ids[4], NUM);
        assert_eq!(ids[5], PUNCT);
    }

    #[test]
    fn test_vocab_size() {
        assert_eq!(VOCAB_SIZE, 57);
        assert_eq!(JAMO_OFFSET + JAMO_COUNT, VOCAB_SIZE);
    }

    #[test]
    fn test_jamo_id_range() {
        // First Jamo: ㄱ (0x3131) → ID 6
        assert_eq!(jamo_to_id('ㄱ'), 6);
        // Last Jamo: ㅣ (0x3163) → ID 56
        assert_eq!(jamo_to_id('ㅣ'), 56);
    }
}
