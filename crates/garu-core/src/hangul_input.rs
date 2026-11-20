use std::borrow::Cow;

use crate::types::Token;

/// Modern Hangul syllables for inference, with boundaries in the original input.
pub(crate) struct HangulInput<'a> {
    pub text: Cow<'a, str>,
    source_offsets: Vec<usize>,
}

impl<'a> HangulInput<'a> {
    pub fn new(source: &'a str) -> Self {
        // Every modern combining jamo starts with this UTF-8 byte. Ordinary
        // precomposed Korean and ASCII need no decoding, allocation or offset map.
        if !source.as_bytes().contains(&0xe1) {
            return Self {
                text: Cow::Borrowed(source),
                source_offsets: Vec::new(),
            };
        }

        let mut chars = source.chars().enumerate().peekable();
        let mut text = String::with_capacity(source.len());
        let mut source_offsets = vec![0];
        while let Some((mut last, mut ch)) = chars.next() {
            while let Some(&(index, next)) = chars.peek() {
                let Some(composed) = compose_pair(ch, next) else {
                    break;
                };
                ch = composed;
                last = index;
                chars.next();
            }
            text.push(ch);
            source_offsets.push(last + 1);
        }
        Self {
            text: Cow::Owned(text),
            source_offsets,
        }
    }

    pub fn restore_offsets(&self, tokens: &mut [Token]) {
        if self.source_offsets.is_empty() {
            return;
        }
        for token in tokens {
            token.start = self.source_offsets[token.start];
            token.end = self.source_offsets[token.end];
        }
    }
}

/// Unicode Hangul composition (L + V → LV, LV + T → LVT).
fn compose_pair(first: char, next: char) -> Option<char> {
    const S_BASE: u32 = 0xac00;
    const L_BASE: u32 = 0x1100;
    const V_BASE: u32 = 0x1161;
    const T_BASE: u32 = 0x11a7;
    const V_COUNT: u32 = 21;
    const T_COUNT: u32 = 28;

    let a = first as u32;
    let b = next as u32;
    let syllable = match (first, next) {
        ('\u{1100}'..='\u{1112}', '\u{1161}'..='\u{1175}') => {
            S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT
        }
        ('\u{ac00}'..='\u{d7a3}', '\u{11a8}'..='\u{11c2}') if (a - S_BASE) % T_COUNT == 0 => {
            a + (b - T_BASE)
        }
        _ => return None,
    };
    char::from_u32(syllable)
}
