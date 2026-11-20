//! Q8 argument/context model for the Korean verb stems 사 and 살.
use crate::model::Analyzer;
use crate::types::{AnalyzeResult, Pos, Token};

const N_BEST: usize = 10;
const WINDOW: usize = 4;
const LOG_ODDS: f64 = std::f64::consts::LN_2 * 2.0; // log(0.8/0.2)

pub(crate) struct Homograph {
    bias: f32,
    scale: f32,
    hashes: Vec<u32>,
    weights: Vec<i8>,
}

#[derive(Clone, Copy)]
struct Word<'a> {
    surface: &'a str,
    start: usize,
    end: usize,
    byte: usize,
}

impl Word<'_> {
    fn contains(&self, token: &Token) -> bool {
        self.start <= token.start && token.start < self.end && token.end <= self.end
    }
}

fn words(text: &str) -> Vec<Word<'_>> {
    let (mut byte_end, mut cp_end) = (0, 0);
    text.split_whitespace()
        .map(|surface| {
            let byte = surface.as_ptr() as usize - text.as_ptr() as usize;
            let start = cp_end + text[byte_end..byte].chars().count();
            let end = start + surface.chars().count();
            byte_end = byte + surface.len();
            cp_end = end;
            Word {
                surface,
                start,
                end,
                byte,
            }
        })
        .collect()
}

fn stripped(s: &str) -> &str {
    s.trim_matches(|c| !('가'..='힣').contains(&c))
}

fn eligible(s: &str) -> bool {
    matches!(
        stripped(s),
        "사는"
            | "살"
            | "산"
            | "산다"
            | "산다고"
            | "사는데"
            | "산다는"
            | "사니까"
            | "사시는"
            | "사세요"
            | "살게요"
    )
}

fn punct(pos: Pos) -> bool {
    matches!(
        pos,
        Pos::SS | Pos::SF | Pos::SP | Pos::SW | Pos::SE | Pos::SO
    )
}

fn feature(parts: &[&str]) -> u32 {
    parts
        .iter()
        .flat_map(|p| p.bytes())
        .fold(2166136261, |hash, b| {
            (hash ^ u32::from(b)).wrapping_mul(16777619)
        })
}

fn add_context(
    hashes: &mut Vec<u32>,
    neighbor: Word<'_>,
    side: &str,
    near: bool,
    tokens: &[Token],
) {
    let mut last = None;
    let mut noun = None;
    for t in tokens.iter().filter(|t| neighbor.contains(t)) {
        let tag = t.pos.as_str();
        let form = t.text.as_str();
        last = Some(tag);
        let mut add = |parts: &[&str]| hashes.push(feature(parts));
        if matches!(
            t.pos,
            Pos::JKO
                | Pos::JKB
                | Pos::JKS
                | Pos::JKG
                | Pos::JKC
                | Pos::JKQ
                | Pos::JX
                | Pos::JC
                | Pos::EC
                | Pos::EF
                | Pos::ETM
                | Pos::ETN
                | Pos::EP
        ) {
            add(&[side, "Gr:", form, "/", tag]);
            if near {
                add(&[side, "1Gr:", form, "/", tag]);
            }
        } else if !tag.starts_with(['S', 'J', 'E']) {
            add(&["lex:", form, "/", tag]);
            add(&[side, "Lex:", form, "/", tag]);
            if near {
                add(&[side, "1Lex:", form, "/", tag]);
            }
            if tag.starts_with('N') {
                noun = Some(form);
            }
        }
    }
    if !near {
        return;
    }
    if let Some(tag) = last {
        hashes.push(feature(&[side, "1Tag:", tag]));
        let clean = stripped(neighbor.surface);
        if !clean.is_empty() {
            let from = clean.char_indices().nth_back(1).map_or(0, |(i, _)| i);
            hashes.push(feature(&[side, "1Sfx:", &clean[from..]]));
        }
        if let Some(noun) = noun {
            if matches!(
                tag,
                "JKO"
                    | "JKB"
                    | "JKS"
                    | "JKG"
                    | "JKC"
                    | "JKQ"
                    | "JX"
                    | "JC"
                    | "EC"
                    | "EF"
                    | "ETM"
                    | "ETN"
                    | "EP"
            ) {
                hashes.push(feature(&[side, "1Arg:", noun, "/", tag]));
            }
        }
    }
}

fn pairs<'a>(toks: &'a [Token], word: Word<'_>) -> Vec<(&'a str, Pos)> {
    let mut result = Vec::new();
    for t in toks.iter().filter(|t| word.contains(t)) {
        if t.pos == Pos::EF && matches!(t.text.as_str(), "ㄹ게요" | "을게요") {
            result.push((t.text.strip_suffix('요').unwrap(), Pos::EF));
            result.push(("요", Pos::JX));
        } else {
            result.push((t.text.as_str(), t.pos));
        }
    }
    result
}

fn available(candidates: &[AnalyzeResult], word: Word<'_>, expected: &[(&str, Pos)]) -> bool {
    candidates
        .iter()
        .any(|c| pairs(&c.tokens, word) == expected)
}

impl Homograph {
    pub(crate) fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 12 {
            return Err("Truncated homograph weights".into());
        }
        let bias = f32::from_le_bytes(data[0..4].try_into().unwrap());
        let scale = f32::from_le_bytes(data[4..8].try_into().unwrap());
        let n = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        if (data.len() - 12) % 5 != 0
            || (data.len() - 12) / 5 != n
            || !bias.is_finite()
            || !scale.is_finite()
            || scale <= 0.
        {
            return Err("Invalid homograph weights".into());
        }
        let mut hashes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for row in data[12..].chunks_exact(5) {
            let hash = u32::from_le_bytes(row[..4].try_into().unwrap());
            if hashes.last().is_some_and(|h| *h >= hash) {
                return Err("Unsorted homograph weights".into());
            }
            hashes.push(hash);
            weights.push(row[4] as i8);
        }
        Ok(Self {
            bias,
            scale,
            hashes,
            weights,
        })
    }

    fn choice(&self, ws: &[Word<'_>], at: usize, tokens: &[Token]) -> Option<&'static str> {
        let mut keys = Vec::new();
        if at == 0 {
            keys.push(feature(&["L:BOS/EOS"]));
        }
        if at + 1 == ws.len() {
            keys.push(feature(&["R:BOS/EOS"]));
        }
        for (i, w) in ws[..at].iter().rev().take(WINDOW).enumerate() {
            add_context(&mut keys, *w, "L", i == 0, tokens);
        }
        for (i, w) in ws[at + 1..].iter().take(WINDOW).enumerate() {
            add_context(&mut keys, *w, "R", i == 0, tokens);
        }
        keys.sort_unstable();
        keys.dedup();
        let sum: i32 = keys
            .iter()
            .filter_map(|h| self.hashes.binary_search(h).ok())
            .map(|i| i32::from(self.weights[i]))
            .sum();
        let z = f64::from(self.bias) + f64::from(self.scale) * f64::from(sum);
        if z >= LOG_ODDS {
            Some("사")
        } else if z <= -LOG_ODDS {
            Some("살")
        } else {
            None
        }
    }

    pub(crate) fn apply(&self, analyzer: &Analyzer, text: &str, tokens: &mut [Token]) {
        if !tokens
            .iter()
            .any(|t| t.pos == Pos::VV && matches!(t.text.as_str(), "살" | "사"))
        {
            return;
        }
        if !text.split_whitespace().any(eligible) {
            return;
        }
        let dict = analyzer.codebook();
        if dict.user_word_pos("사").is_some() || dict.user_word_pos("살").is_some() {
            return;
        }
        let ws = words(text);
        let mut globally = None;
        let mut edits = Vec::new();
        for (i, &w) in ws.iter().enumerate() {
            if !eligible(w.surface)
                || dict.user_word_pos(w.surface).is_some()
                || dict.user_word_pos(stripped(w.surface)).is_some()
            {
                continue;
            }
            let Some((j, verb)) = tokens
                .iter()
                .enumerate()
                .find(|(_, t)| w.contains(t) && !punct(t.pos))
            else {
                continue;
            };
            if verb.pos != Pos::VV || !matches!(verb.text.as_str(), "사" | "살") {
                continue;
            }
            let Some(selected) = self.choice(&ws, i, tokens) else {
                continue;
            };
            if verb.text == selected {
                continue;
            }
            let mut wanted = pairs(tokens, w);
            let root = wanted.iter().position(|(_, p)| !punct(*p)).unwrap();
            wanted[root].0 = selected;
            let pool =
                globally.get_or_insert_with(|| analyzer.analyze_topn_for_rerank(text, N_BEST));
            if !available(pool, w, &wanted) {
                let left = ws[i.saturating_sub(1)];
                let right = ws[(i + 1).min(ws.len() - 1)];
                let end = right.byte + right.surface.len();
                if left.byte == 0 && end == text.len() {
                    continue;
                }
                let nearby = analyzer.analyze_topn_for_rerank(&text[left.byte..end], N_BEST);
                let local = Word {
                    start: w.start - left.start,
                    end: w.end - left.start,
                    ..w
                };
                if !available(&nearby, local, &wanted) {
                    continue;
                }
            }
            edits.push((j, selected));
        }
        for (j, selected) in edits {
            tokens[j].text = selected.to_owned();
        }
    }
}
