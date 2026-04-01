//! Analyzer — thin wrapper around CodebookAnalyzer with optional CNN reranker.

use crate::codebook::CodebookAnalyzer;
use crate::cnn::Cnn2;
use crate::types::{AnalyzeResult, Pos, Token};

pub struct Analyzer {
    codebook: CodebookAnalyzer,
    cnn: Option<Cnn2>,
}

impl Analyzer {
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let codebook = CodebookAnalyzer::from_bytes(data)?;
        Ok(Self { codebook, cnn: None })
    }

    /// Load CNN reranker from separate binary data.
    pub fn load_cnn(&mut self, data: &[u8]) -> Result<(), String> {
        self.cnn = Some(Cnn2::from_bytes(data)?);
        Ok(())
    }

    pub fn analyze(&self, text: &str) -> AnalyzeResult {
        let mut result = self.codebook.analyze(text);

        // Apply CNN reranking if available
        if let Some(cnn) = &self.cnn {
            Self::apply_cnn_rerank(cnn, text, &mut result.tokens);
        }

        result
    }

    /// CNN reranking with confidence-based POS override and segmentation correction.
    fn apply_cnn_rerank(cnn: &Cnn2, text: &str, tokens: &mut Vec<Token>) {
        let cnn_result = cnn.predict(text);
        if cnn_result.is_empty() {
            return;
        }

        const CONFIDENCE_THRESHOLD: f32 = 0.9;

        // Build CNN morpheme sequence from BIO tags with per-morpheme confidence
        let mut cnn_morphs: Vec<(String, Pos, f32)> = Vec::new(); // (form, pos, min_confidence)
        let mut cur_form = String::new();
        let mut cur_pos: Option<Pos> = None;
        let mut cur_conf: f32 = 1.0;

        for &(ch, label, conf) in &cnn_result {
            if ch == ' ' {
                if !cur_form.is_empty() {
                    if let Some(p) = cur_pos {
                        cnn_morphs.push((cur_form.clone(), p, cur_conf));
                    }
                    cur_form.clear();
                    cur_pos = None;
                    cur_conf = 1.0;
                }
                continue;
            }
            if label.starts_with("B-") {
                if !cur_form.is_empty() {
                    if let Some(p) = cur_pos {
                        cnn_morphs.push((cur_form.clone(), p, cur_conf));
                    }
                }
                cur_form = ch.to_string();
                cur_pos = Self::parse_pos(&label[2..]);
                cur_conf = conf;
            } else if label.starts_with("I-") {
                cur_form.push(ch);
                cur_conf = cur_conf.min(conf);
            } else {
                if !cur_form.is_empty() {
                    if let Some(p) = cur_pos {
                        cnn_morphs.push((cur_form.clone(), p, cur_conf));
                    }
                }
                cur_form = ch.to_string();
                cur_pos = Some(Pos::SW);
                cur_conf = conf;
            }
        }
        if !cur_form.is_empty() {
            if let Some(p) = cur_pos {
                cnn_morphs.push((cur_form, p, cur_conf));
            }
        }

        // Split tokens into eojeol groups (by start/end offsets)
        let mut eojeol_groups: Vec<(usize, usize)> = Vec::new(); // (token_start, token_end)
        if !tokens.is_empty() {
            let mut gs = 0;
            for i in 1..tokens.len() {
                if tokens[i].start != tokens[gs].start {
                    eojeol_groups.push((gs, i));
                    gs = i;
                }
            }
            eojeol_groups.push((gs, tokens.len()));
        }

        // Split CNN morphs into eojeol groups (by space boundaries)
        let mut cnn_eojeol_groups: Vec<(usize, usize)> = Vec::new();
        {
            let mut gs = 0;
            let mut char_pos = 0;
            for (ci, (form, _, _)) in cnn_morphs.iter().enumerate() {
                let new_char_pos = char_pos + form.chars().count();
                // Check if there's a space after this morpheme
                if ci + 1 < cnn_morphs.len() {
                    // Space detection: check if next morpheme starts after a gap
                    let next_start = new_char_pos;
                    let text_chars: Vec<char> = text.chars().collect();
                    if next_start < text_chars.len() && text_chars[next_start] == ' ' {
                        cnn_eojeol_groups.push((gs, ci + 1));
                        gs = ci + 1;
                    }
                }
                char_pos = new_char_pos;
            }
            if gs < cnn_morphs.len() {
                cnn_eojeol_groups.push((gs, cnn_morphs.len()));
            }
        }

        // Process each eojeol: compare Viterbi vs CNN
        let replacements: Vec<(usize, usize, Vec<Token>)> = Vec::new();

        for (eg_idx, &(vit_start, vit_end)) in eojeol_groups.iter().enumerate() {
            if eg_idx >= cnn_eojeol_groups.len() {
                break;
            }
            let (cnn_start, cnn_end) = cnn_eojeol_groups[eg_idx];

            let vit_slice = &tokens[vit_start..vit_end];
            let cnn_slice = &cnn_morphs[cnn_start..cnn_end];

            // Check if same segmentation (same morpheme count and forms match)
            let same_seg = vit_slice.len() == cnn_slice.len()
                && vit_slice.iter().zip(cnn_slice.iter()).all(|(v, c)| v.text == c.0);

            if same_seg {
                // Same segmentation → confidence-based POS override
                for (vi, ci) in (vit_start..vit_end).zip(cnn_start..cnn_end) {
                    let (_, cnn_pos, conf) = &cnn_morphs[ci];
                    if *cnn_pos == tokens[vi].pos || *conf < CONFIDENCE_THRESHOLD {
                        continue;
                    }
                    if Self::is_ambiguous_pair(tokens[vi].pos, *cnn_pos) {
                        tokens[vi].pos = *cnn_pos;
                    }
                }
            } else {
                // Different segmentation → disabled for now (too aggressive)
                // TODO: enable with higher threshold and validation
            }
        }

        // Apply segmentation replacements in reverse order
        for (start, end, new_tokens) in replacements.into_iter().rev() {
            tokens.splice(start..end, new_tokens);
        }
    }

    fn is_ambiguous_pair(a: Pos, b: Pos) -> bool {
        matches!(
            (a, b),
            (Pos::NP, Pos::VV) | (Pos::VV, Pos::NP) |
            (Pos::XSV, Pos::XSA) | (Pos::XSA, Pos::XSV) |
            (Pos::VV, Pos::VA) | (Pos::VA, Pos::VV) |
            (Pos::VV, Pos::VX) | (Pos::VX, Pos::VV) |
            (Pos::VA, Pos::VX) | (Pos::VX, Pos::VA) |
            (Pos::XSV, Pos::VV) | (Pos::VV, Pos::XSV) |
            (Pos::MM, Pos::ETM) | (Pos::ETM, Pos::MM) |
            (Pos::MM, Pos::NR) | (Pos::NR, Pos::MM) |
            (Pos::NNG, Pos::NNP) | (Pos::NNP, Pos::NNG) |
            (Pos::NNG, Pos::NNB) | (Pos::NNB, Pos::NNG) |
            (Pos::NNG, Pos::MAG) | (Pos::MAG, Pos::NNG) |
            (Pos::NNG, Pos::MM) | (Pos::MM, Pos::NNG) |
            (Pos::EC, Pos::EF) | (Pos::EF, Pos::EC) |
            (Pos::JX, Pos::JKS) | (Pos::JKS, Pos::JX) |
            (Pos::JC, Pos::JKB) | (Pos::JKB, Pos::JC) |
            (Pos::MAG, Pos::IC) | (Pos::IC, Pos::MAG) |
            (Pos::MM, Pos::IC) | (Pos::IC, Pos::MM)
        )
    }

    fn parse_pos(s: &str) -> Option<Pos> {
        match s {
            "NNG" => Some(Pos::NNG), "NNP" => Some(Pos::NNP), "NNB" => Some(Pos::NNB),
            "NR" => Some(Pos::NR), "NP" => Some(Pos::NP),
            "VV" => Some(Pos::VV), "VA" => Some(Pos::VA), "VX" => Some(Pos::VX),
            "VCP" => Some(Pos::VCP), "VCN" => Some(Pos::VCN),
            "MAG" => Some(Pos::MAG), "MAJ" => Some(Pos::MAJ), "MM" => Some(Pos::MM),
            "IC" => Some(Pos::IC), "XR" => Some(Pos::XR),
            "JKS" => Some(Pos::JKS), "JKC" => Some(Pos::JKC), "JKG" => Some(Pos::JKG),
            "JKO" => Some(Pos::JKO), "JKB" => Some(Pos::JKB), "JKV" => Some(Pos::JKV),
            "JKQ" => Some(Pos::JKQ), "JX" => Some(Pos::JX), "JC" => Some(Pos::JC),
            "EP" => Some(Pos::EP), "EF" => Some(Pos::EF), "EC" => Some(Pos::EC),
            "ETN" => Some(Pos::ETN), "ETM" => Some(Pos::ETM),
            "XPN" => Some(Pos::XPN), "XSN" => Some(Pos::XSN), "XSV" => Some(Pos::XSV),
            "XSA" => Some(Pos::XSA),
            "SF" => Some(Pos::SF), "SP" => Some(Pos::SP), "SS" => Some(Pos::SS),
            "SE" => Some(Pos::SE), "SO" => Some(Pos::SO), "SW" => Some(Pos::SW),
            "SL" => Some(Pos::SL), "SH" => Some(Pos::SH), "SN" => Some(Pos::SN),
            _ => None,
        }
    }

    pub fn analyze_topn(&self, _text: &str, _n: usize) -> Vec<AnalyzeResult> {
        vec![self.analyze(_text)]
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.analyze(text)
            .tokens
            .into_iter()
            .map(|t| t.text)
            .collect()
    }

    pub fn has_cnn(&self) -> bool {
        self.cnn.is_some()
    }
}
