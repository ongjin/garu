//! Analyzer — CodebookAnalyzer + CNN reranker.
//!
//! Always generates top-N Viterbi candidates and selects the best one
//! based on combined Viterbi cost + CNN agreement score.

use crate::codebook::CodebookAnalyzer;
use crate::cnn::Cnn2;
use crate::types::{AnalyzeResult, Pos, Token};

pub struct Analyzer {
    codebook: CodebookAnalyzer,
    cnn: Cnn2,
}

/// CNN reranking weight: how much CNN agreement influences candidate selection.
const CNN_WEIGHT: f32 = 5.0;
/// Confidence threshold for fine-grained POS override on selected candidate.
const POS_CONFIDENCE: f32 = 0.9;
/// Number of Viterbi candidates to generate.
const NBEST_K: usize = 5;

impl Analyzer {
    pub fn from_bytes(model_data: &[u8], cnn_data: &[u8]) -> Result<Self, String> {
        let codebook = CodebookAnalyzer::from_bytes(model_data)?;
        let cnn = Cnn2::from_bytes(cnn_data)?;
        Ok(Self { codebook, cnn })
    }

    pub fn analyze(&self, text: &str) -> AnalyzeResult {
        let mut candidates = self.codebook.analyze_topn(text, NBEST_K);

        let cnn_result = self.cnn.predict(text);
        let cnn_morphs = Self::build_cnn_morphs(&cnn_result);

        if candidates.len() <= 1 {
            if let Some(cand) = candidates.first_mut() {
                Self::apply_pos_override(&cnn_morphs, text, &mut cand.tokens);
            }
            return candidates.into_iter().next().unwrap_or(AnalyzeResult {
                tokens: vec![], score: 0.0, elapsed_ms: 0.0,
            });
        }

        // Score each candidate by combined Viterbi cost + CNN agreement
        let mut best_idx = 0;
        let mut best_combined = f32::INFINITY;
        for (i, cand) in candidates.iter().enumerate() {
            let agreement = Self::score_cnn_agreement(&cnn_morphs, text, &cand.tokens);
            let combined = cand.score - CNN_WEIGHT * agreement;
            if combined < best_combined {
                best_combined = combined;
                best_idx = i;
            }
        }

        let mut result = candidates.swap_remove(best_idx);
        Self::apply_pos_override(&cnn_morphs, text, &mut result.tokens);
        result
    }

    pub fn analyze_topn(&self, text: &str, n: usize) -> Vec<AnalyzeResult> {
        let mut results = self.codebook.analyze_topn(text, n);
        let cnn_result = self.cnn.predict(text);
        let cnn_morphs = Self::build_cnn_morphs(&cnn_result);
        for result in &mut results {
            Self::apply_pos_override(&cnn_morphs, text, &mut result.tokens);
        }
        results
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.analyze(text).tokens.into_iter().map(|t| t.text).collect()
    }

    // -----------------------------------------------------------------------
    // CNN morpheme extraction from BIO tags
    // -----------------------------------------------------------------------

    fn build_cnn_morphs<'a>(cnn_result: &[(char, &'a str, f32)]) -> Vec<(String, Pos, f32)> {
        let mut morphs: Vec<(String, Pos, f32)> = Vec::new();
        let mut cur_form = String::new();
        let mut cur_pos: Option<Pos> = None;
        let mut cur_conf: f32 = 1.0;

        for &(ch, label, conf) in cnn_result {
            if ch == ' ' {
                if let Some(p) = cur_pos.take() {
                    if !cur_form.is_empty() { morphs.push((cur_form.clone(), p, cur_conf)); }
                }
                cur_form.clear();
                cur_conf = 1.0;
                continue;
            }
            if label.starts_with("B-") {
                if let Some(p) = cur_pos.take() {
                    if !cur_form.is_empty() { morphs.push((cur_form.clone(), p, cur_conf)); }
                }
                cur_form = ch.to_string();
                cur_pos = Pos::from_str(&label[2..]);
                cur_conf = conf;
            } else if label.starts_with("I-") {
                cur_form.push(ch);
                cur_conf = cur_conf.min(conf);
            } else {
                if let Some(p) = cur_pos.take() {
                    if !cur_form.is_empty() { morphs.push((cur_form.clone(), p, cur_conf)); }
                }
                cur_form = ch.to_string();
                cur_pos = Some(Pos::SW);
                cur_conf = conf;
            }
        }
        if let Some(p) = cur_pos {
            if !cur_form.is_empty() { morphs.push((cur_form, p, cur_conf)); }
        }
        morphs
    }

    // -----------------------------------------------------------------------
    // CNN agreement scoring for N-best reranking
    // -----------------------------------------------------------------------

    fn score_cnn_agreement(
        cnn_morphs: &[(String, Pos, f32)],
        text: &str,
        tokens: &[Token],
    ) -> f32 {
        let vit_groups = Self::eojeol_groups(tokens);
        let cnn_groups = Self::cnn_eojeol_groups(cnn_morphs, text);

        let mut score = 0.0;
        for (eg_idx, &(vs, ve)) in vit_groups.iter().enumerate() {
            if eg_idx >= cnn_groups.len() { break; }
            let (cs, ce) = cnn_groups[eg_idx];
            let vit = &tokens[vs..ve];
            let cnn = &cnn_morphs[cs..ce];

            let same_seg = vit.len() == cnn.len()
                && vit.iter().zip(cnn.iter()).all(|(v, c)| v.text == c.0);

            if same_seg {
                for (v, c) in vit.iter().zip(cnn.iter()) {
                    if v.pos == c.1 {
                        score += c.2;
                    }
                }
            } else {
                for cm in cnn {
                    if cm.2 > 0.8 && vit.iter().any(|v| v.pos == cm.1) {
                        score += cm.2 * 0.3;
                    }
                }
            }
        }
        score
    }

    // -----------------------------------------------------------------------
    // POS-level override (applied on final selected candidate)
    // -----------------------------------------------------------------------

    fn apply_pos_override(
        cnn_morphs: &[(String, Pos, f32)],
        text: &str,
        tokens: &mut Vec<Token>,
    ) {
        let vit_groups = Self::eojeol_groups(tokens);
        let cnn_groups = Self::cnn_eojeol_groups(cnn_morphs, text);

        for (eg_idx, &(vs, ve)) in vit_groups.iter().enumerate() {
            if eg_idx >= cnn_groups.len() { break; }
            let (cs, ce) = cnn_groups[eg_idx];
            let cnn_slice = &cnn_morphs[cs..ce];
            let vit_len = ve - vs;

            if vit_len != cnn_slice.len() { continue; }
            let same_seg = (0..vit_len).all(|i| tokens[vs + i].text == cnn_slice[i].0);
            if !same_seg { continue; }

            for i in 0..vit_len {
                let (_, cnn_pos, conf) = &cnn_slice[i];
                if *cnn_pos == tokens[vs + i].pos || *conf < POS_CONFIDENCE {
                    continue;
                }
                if Self::is_ambiguous_pair(tokens[vs + i].pos, *cnn_pos) {
                    tokens[vs + i].pos = *cnn_pos;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Eojeol grouping helpers
    // -----------------------------------------------------------------------

    fn eojeol_groups(tokens: &[Token]) -> Vec<(usize, usize)> {
        let mut groups = Vec::new();
        if tokens.is_empty() { return groups; }
        let mut gs = 0;
        for i in 1..tokens.len() {
            if tokens[i].start != tokens[gs].start {
                groups.push((gs, i));
                gs = i;
            }
        }
        groups.push((gs, tokens.len()));
        groups
    }

    fn cnn_eojeol_groups(cnn_morphs: &[(String, Pos, f32)], text: &str) -> Vec<(usize, usize)> {
        let mut groups = Vec::new();
        let text_chars: Vec<char> = text.chars().collect();
        let mut gs = 0;
        let mut char_pos = 0;
        for (ci, (form, _, _)) in cnn_morphs.iter().enumerate() {
            let new_char_pos = char_pos + form.chars().count();
            if ci + 1 < cnn_morphs.len() && new_char_pos < text_chars.len()
                && text_chars[new_char_pos] == ' '
            {
                groups.push((gs, ci + 1));
                gs = ci + 1;
            }
            char_pos = new_char_pos;
        }
        if gs < cnn_morphs.len() {
            groups.push((gs, cnn_morphs.len()));
        }
        groups
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

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
}
