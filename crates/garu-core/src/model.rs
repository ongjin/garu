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
/// Wider pool used only for short SNS-style lexical ambiguity.
const CONTEXT_NBEST_K: usize = 10;

impl Analyzer {
    pub fn from_bytes(model_data: &[u8], cnn_data: &[u8]) -> Result<Self, String> {
        let codebook = CodebookAnalyzer::from_bytes(model_data)?;
        let cnn = Cnn2::from_bytes(cnn_data)?;
        Ok(Self { codebook, cnn })
    }

    pub fn analyze(&self, text: &str) -> AnalyzeResult {
        let mut candidates = self.codebook.analyze_topn(text, NBEST_K);
        if Self::should_expand_contextual_nbest(text) {
            candidates = self.codebook.analyze_topn(text, CONTEXT_NBEST_K);
        }

        let cnn_result = self.cnn.predict(text);
        let cnn_morphs = Self::build_cnn_morphs(&cnn_result);

        if candidates.len() <= 1 {
            if let Some(cand) = candidates.first_mut() {
                Self::apply_pos_override(&cnn_morphs, text, &mut cand.tokens);
                CodebookAnalyzer::apply_adj_root_xsa(&mut cand.tokens);
                Self::apply_protected_auxiliary_rules(&mut cand.tokens);
            }
            return candidates.into_iter().next().unwrap_or(AnalyzeResult {
                tokens: vec![], score: 0.0, elapsed_ms: 0.0,
            });
        }

        if let Some(idx) = candidates
            .iter()
            .position(|cand| Self::has_dependent_noun_adjective_nominal_de_pattern(&cand.tokens))
        {
            let mut result = candidates.swap_remove(idx);
            Self::apply_protected_auxiliary_rules(&mut result.tokens);
            return result;
        }

        if candidates
            .iter()
            .any(|cand| Self::has_dependent_noun_adjective_ec_de_pattern(&cand.tokens))
        {
            let mut expanded = self.codebook.analyze_topn(text, 12);
            if let Some(idx) = expanded
                .iter()
                .position(|cand| Self::has_dependent_noun_adjective_nominal_de_pattern(&cand.tokens))
            {
                let mut result = expanded.swap_remove(idx);
                Self::apply_protected_auxiliary_rules(&mut result.tokens);
                return result;
            }
        }

        if Self::has_dependent_noun_adjective_pattern(&candidates[0].tokens) {
            let mut result = candidates.swap_remove(0);
            Self::apply_protected_auxiliary_rules(&mut result.tokens);
            return result;
        }

        // Score each candidate by combined Viterbi cost + CNN agreement
        let mut best_idx = 0;
        let mut best_combined = f32::INFINITY;
        for (i, cand) in candidates.iter().enumerate() {
            let agreement = Self::score_cnn_agreement(&cnn_morphs, text, &cand.tokens);
            let context_bonus = Self::score_contextual_rerank_bonus(text, &cand.tokens);
            let combined = cand.score - CNN_WEIGHT * agreement - context_bonus;
            if combined < best_combined {
                best_combined = combined;
                best_idx = i;
            }
        }

        let mut result = candidates.swap_remove(best_idx);
        Self::apply_pos_override(&cnn_morphs, text, &mut result.tokens);
        CodebookAnalyzer::apply_adj_root_xsa(&mut result.tokens);
        Self::apply_protected_auxiliary_rules(&mut result.tokens);
        result
    }

    pub fn analyze_topn(&self, text: &str, n: usize) -> Vec<AnalyzeResult> {
        let mut results = self.codebook.analyze_topn(text, n);
        let cnn_result = self.cnn.predict(text);
        let cnn_morphs = Self::build_cnn_morphs(&cnn_result);
        for result in &mut results {
            Self::apply_pos_override(&cnn_morphs, text, &mut result.tokens);
            CodebookAnalyzer::apply_adj_root_xsa(&mut result.tokens);
            Self::apply_protected_auxiliary_rules(&mut result.tokens);
        }
        results
    }

    fn should_expand_contextual_nbest(text: &str) -> bool {
        text.contains("대박")
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

    fn score_contextual_rerank_bonus(text: &str, tokens: &[Token]) -> f32 {
        let mut bonus = 0.0;
        let question_like = text.contains('?') || text.ends_with('까') || text.ends_with("냐");

        for (i, token) in tokens.iter().enumerate() {
            if question_like && token.text == "뭐" && token.pos == Pos::NP {
                bonus += 4.0;
            }

            if token.text == "오늘" {
                let next = tokens.get(i + 1);
                if token.pos == Pos::NNG && next.map_or(false, |t| matches!(t.pos, Pos::NNG | Pos::NNP | Pos::NP)) {
                    bonus += 2.0;
                }
                if token.pos == Pos::MAG && next.map_or(false, |t| matches!(t.pos, Pos::MAG | Pos::VA | Pos::VV)) {
                    bonus += 2.0;
                }
                if token.pos == Pos::MAG && next.map_or(false, |t| t.text == "너무" && t.pos == Pos::MAG) {
                    bonus += 3.0;
                }
            }

            if token.text == "있"
                && token.pos == Pos::VA
                && i > 0
                && matches!(tokens[i - 1].pos, Pos::NNG | Pos::NNP | Pos::NP)
                && tokens.get(i + 1).map_or(false, |t| t.pos == Pos::EF)
            {
                bonus += 5.0;
            }

            if token.text == "냐" && token.pos == Pos::EF {
                bonus += 4.0;
            }

            if token.text == "대박" && matches!(token.pos, Pos::NNG | Pos::IC) {
                bonus += 4.0;
            }
        }

        for window in tokens.windows(2) {
            if window[0].pos == Pos::NNG && window[1].text == "냐" && window[1].pos == Pos::EF {
                let surface = format!("{}냐", window[0].text);
                if text.contains(&surface) {
                    bonus += 3.0;
                }
            }

            if window[0].text == "피곤" && window[0].pos == Pos::NNG
                && window[1].text == "하" && window[1].pos == Pos::XSA
            {
                bonus += 5.0;
            }
        }

        bonus
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
                if Self::is_protected_auxiliary(tokens, vs + i) {
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

    fn is_protected_auxiliary(tokens: &[Token], idx: usize) -> bool {
        if idx < 2 || tokens[idx].pos != Pos::VX || tokens[idx].text != "있" {
            return false;
        }
        tokens[idx - 1].pos == Pos::NNB
            && tokens[idx - 1].text == "수"
            && tokens[idx - 2].pos == Pos::ETM
    }

    fn apply_protected_auxiliary_rules(tokens: &mut [Token]) {
        for i in 2..tokens.len() {
            if tokens[i].text == "있"
                && matches!(tokens[i].pos, Pos::VV | Pos::VA)
                && tokens[i - 1].pos == Pos::NNB
                && tokens[i - 1].text == "수"
                && tokens[i - 2].pos == Pos::ETM
                && tokens[i].start == tokens[i - 1].start
                && tokens[i - 1].start == tokens[i - 2].start
            {
                tokens[i].pos = Pos::VX;
            }
        }
    }

    fn has_dependent_noun_adjective_pattern(tokens: &[Token]) -> bool {
        tokens.windows(3).any(|window| {
            window[0].pos == Pos::ETM
                && window[1].text == "만"
                && window[1].pos == Pos::NNB
                && window[2].text == "하"
                && window[2].pos == Pos::XSA
        })
    }

    fn has_dependent_noun_adjective_nominal_de_pattern(tokens: &[Token]) -> bool {
        tokens.windows(5).any(|window| {
            window[0].pos == Pos::ETM
                && window[1].text == "만"
                && window[1].pos == Pos::NNB
                && window[2].text == "하"
                && window[2].pos == Pos::XSA
                && window[3].text == "ㄴ"
                && window[3].pos == Pos::ETM
                && window[4].text == "데"
                && window[4].pos == Pos::NNB
        })
    }

    fn has_dependent_noun_adjective_ec_de_pattern(tokens: &[Token]) -> bool {
        tokens.windows(4).any(|window| {
            window[0].pos == Pos::ETM
                && window[1].text == "만"
                && window[1].pos == Pos::NNB
                && window[2].text == "하"
                && window[2].pos == Pos::XSA
                && window[3].text == "ㄴ데"
                && window[3].pos == Pos::EC
        })
    }
}
