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

    /// CNN reranking: override POS tags where CNN disagrees with Viterbi.
    /// CNN operates at syllable level (BIO tagging), so we compare
    /// CNN's POS prediction with Viterbi's POS for each morpheme.
    fn apply_cnn_rerank(cnn: &Cnn2, text: &str, tokens: &mut [Token]) {
        let cnn_result = cnn.predict(text);
        if cnn_result.is_empty() {
            return;
        }

        // Build CNN morpheme sequence from BIO tags
        let mut cnn_morphs: Vec<(String, Pos)> = Vec::new();
        let mut cur_form = String::new();
        let mut cur_pos: Option<Pos> = None;

        for &(ch, label) in &cnn_result {
            if ch == ' ' {
                if !cur_form.is_empty() {
                    if let Some(p) = cur_pos {
                        cnn_morphs.push((cur_form.clone(), p));
                    }
                    cur_form.clear();
                    cur_pos = None;
                }
                continue;
            }
            if label.starts_with("B-") {
                if !cur_form.is_empty() {
                    if let Some(p) = cur_pos {
                        cnn_morphs.push((cur_form.clone(), p));
                    }
                }
                cur_form = ch.to_string();
                cur_pos = Self::parse_pos(&label[2..]);
            } else if label.starts_with("I-") {
                cur_form.push(ch);
            } else {
                if !cur_form.is_empty() {
                    if let Some(p) = cur_pos {
                        cnn_morphs.push((cur_form.clone(), p));
                    }
                }
                cur_form = ch.to_string();
                cur_pos = Some(Pos::SW);
            }
        }
        if !cur_form.is_empty() {
            if let Some(p) = cur_pos {
                cnn_morphs.push((cur_form, p));
            }
        }

        // Sequential matching: align Viterbi tokens with CNN morphemes by order.
        // Override POS where they disagree on known ambiguous pairs.
        let mut ci = 0;
        for token in tokens.iter_mut() {
            // Find matching CNN morpheme by text
            while ci < cnn_morphs.len() && cnn_morphs[ci].0 != token.text {
                ci += 1;
            }
            if ci >= cnn_morphs.len() {
                break;
            }
            let cnn_pos = cnn_morphs[ci].1;
            ci += 1;

            if cnn_pos == token.pos {
                continue;
            }

            let should_override = matches!(
                (token.pos, cnn_pos),
                (Pos::NP, Pos::VV) | (Pos::VV, Pos::NP) |
                (Pos::XSV, Pos::XSA) | (Pos::XSA, Pos::XSV) |
                (Pos::VV, Pos::VX) | (Pos::VX, Pos::VV) |
                (Pos::VA, Pos::VX) | (Pos::VX, Pos::VA) |
                (Pos::MM, Pos::ETM) | (Pos::ETM, Pos::MM) |
                (Pos::NNG, Pos::NNP) | (Pos::NNP, Pos::NNG) |
                (Pos::JX, Pos::JKS) | (Pos::JKS, Pos::JX)
            );
            if should_override {
                token.pos = cnn_pos;
            }
        }
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
