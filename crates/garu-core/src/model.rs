//! Analyzer — thin wrapper around CodebookAnalyzer.

use crate::codebook::CodebookAnalyzer;
use crate::types::AnalyzeResult;

pub struct Analyzer {
    codebook: CodebookAnalyzer,
}

impl Analyzer {
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let codebook = CodebookAnalyzer::from_bytes(data)?;
        Ok(Self { codebook })
    }

    pub fn analyze(&self, text: &str) -> AnalyzeResult {
        self.codebook.analyze(text)
    }

    pub fn analyze_topn(&self, _text: &str, _n: usize) -> Vec<AnalyzeResult> {
        // Codebook doesn't support topN — return single result
        vec![self.analyze(_text)]
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.analyze(text)
            .tokens
            .into_iter()
            .map(|t| t.text)
            .collect()
    }
}
