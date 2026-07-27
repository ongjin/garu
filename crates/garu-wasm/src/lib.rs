use garu_core::model::Analyzer;
use garu_core::types::AnalyzeResult;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct JsToken {
    text: String,
    pos: String,
    start: usize,
    end: usize,
    score: Option<f32>,
}

#[derive(Serialize)]
struct JsAnalyzeResult {
    tokens: Vec<JsToken>,
    score: f32,
    elapsed: f64,
}

fn convert_result(result: AnalyzeResult) -> JsAnalyzeResult {
    JsAnalyzeResult {
        tokens: result
            .tokens
            .into_iter()
            .map(|t| JsToken {
                text: t.text,
                pos: t.pos.as_str().to_string(),
                start: t.start,
                end: t.end,
                score: t.score,
            })
            .collect(),
        score: result.score,
        elapsed: result.elapsed_ms,
    }
}

#[wasm_bindgen]
pub struct GaruWasm {
    analyzer: Analyzer,
}

#[wasm_bindgen]
impl GaruWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(model_data: &[u8], normalize_jamo: Option<bool>) -> Result<GaruWasm, JsError> {
        let mut opts = garu_core::AnalyzerOptions::default();
        if let Some(b) = normalize_jamo {
            opts.normalize_jamo = b;
        }
        let analyzer = Analyzer::from_bytes_with_options(model_data, opts)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(GaruWasm { analyzer })
    }

    pub fn analyze(&self, text: &str) -> Result<JsValue, JsError> {
        let result = self.analyzer.analyze(text);
        let js_result = convert_result(result);
        serde_wasm_bindgen::to_value(&js_result).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn analyze_topn(&self, text: &str, n: usize) -> Result<JsValue, JsError> {
        let results = self.analyzer.analyze_topn(text, n);
        let js_results: Vec<JsAnalyzeResult> = results.into_iter().map(convert_result).collect();
        serde_wasm_bindgen::to_value(&js_results).map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn tokenize(&self, text: &str) -> Result<JsValue, JsError> {
        let tokens = self.analyzer.tokenize(text);
        serde_wasm_bindgen::to_value(&tokens).map_err(|e| JsError::new(&e.to_string()))
    }

    /// 사용자 단어 등록 — 모델 리빌드 없이 도메인 어휘를 인식시킨다.
    pub fn add_user_word(&mut self, surface: &str, pos: &str, freq: Option<u32>) -> Result<(), JsError> {
        let parsed = garu_core::types::Pos::from_str(pos)
            .ok_or_else(|| JsError::new(&format!("unknown POS tag: {pos}")))?;
        self.analyzer.add_user_word(surface, parsed, freq);
        Ok(())
    }

    /// 등록된 사용자 단어를 모두 제거.
    pub fn clear_user_words(&mut self) {
        self.analyzer.clear_user_words();
    }

    /// 등록된 사용자 단어 수.
    pub fn user_word_count(&self) -> usize {
        self.analyzer.user_word_count()
    }

    #[wasm_bindgen]
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}
