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
    pub fn new(model_data: &[u8], cnn_data: &[u8]) -> Result<GaruWasm, JsError> {
        let analyzer = Analyzer::from_bytes(model_data, cnn_data)
            .map_err(|e| JsError::new(&e))?;
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

    #[wasm_bindgen]
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}
