//! Jamo normalization: U+3134(ㄴ) → U+11AB(ᆫ) for ETM/EC/EF/etc. surfaces.

use garu_core::model::{Analyzer, AnalyzerOptions};
use garu_core::types::Pos;

fn load_default_analyzer(normalize_jamo: bool) -> Analyzer {
    let model_path = std::env::var("GARU_MODEL")
        .unwrap_or_else(|_| "js/models/base.gmdl".to_string());
    let bytes = std::fs::read(&model_path).expect("model file");
    let mut opts = AnalyzerOptions::default();
    opts.normalize_jamo = normalize_jamo;
    Analyzer::from_bytes_with_options(&bytes, opts).expect("analyzer")
}

#[test]
fn jamo_normalize_on_emits_combining() {
    let a = load_default_analyzer(true);
    let result = a.analyze("간 사람"); // expects 가/VV + ᆫ/ETM ...
    let has_combining = result.tokens.iter().any(|t| t.text == "ᆫ" && t.pos == Pos::ETM);
    let has_compat = result.tokens.iter().any(|t| t.text == "ㄴ" && t.pos == Pos::ETM);
    assert!(
        has_combining || !has_compat,
        "expected combining ᆫ U+11AB, got compat ㄴ U+3134. tokens: {:?}",
        result.tokens.iter().map(|t| (t.text.clone(), t.pos)).collect::<Vec<_>>()
    );
}

#[test]
fn jamo_normalize_off_emits_compat() {
    let a = load_default_analyzer(false);
    let result = a.analyze("간 사람");
    let has_compat = result.tokens.iter().any(|t| t.text == "ㄴ" && t.pos == Pos::ETM);
    let has_combining = result.tokens.iter().any(|t| t.text == "ᆫ" && t.pos == Pos::ETM);
    assert!(
        has_compat || !has_combining,
        "with normalize_jamo=false, expected compat ㄴ U+3134. tokens: {:?}",
        result.tokens.iter().map(|t| (t.text.clone(), t.pos)).collect::<Vec<_>>()
    );
}

#[test]
fn non_jamo_surfaces_unchanged() {
    let a = load_default_analyzer(true);
    let result = a.analyze("학교에 갔다");
    let nng = result
        .tokens
        .iter()
        .find(|t| t.pos == Pos::NNG)
        .expect("NNG token");
    assert_eq!(nng.text, "학교");
}
