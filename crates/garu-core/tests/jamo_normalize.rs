//! Jamo normalization: U+3139(ㄹ) → U+11AF(ᆯ) for ETM surfaces, etc.
//! "갈 사람" → 가/VV + ㄹ/ETM(U+3139 compat) or ᆯ/ETM(U+11AF combining).

use garu_core::model::{Analyzer, AnalyzerOptions};
use garu_core::types::Pos;

fn load_default_analyzer(normalize_jamo: bool) -> Analyzer {
    let model_path = std::env::var("GARU_MODEL")
        .unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/codebook.gmdl").to_string()
        });
    let bytes = std::fs::read(&model_path).expect("model file");
    let mut opts = AnalyzerOptions::default();
    opts.normalize_jamo = normalize_jamo;
    Analyzer::from_bytes_with_options(&bytes, opts).expect("analyzer")
}

#[test]
fn jamo_normalize_on_emits_combining() {
    let a = load_default_analyzer(true);
    // "갈 사람" → 가/VV + ㄹ/ETM. With normalize_jamo=true, ㄹ(U+3139) → ᆯ(U+11AF).
    let result = a.analyze("갈 사람");
    let has_combining = result.tokens.iter().any(|t| t.text == "ᆯ" && t.pos == Pos::ETM);
    let has_compat = result.tokens.iter().any(|t| t.text == "ㄹ" && t.pos == Pos::ETM);
    assert!(
        has_combining,
        "expected combining ᆯ U+11AF with normalize_jamo=true, but got: has_combining={has_combining}, has_compat={has_compat}. tokens: {:?}",
        result.tokens.iter().map(|t| (t.text.clone(), t.pos)).collect::<Vec<_>>()
    );
    assert!(
        !has_compat,
        "should not have compat ㄹ U+3139 with normalize_jamo=true. tokens: {:?}",
        result.tokens.iter().map(|t| (t.text.clone(), t.pos)).collect::<Vec<_>>()
    );
}

#[test]
fn jamo_normalize_off_emits_compat() {
    let a = load_default_analyzer(false);
    // "갈 사람" → 가/VV + ㄹ/ETM. With normalize_jamo=false, stays compat U+3139.
    let result = a.analyze("갈 사람");
    let has_compat = result.tokens.iter().any(|t| t.text == "ㄹ" && t.pos == Pos::ETM);
    let has_combining = result.tokens.iter().any(|t| t.text == "ᆯ" && t.pos == Pos::ETM);
    assert!(
        has_compat,
        "with normalize_jamo=false, expected compat ㄹ U+3139. tokens: {:?}",
        result.tokens.iter().map(|t| (t.text.clone(), t.pos)).collect::<Vec<_>>()
    );
    assert!(
        !has_combining,
        "should not have combining ᆯ U+11AF with normalize_jamo=false. tokens: {:?}",
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
