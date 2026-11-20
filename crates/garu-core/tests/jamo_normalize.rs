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

#[test]
fn decomposed_training_candidates() {
    let analyzer = load_default_analyzer(true);
    let expected = analyzer.analyze_topn_for_rerank("갈 사람", 5);
    let actual = analyzer.analyze_topn_for_rerank("갈 사람", 5);
    // Source code-point boundaries for 갈, space, 사, 람.
    let offsets = [0, 3, 4, 6, 9];
    assert_eq!(actual.len(), expected.len());
    for (raw, canonical) in actual.iter().zip(&expected) {
        assert_eq!(raw.score, canonical.score);
        assert_eq!(raw.tokens.len(), canonical.tokens.len());
        for (a, b) in raw.tokens.iter().zip(&canonical.tokens) {
            assert_eq!((&a.text, a.pos), (&b.text, b.pos));
            assert_eq!((a.start, a.end), (offsets[b.start], offsets[b.end]));
        }
    }
}

#[test]
fn promise_display_and_reranker() {
    let analyzer = load_default_analyzer(true);
    let text = "할게요!";
    let selected = analyzer.analyze(text);
    let public = analyzer.analyze_topn(text, 10);
    for result in [&selected, &public[0]] {
        let forms: Vec<_> = result.tokens.iter().map(|t| (t.text.as_str(), t.pos)).collect();
        assert_eq!(forms, [("하", Pos::VV), ("ᆯ게", Pos::EF), ("요", Pos::JX), ("!", Pos::SF)]);
        assert!(result.tokens.iter().all(|t| t.start == 0 && t.end == text.chars().count()));
    }
    let raw = analyzer.analyze_topn_for_rerank(text, 10);
    assert!(raw[0].tokens.iter().any(|t| t.text == "ㄹ게요" && t.pos == Pos::EF));
}
