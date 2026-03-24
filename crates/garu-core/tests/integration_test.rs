use garu_core::model::{Analyzer, Model};

#[test]
fn test_load_and_analyze() {
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/base.gmdl");
    let data = std::fs::read(model_path).expect("Failed to read model file");
    let model = Model::from_bytes(&data).expect("Failed to load model");
    let analyzer = Analyzer::new(model);

    let result = analyzer.analyze("나는 학교에 갔다");
    println!("Tokens:");
    for token in &result.tokens {
        println!("  {} {:?} [{}:{}]", token.text, token.pos, token.start, token.end);
    }
    assert!(!result.tokens.is_empty(), "Should produce tokens");
    assert!(result.elapsed_ms >= 0.0);

    // Empty string
    let empty = analyzer.analyze("");
    assert!(empty.tokens.is_empty());

    // Tokenize
    let tokens = analyzer.tokenize("서울특별시 강남구");
    println!("Tokenize: {:?}", tokens);
    assert!(!tokens.is_empty());
}
