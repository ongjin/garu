use garu_core::model::{Analyzer, Model};

#[test]
fn test_load_and_analyze_v2() {
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/base.gmdl");
    let data = std::fs::read(model_path).expect("Failed to read model file");
    let model = Model::from_bytes(&data).expect("Failed to load model");
    let analyzer = Analyzer::new(model);

    // Basic analysis
    let result = analyzer.analyze("나는 학교에 갔다");
    println!("Tokens:");
    for token in &result.tokens {
        println!("  {} {:?} [{}:{}]", token.text, token.pos, token.start, token.end);
    }
    assert!(!result.tokens.is_empty(), "Should produce tokens");

    // Contraction decomposition
    let result2 = analyzer.analyze("됐다");
    let tags: Vec<&str> = result2.tokens.iter().map(|t| t.pos.as_str()).collect();
    println!("됐다 tags: {:?}", tags);
    // Should contain VV and EP from compound tag decomposition

    // Empty string
    let empty = analyzer.analyze("");
    assert!(empty.tokens.is_empty());

    // Tokenize
    let tokens = analyzer.tokenize("서울특별시 강남구");
    println!("Tokenize: {:?}", tokens);
    assert!(!tokens.is_empty());

    // Dict lookup: English proper nouns
    let result3 = analyzer.analyze("Next.js는 좋은 프레임워크이다");
    println!("\nNext.js 테스트:");
    for token in &result3.tokens {
        println!("  {} {:?} [{}:{}]", token.text, token.pos, token.start, token.end);
    }
}
