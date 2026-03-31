use garu_core::model::Analyzer;

#[test]
fn test_codebook_analyzer_v3() {
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/codebook.gmdl");
    let data = std::fs::read(model_path).expect("Failed to read codebook model");
    let analyzer = Analyzer::from_bytes(&data).expect("Failed to load codebook model");

    // Basic analysis
    let result = analyzer.analyze("나는 학교에서 공부했다");
    println!("\n=== Codebook v3 ===");
    println!("Input: 나는 학교에서 공부했다");
    for token in &result.tokens {
        println!("  {} {:?} [{}:{}]", token.text, token.pos, token.start, token.end);
    }
    assert!(!result.tokens.is_empty(), "Should produce tokens");

    // Contracted verbs
    let result2 = analyzer.analyze("했다");
    println!("\n했다 tokens:");
    for token in &result2.tokens {
        println!("  {} {:?}", token.text, token.pos);
    }
    assert!(!result2.tokens.is_empty());

    // Empty string
    let empty = analyzer.analyze("");
    assert!(empty.tokens.is_empty());

    // English/mixed text
    let result3 = analyzer.analyze("TypeScript는 좋다");
    println!("\nTypeScript는 좋다:");
    for token in &result3.tokens {
        println!("  {} {:?} [{}:{}]", token.text, token.pos, token.start, token.end);
    }
    assert!(!result3.tokens.is_empty());

    // Tokenize
    let tokens = analyzer.tokenize("서울에서 공부했다");
    println!("\nTokenize: {:?}", tokens);
    assert!(!tokens.is_empty());

    // Jongseong split tests
    let jongseong_cases = [
        ("고친다", vec![("고치", "VV"), ("ㄴ다", "EF")]),
        ("소 잃고 외양간 고친다", vec![("소", "NNG"), ("잃", "VV"), ("고", "EC"), ("외양간", "NNG"), ("고치", "VV"), ("ㄴ다", "EF")]),
    ];
    for (input, expected) in &jongseong_cases {
        let r = analyzer.analyze(input);
        println!("\n{input}:");
        for t in &r.tokens { println!("  {}\t{:?}", t.text, t.pos); }
        assert_eq!(r.tokens.len(), expected.len(), "Token count mismatch for '{input}'");
        for (t, (exp_text, _)) in r.tokens.iter().zip(expected.iter()) {
            assert_eq!(&t.text, exp_text, "Mismatch in '{input}': got '{}', expected '{}'", t.text, exp_text);
        }
    }
}
