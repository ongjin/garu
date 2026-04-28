use garu_core::model::Analyzer;
use garu_core::types::Pos;

fn load_analyzer() -> Analyzer {
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/codebook.gmdl");
    let cnn_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/cnn2.bin");
    let model_data = std::fs::read(model_path).expect("Failed to read model");
    let cnn_data = std::fs::read(cnn_path).expect("Failed to read CNN model");
    Analyzer::from_bytes(&model_data, &cnn_data).expect("Failed to load analyzer")
}

#[test]
fn test_codebook_analyzer_v3() {
    let analyzer = load_analyzer();

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

    // Conjugation tests: jongseong split + vowel contraction
    let conjugation_cases: Vec<(&str, Vec<&str>)> = vec![
        ("고친다", vec!["고치", "ㄴ다"]),
        ("소 잃고 외양간 고친다", vec!["소", "잃", "고", "외양간", "고치", "ㄴ다"]),
        ("건너라", vec!["건너", "어라"]),
        ("가라", vec!["가", "아라"]),
    ];
    for (input, expected) in &conjugation_cases {
        let r = analyzer.analyze(input);
        println!("\n{input}:");
        for t in &r.tokens { println!("  {}\t{:?}", t.text, t.pos); }
        let got: Vec<&str> = r.tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(got, *expected, "Mismatch for '{input}'");
    }

    // XSV/XSA disambiguation tests
    let xsa_cases = [
        "어제 늦게까지 공부했더니 피곤하다",
        "피곤한 하루",
        "공부하는 학생",
    ];
    for input in &xsa_cases {
        let r = analyzer.analyze(input);
        println!("\n{input}:");
        for t in &r.tokens { println!("  {}\t{:?}", t.text, t.pos); }
    }
}

#[test]
fn test_cnn2_ensemble() {
    let analyzer = load_analyzer();

    // Test: "나는 하늘을 나는 새를 보았다"
    let r = analyzer.analyze("나는 하늘을 나는 새를 보았다");
    println!("\n나는 하늘을 나는 새를 보았다:");
    for t in &r.tokens { println!("  {}\t{:?}", t.text, t.pos); }
    assert_eq!(r.tokens[0].text, "나");
    assert_eq!(r.tokens[0].pos, garu_core::types::Pos::NP);

    // Test: 피곤하다
    let r2 = analyzer.analyze("피곤하다");
    println!("\n피곤하다:");
    for t in &r2.tokens { println!("  {}\t{:?}", t.text, t.pos); }

    // Test: 공부하는 학생
    let r3 = analyzer.analyze("공부하는 학생");
    println!("\n공부하는 학생:");
    for t in &r3.tokens { println!("  {}\t{:?}", t.text, t.pos); }
}

fn assert_analysis(analyzer: &Analyzer, input: &str, expected: &[(&str, Pos)]) {
    let result = analyzer.analyze(input);
    let got: Vec<(&str, Pos)> = result
        .tokens
        .iter()
        .map(|token| (token.text.as_str(), token.pos))
        .collect();
    assert_eq!(got, expected, "analysis mismatch for '{input}'");
}

#[test]
fn test_dependency_noun_constructions() {
    let analyzer = load_analyzer();

    assert_analysis(
        &analyzer,
        "갈수있는데",
        &[
            ("가", Pos::VV),
            ("ㄹ", Pos::ETM),
            ("수", Pos::NNB),
            ("있", Pos::VX),
            ("는데", Pos::EC),
        ],
    );

    assert_analysis(
        &analyzer,
        "올만한데",
        &[
            ("오", Pos::VV),
            ("ㄹ", Pos::ETM),
            ("만", Pos::NNB),
            ("하", Pos::XSA),
            ("ㄴ", Pos::ETM),
            ("데", Pos::NNB),
        ],
    );

    assert_analysis(
        &analyzer,
        "볼만하다",
        &[
            ("보", Pos::VV),
            ("ㄹ", Pos::ETM),
            ("만", Pos::NNB),
            ("하", Pos::XSA),
            ("다", Pos::EF),
        ],
    );

    assert_analysis(
        &analyzer,
        "갈만해",
        &[
            ("가", Pos::VV),
            ("ㄹ", Pos::ETM),
            ("만", Pos::NNB),
            ("하", Pos::XSA),
            ("아", Pos::EF),
        ],
    );

    assert_analysis(
        &analyzer,
        "들만한데",
        &[
            ("들", Pos::VV),
            ("ㄹ", Pos::ETM),
            ("만", Pos::NNB),
            ("하", Pos::XSA),
            ("ㄴ", Pos::ETM),
            ("데", Pos::NNB),
        ],
    );

    assert_analysis(
        &analyzer,
        "갈리없는데",
        &[
            ("가", Pos::VV),
            ("ㄹ", Pos::ETM),
            ("리", Pos::NNB),
            ("없", Pos::VA),
            ("는데", Pos::EC),
        ],
    );
}

#[test]
fn test_nbest_path_applies_determiner_postprocess() {
    let analyzer = load_analyzer();
    let result = analyzer.analyze("검찰은 방 전 사장을 조만간 피의자 신분으로 소환해 조사할 방침이다.");

    let window = result
        .tokens
        .windows(3)
        .find(|tokens| tokens[0].text == "방" && tokens[1].text == "전" && tokens[2].text == "사장")
        .expect("expected 방 전 사장 window");

    assert_eq!(window[1].pos, Pos::MM);
}
