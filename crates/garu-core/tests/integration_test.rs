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
            ("ㄴ데", Pos::EC),
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
            ("ㄴ데", Pos::EC),
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

#[test]
fn test_contextual_reranking_everyday_and_sns() {
    let analyzer = load_analyzer();

    assert_analysis(
        &analyzer,
        "오늘 점심 뭐 먹을까?",
        &[
            ("오늘", Pos::NNG),
            ("점심", Pos::NNG),
            ("뭐", Pos::NP),
            ("먹", Pos::VV),
            ("을까", Pos::EF),
            ("?", Pos::SF),
        ],
    );

    assert_analysis(
        &analyzer,
        "여기 자리 있어요?",
        &[
            ("여기", Pos::NP),
            ("자리", Pos::NNG),
            ("있", Pos::VA),
            ("어요", Pos::EF),
            ("?", Pos::SF),
        ],
    );

    assert_analysis(
        &analyzer,
        "이거 실화냐 대박",
        &[
            ("이거", Pos::NP),
            ("실화", Pos::NNG),
            ("냐", Pos::EF),
            ("대박", Pos::NNG),
        ],
    );

    assert_analysis(
        &analyzer,
        "아 진짜 오늘 너무 피곤하다",
        &[
            ("아", Pos::IC),
            ("진짜", Pos::MAG),
            ("오늘", Pos::MAG),
            ("너무", Pos::MAG),
            ("피곤", Pos::NNG),
            ("하", Pos::XSA),
            ("다", Pos::EF),
        ],
    );
}

// ----- Issue #1 (sgbai78) regression suite -----

#[test]
fn test_issue1_sentence_final_ef_no_punctuation() {
    let analyzer = load_analyzer();

    // 어/아 EC at sentence end after VV/VA/EP should become EF.
    // Surface form follows Garu's vowel-harmony rule: 어→아 only when the
    // preceding syllable's vowel is ㅏ/ㅗ (배고프 ㅡ stays 어, 가 ㅏ becomes 아).
    assert_analysis(
        &analyzer,
        "배고파",
        &[("배고프", Pos::VA), ("어", Pos::EF)],
    );
    assert_analysis(
        &analyzer,
        "졸려",
        &[("졸리", Pos::VV), ("어", Pos::EF)],
    );

    let result = analyzer.analyze("다 끝났어");
    let last = result.tokens.last().expect("must have tokens");
    assert_eq!(last.pos, Pos::EF, "last morpheme must be EF in '다 끝났어'");

    let result = analyzer.analyze("나 이제 가");
    let last = result.tokens.last().expect("must have tokens");
    assert_eq!(last.pos, Pos::EF, "last morpheme must be EF in '나 이제 가'");
}

#[test]
fn test_issue1_haeyo_endings() {
    let analyzer = load_analyzer();

    // 어때요 → 어떻/VA + 어요/EF
    assert_analysis(
        &analyzer,
        "어때요",
        &[("어떻", Pos::VA), ("어요", Pos::EF)],
    );

    // 감사해요 → 감사/NNG + 하/XSV + 아요/EF (어근 회복)
    assert_analysis(
        &analyzer,
        "감사해요",
        &[("감사", Pos::NNG), ("하", Pos::XSV), ("아요", Pos::EF)],
    );
}

#[test]
fn test_issue1_mag_copula_ya() {
    let analyzer = load_analyzer();

    // 별로야 → 별로/MAG + 이/VCP + 야/EF
    assert_analysis(
        &analyzer,
        "별로야",
        &[("별로", Pos::MAG), ("이", Pos::VCP), ("야", Pos::EF)],
    );
}

#[test]
fn test_issue1_span_arc_guard_eojweo() {
    let analyzer = load_analyzer();

    // 틀어줘 → 틀/VV + 어/EC + 주/VX + 어/EF (no NNG span)
    let result = analyzer.analyze("틀어줘");
    let pos_seq: Vec<Pos> = result.tokens.iter().map(|t| t.pos).collect();
    assert!(
        !pos_seq.contains(&Pos::NNG) || result.tokens.len() > 1,
        "틀어줘 should not be a single NNG span: {:?}",
        result.tokens.iter().map(|t| (t.text.as_str(), t.pos)).collect::<Vec<_>>()
    );
    assert!(
        result.tokens.iter().any(|t| t.text == "주" && t.pos == Pos::VX),
        "틀어줘 must contain 주/VX"
    );
}

#[test]
fn test_issue1_jongseong_ss_fallback() {
    let analyzer = load_analyzer();

    // 갔거든 / 왔거든 / 봤거든: ㅆ jongseong split must reach EP+EF combinations
    // such as 았거든/었거든 even though `ㅆ거든` is absent from the suffix codebook.
    assert_analysis(
        &analyzer,
        "갔거든",
        &[("가", Pos::VV), ("았", Pos::EP), ("거든", Pos::EF)],
    );
    let result = analyzer.analyze("왔거든");
    assert!(
        result.tokens.iter().any(|t| t.text == "오" && matches!(t.pos, Pos::VV | Pos::VX)),
        "왔거든 must contain 오 stem: {:?}",
        result.tokens.iter().map(|t| (t.text.as_str(), t.pos)).collect::<Vec<_>>()
    );
    assert!(result.tokens.iter().any(|t| t.text == "거든" && t.pos == Pos::EF));
}

#[test]
fn test_issue1_han_standalone_mm() {
    let analyzer = load_analyzer();

    // 노래 한 곡 틀어줘 → 한/MM (not 하/XSV+ㄴ/ETM cross-eojeol)
    let result = analyzer.analyze("노래 한 곡 틀어줘");
    let han_token = result.tokens.iter().find(|t| t.text == "한")
        .expect("expected 한 token");
    assert_eq!(han_token.pos, Pos::MM);

    // 한 명 / 한 대
    let result = analyzer.analyze("한 명");
    assert_eq!(result.tokens[0].text, "한");
    assert_eq!(result.tokens[0].pos, Pos::MM);
    let result = analyzer.analyze("한 대");
    assert_eq!(result.tokens[0].text, "한");
    assert_eq!(result.tokens[0].pos, Pos::MM);

    // Regression: multi-syllable XX한 must keep XSV+ETM
    let result = analyzer.analyze("공부한 사람");
    assert!(result.tokens.iter().any(|t| t.text == "하" && t.pos == Pos::XSV));
    assert!(result.tokens.iter().any(|t| t.text == "ㄴ" && t.pos == Pos::ETM));
}

#[test]
fn test_issue1_myeoch_si_ya() {
    let analyzer = load_analyzer();

    // 몇 + 시야 (sentence-final) → 시/NNB + 이/VCP + 야/EF
    let result = analyzer.analyze("지금 몇 시야");
    let last3: Vec<(&str, Pos)> = result.tokens.iter().rev().take(3)
        .map(|t| (t.text.as_str(), t.pos)).collect();
    assert_eq!(last3[0], ("야", Pos::EF));
    assert_eq!(last3[1], ("이", Pos::VCP));
    assert_eq!(last3[2], ("시", Pos::NNB));

    // With SF
    let result = analyzer.analyze("몇 시야?");
    assert!(result.tokens.iter().any(|t| t.text == "시" && t.pos == Pos::NNB));
    assert!(result.tokens.iter().any(|t| t.pos == Pos::VCP));

    // Regression: 시야 (sight) must be preserved when not after 몇
    let result = analyzer.analyze("내 시야가 좁다");
    assert!(result.tokens.iter().any(|t| t.text == "시야" && t.pos == Pos::NNG));
}

#[test]
fn test_issue1_sn_unit_copula() {
    let analyzer = load_analyzer();

    // 24.7도예요 → 24.7/SN + 도/NNB + 이/VCP + 예요/EF (도예/NNG span 거부)
    let result = analyzer.analyze("24.7도예요");
    let texts: Vec<&str> = result.tokens.iter().map(|t| t.text.as_str()).collect();
    assert!(
        !texts.contains(&"도예"),
        "24.7도예요 must not contain 도예/NNG span: {:?}",
        result.tokens.iter().map(|t| (t.text.as_str(), t.pos)).collect::<Vec<_>>()
    );
    assert!(
        result.tokens.iter().any(|t| t.text == "도" && t.pos == Pos::NNB),
        "24.7도예요 must contain 도/NNB"
    );
    assert!(
        result.tokens.iter().any(|t| t.pos == Pos::VCP),
        "24.7도예요 must contain a VCP morpheme"
    );

    // 3개예요 (regression: must keep working)
    assert_analysis(
        &analyzer,
        "3개예요",
        &[("3", Pos::SN), ("개", Pos::NNB), ("이", Pos::VCP), ("예요", Pos::EF)],
    );
}

#[test]
fn test_issue2_lge_monosyllable_ha() {
    // 할게 → 하/VV + ㄹ게/EF (의존명사 분해 거부)
    let analyzer = load_analyzer();
    let result = analyzer.analyze("할게");
    let texts: Vec<&str> = result.tokens.iter().map(|t| t.text.as_str()).collect();
    assert!(
        !texts.contains(&"거"),
        "할게 must not decompose to 거/NNB: {:?}",
        result.tokens.iter().map(|t| (t.text.as_str(), t.pos)).collect::<Vec<_>>()
    );
    assert!(
        result.tokens.iter().any(|t| t.text == "ㄹ게" && t.pos == Pos::EF),
        "할게 must end with ㄹ게/EF"
    );

    // 내가 할게 (sentence-level)
    let result = analyzer.analyze("내가 할게");
    assert!(
        result.tokens.iter().any(|t| t.text == "ㄹ게" && t.pos == Pos::EF),
        "내가 할게 must end with ㄹ게/EF"
    );
}

#[test]
fn test_issue2_lge_monosyllable_l_jongseong() {
    // 살게/갈게/들게 → X/VV + ㄹ게/EF (EC 거부, ㄹ받침 단음절 어간)
    let analyzer = load_analyzer();
    for input in ["내가 살게", "내가 갈게", "내가 들게"] {
        let result = analyzer.analyze(input);
        let pairs: Vec<(&str, Pos)> = result.tokens.iter()
            .map(|t| (t.text.as_str(), t.pos)).collect();
        assert!(
            result.tokens.iter().any(|t| t.text == "ㄹ게" && t.pos == Pos::EF),
            "{} must end with ㄹ게/EF, got: {:?}", input, pairs
        );
    }
}

#[test]
fn test_issue2_lge_monosyllable_eul() {
    // 먹을게 → 먹/VV + 을게/EF (의존명사 분해 거부)
    let analyzer = load_analyzer();
    let result = analyzer.analyze("먹을게");
    let pairs: Vec<(&str, Pos)> = result.tokens.iter()
        .map(|t| (t.text.as_str(), t.pos)).collect();
    assert!(
        !pairs.iter().any(|(t, _)| *t == "거"),
        "먹을게 must not decompose to 거/NNB: {:?}", pairs
    );
    assert!(
        result.tokens.iter().any(|t| t.text == "을게" && t.pos == Pos::EF),
        "먹을게 must end with 을게/EF"
    );

    // Regression: existing 받을게/씻을게 still work
    for input in ["받을게", "씻을게", "앉을게"] {
        let result = analyzer.analyze(input);
        assert!(
            result.tokens.iter().any(|t| t.text == "을게" && t.pos == Pos::EF),
            "{} must end with 을게/EF", input
        );
    }
}

#[test]
fn test_issue2_iri_mag_demotion() {
    let analyzer = load_analyzer();

    // 이리 단독 → MAG (NNP 거부)
    let result = analyzer.analyze("이리");
    assert!(
        result.tokens.iter().any(|t| t.text == "이리" && t.pos == Pos::MAG),
        "이리 단독은 MAG여야 함: {:?}",
        result.tokens.iter().map(|t| (t.text.as_str(), t.pos)).collect::<Vec<_>>()
    );

    // 이리 와 → 이리/MAG + 오/VV + 아/EF (와/JKB 거부)
    let result = analyzer.analyze("이리 와");
    let pairs: Vec<(&str, Pos)> = result.tokens.iter()
        .map(|t| (t.text.as_str(), t.pos)).collect();
    assert!(
        pairs.iter().any(|(t, p)| *t == "이리" && *p == Pos::MAG),
        "이리 와 → 이리/MAG: {:?}", pairs
    );
    assert!(
        !pairs.iter().any(|(t, p)| *t == "와" && *p == Pos::JKB),
        "이리 와 must not have 와/JKB: {:?}", pairs
    );

    // 이리 와봐 → 이리/MAG ...
    let result = analyzer.analyze("이리 와봐");
    assert!(
        result.tokens.iter().any(|t| t.text == "이리" && t.pos == Pos::MAG),
        "이리 와봐 → 이리/MAG"
    );
}

#[test]
fn test_issue2_oneora_recovery() {
    let analyzer = load_analyzer();

    // 오너라 단독 → 오/VV + 너라/EF (오너/NNG span 거부)
    let result = analyzer.analyze("오너라");
    let pairs: Vec<(&str, Pos)> = result.tokens.iter()
        .map(|t| (t.text.as_str(), t.pos)).collect();
    assert!(
        !pairs.iter().any(|(t, _)| *t == "오너"),
        "오너라 must not contain 오너 span: {:?}", pairs
    );
    assert!(
        result.tokens.iter().any(|t| t.text == "너라" && t.pos == Pos::EF),
        "오너라 must end with 너라/EF"
    );

    // 이리 오너라 → 이리/MAG + 오/VV + 너라/EF
    let result = analyzer.analyze("이리 오너라");
    let pairs: Vec<(&str, Pos)> = result.tokens.iter()
        .map(|t| (t.text.as_str(), t.pos)).collect();
    assert_eq!(
        pairs,
        vec![("이리", Pos::MAG), ("오", Pos::VV), ("너라", Pos::EF)],
        "이리 오너라 분석 불일치"
    );

    // Regression: 회사 오너 (proper noun usage) must be preserved
    let result = analyzer.analyze("회사 오너가 와");
    assert!(
        result.tokens.iter().any(|t| t.text == "오너" && t.pos == Pos::NNG),
        "회사 오너가 와: 오너/NNG must be preserved"
    );
}
