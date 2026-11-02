
use garu_core::model::Analyzer;
use garu_core::types::Token;
use std::fs;
use std::io::{self, BufRead, Write};

fn write_json(out: &mut impl Write, tokens: &[Token]) {
    let pairs: Vec<(&str, &str)> = tokens.iter()
        .map(|t| (t.text.as_str(), t.pos.as_str()))
        .collect();
    serde_json::to_writer(&mut *out, &pairs).unwrap();
    writeln!(out).unwrap();
}

/// `--json-spans`: [형태, 품사, 시작, 끝] (trim된 입력 줄 기준 문자 위치)
fn write_json_spans(out: &mut impl Write, tokens: &[Token]) {
    let rows: Vec<(&str, &str, usize, usize)> = tokens.iter()
        .map(|t| (t.text.as_str(), t.pos.as_str(), t.start, t.end))
        .collect();
    serde_json::to_writer(&mut *out, &rows).unwrap();
    writeln!(out).unwrap();
}

fn main() {
    let model_path = std::env::var("GARU_MODEL").unwrap_or_else(|_| "models/codebook.gmdl".to_string());
    let model_data = fs::read(&model_path).expect("Failed to read model");

    let analyzer = Analyzer::from_bytes(&model_data)
        .expect("Failed to load model");

    let input_path = std::env::args().nth(1).expect("Need input file path");
    let output_format = std::env::args().nth(2).unwrap_or_default();
    let json_mode = output_format == "--json";
    let spans_mode = output_format == "--json-spans";

    let file = fs::File::open(&input_path).expect("Failed to open input");
    let reader = io::BufReader::new(file);

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let line = line.trim();
        if line.is_empty() {
            writeln!(out, "[]").unwrap();
            continue;
        }
        let result = analyzer.analyze(line);

        if spans_mode {
            write_json_spans(&mut out, &result.tokens);
        } else if json_mode {
            write_json(&mut out, &result.tokens);
        } else {
            let tokens: Vec<String> = result.tokens.iter()
                .map(|t| format!("{}\t{}", t.text, t.pos.as_str()))
                .collect();
            writeln!(out, "{}", tokens.join("\n")).unwrap();
            writeln!(out, "---").unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use garu_core::types::Pos;

    #[test]
    fn json_controls_roundtrip() {
        // Embedded newlines cannot be passed through the line-oriented CLI.
        let text = "한글\\\"\n\r\t\u{0000}\u{0008}\u{000c}\u{001f}";
        let tokens = vec![Token {
            text: text.to_string(),
            pos: Pos::SW,
            start: 0,
            end: text.chars().count(),
            score: None,
        }];
        let mut output = Vec::new();
        write_json(&mut output, &tokens);

        let decoded: Vec<(String, String)> = serde_json::from_slice(&output).unwrap();
        assert_eq!(decoded, vec![(text.to_string(), "SW".to_string())]);
        assert_eq!(output.iter().filter(|&&byte| byte == b'\n').count(), 1);
    }
}
