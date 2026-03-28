
use garu_core::model::Analyzer;
use std::fs;
use std::io::{self, BufRead, Write};

fn main() {
    let model_path = std::env::var("GARU_MODEL").unwrap_or_else(|_| "models/codebook.gmdl".to_string());
    let model_data = fs::read(&model_path).expect("Failed to read model");
    let analyzer = Analyzer::from_bytes(&model_data)
        .expect("Failed to load model");

    let input_path = std::env::args().nth(1).expect("Need input file path");
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
        let tokens: Vec<String> = result.tokens.iter()
            .map(|t| format!("{}\t{}", t.text, t.pos.as_str()))
            .collect();
        writeln!(out, "{}", tokens.join("\n")).unwrap();
        writeln!(out, "---").unwrap();
    }
}
