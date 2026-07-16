//! Dump lattice arcs for many sentences as JSONL (track3 Phase 0 측정용).
//! usage: dump_arcs_batch <texts.txt>   (GARU_MODEL env로 모델 지정)
use garu_core::codebook::CodebookAnalyzer;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let input_path = args.get(1).expect("usage: dump_arcs_batch <texts.txt>");
    let model_path = env::var("GARU_MODEL").unwrap_or_else(|_| "models/codebook.gmdl".to_string());
    let model_data = fs::read(&model_path).expect("failed to read model");
    let cb = CodebookAnalyzer::from_bytes(&model_data).expect("failed to load");

    let file = fs::File::open(input_path).expect("failed to open input");
    let reader = io::BufReader::new(file);
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in reader.lines() {
        let text = line.expect("read line");
        let arcs = cb.dump_arcs(text.trim());
        let parts: Vec<String> = arcs
            .iter()
            .map(|(start, end, morphemes, cost)| {
                let ms: Vec<String> = morphemes
                    .iter()
                    .map(|(f, p)| format!("[\"{}\",\"{}\"]", esc(f), p.as_str()))
                    .collect();
                format!("[{},{},[{}],{}]", start, end, ms.join(","), cost)
            })
            .collect();
        writeln!(out, "[{}]", parts.join(",")).unwrap();
    }
}
