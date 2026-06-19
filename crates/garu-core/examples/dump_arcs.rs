//! Dump raw lattice arcs (before cache injection / Viterbi) sorted by cost.
//! Debug-only: trace which arcs exist for a given text and their origin cost.
use garu_core::codebook::CodebookAnalyzer;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let text = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let model_path = env::var("GARU_MODEL").unwrap_or_else(|_| "models/codebook.gmdl".to_string());
    let model_data = fs::read(&model_path).expect("failed to read model");
    let cb = CodebookAnalyzer::from_bytes(&model_data).expect("failed to load");

    let mut arcs = cb.dump_arcs(text);
    // sort by (start, end, cost)
    arcs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.3.partial_cmp(&b.3).unwrap()));
    for (start, end, morphemes, cost) in arcs {
        let m: Vec<String> = morphemes.iter().map(|(f, p)| format!("{}/{}", f, p.as_str())).collect();
        println!("[{:>2}..{:<2}] cost={:>7.3}  {}", start, end, cost, m.join(" "));
    }
}
