//! Build an FST dictionary section for GMDL models.
//!
//! Reads a sorted word list (one word per line, optionally with POS byte)
//! and outputs the Dict v2 binary (GARU magic + version 2 + FST bytes).
//!
//! Usage:
//!   build-dict <input.txt> <output.bin>
//!
//! Input format (tab-separated, sorted by word):
//!   word\tpos_byte
//!
//! If no pos_byte, defaults to 1 (NNP).

use std::fs;
use std::io::{self, BufRead, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: build-dict <input.txt> <output.bin>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    // Read and parse input
    let file = fs::File::open(input_path).expect("Cannot open input file");
    let reader = io::BufReader::new(file);

    let mut entries: Vec<(String, u64)> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let (word, pos_byte) = if let Some(idx) = line.find('\t') {
            let word = &line[..idx];
            let pos: u64 = line[idx + 1..].parse().unwrap_or(1);
            (word.to_string(), pos)
        } else {
            (line.to_string(), 1) // default NNP
        };

        entries.push((word, pos_byte));
    }

    // Sort by word (FST requires sorted input)
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    // Deduplicate
    entries.dedup_by(|a, b| a.0 == b.0);

    eprintln!("Building FST from {} entries...", entries.len());

    // Build FST map
    let mut builder = fst::MapBuilder::memory();
    for (word, pos_byte) in &entries {
        builder.insert(word.as_bytes(), *pos_byte).expect("FST insert failed");
    }
    let fst_bytes = builder.into_inner().expect("FST build failed");

    eprintln!("FST size: {} KB", fst_bytes.len() / 1024);

    // Build Dict v2 binary: "GARU" + version(2) + fst_len + fst_bytes
    let mut output = Vec::new();
    output.extend_from_slice(b"GARU");
    output.extend_from_slice(&2u32.to_le_bytes()); // version 2
    output.extend_from_slice(&(fst_bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(&fst_bytes);

    fs::write(output_path, &output).expect("Cannot write output file");
    eprintln!(
        "Output: {} ({} KB)",
        output_path,
        output.len() / 1024
    );
}
