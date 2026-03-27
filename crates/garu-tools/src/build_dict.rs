//! Build an FST dictionary section for GMDL models.
//!
//! Reads a sorted word list (one word per line, with POS byte and optional freq)
//! and outputs the Dict v2 binary (GARU magic + version 2 + FST bytes).
//!
//! Usage:
//!   build-dict <input.txt> <output.bin>
//!
//! Input format (tab-separated, sorted by word):
//!   word\tpos_byte\tfreq
//!
//! If no pos_byte, defaults to 1 (NNP). If no freq, defaults to 1.
//! FST value encoding: pos_byte | (quantized_freq << 8)

use std::fs;
use std::io::{self, BufRead};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: build-dict <input.txt> <output.bin>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    // Read and parse input (two passes: collect, then encode with max_freq)
    let file = fs::File::open(input_path).expect("Cannot open input file");
    let reader = io::BufReader::new(file);

    let mut raw_entries: Vec<(String, u8, u64)> = Vec::new();
    let mut max_freq: u64 = 0;

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        let word = parts[0].to_string();
        let pos: u8 = if parts.len() > 1 { parts[1].parse().unwrap_or(1) } else { 1 };
        let freq: u64 = if parts.len() > 2 { parts[2].parse().unwrap_or(1) } else { 1 };
        if freq > max_freq {
            max_freq = freq;
        }
        raw_entries.push((word, pos, freq));
    }

    if max_freq == 0 {
        max_freq = 1;
    }

    // Sort by word bytes (FST requires sorted input)
    raw_entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    // Deduplicate by word
    raw_entries.dedup_by(|a, b| a.0 == b.0);

    eprintln!("Building FST from {} entries (max_freq={})...", raw_entries.len(), max_freq);

    // Encode: value = pos_byte | (quantized_freq << 8)
    let mut entries: Vec<(String, u64)> = Vec::new();
    for (word, pos, freq) in &raw_entries {
        let qfreq = ((*freq as f64 / max_freq as f64) * 65535.0).round() as u64;
        let qfreq = qfreq.min(65535);
        let value = (*pos as u64) | (qfreq << 8);
        entries.push((word.clone(), value));
    }

    // Build FST map
    let mut builder = fst::MapBuilder::memory();
    for (word, value) in &entries {
        builder.insert(word.as_bytes(), *value).expect("FST insert failed");
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
