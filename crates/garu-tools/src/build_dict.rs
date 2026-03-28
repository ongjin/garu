//! Build an FST dictionary section for GMDL models.
//!
//! Supports dual-POS encoding: packs up to 2 POS per word into u64.
//!
//! Input format (tab-separated, sorted by word):
//!   word\tpos_byte\tfreq
//!   (duplicate words allowed — first 2 POS kept)
//!
//! FST value encoding:
//!   bits 0-7:   primary pos_byte
//!   bits 8-23:  primary quantized_freq (u16)
//!   bits 24-31: secondary pos_byte (0xFF = none)
//!   bits 32-47: secondary quantized_freq (u16)

use std::collections::BTreeMap;
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

    let file = fs::File::open(input_path).expect("Cannot open input file");
    let reader = io::BufReader::new(file);

    // Collect ALL entries per word (may have multiple POS)
    let mut word_entries: BTreeMap<String, Vec<(u8, u64)>> = BTreeMap::new();
    let mut max_freq: u64 = 0;

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let line = line.trim().to_string();
        if line.is_empty() { continue; }

        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        let word = parts[0].to_string();
        let pos: u8 = if parts.len() > 1 { parts[1].parse().unwrap_or(1) } else { 1 };
        let freq: u64 = if parts.len() > 2 { parts[2].parse().unwrap_or(1) } else { 1 };
        if freq > max_freq { max_freq = freq; }
        word_entries.entry(word).or_default().push((pos, freq));
    }

    if max_freq == 0 { max_freq = 1; }

    let mut dual_count = 0;
    let total = word_entries.len();

    // Build FST entries with dual-POS encoding
    let mut fst_entries: Vec<(String, u64)> = Vec::with_capacity(total);
    for (word, mut entries) in word_entries {
        // Sort by freq descending, keep top 2
        entries.sort_by(|a, b| b.1.cmp(&a.1));

        let (pos1, freq1) = entries[0];
        let qfreq1 = ((freq1 as f64 / max_freq as f64) * 65535.0).round().max(1.0).min(65535.0) as u64;

        let mut value = (pos1 as u64) | (qfreq1 << 8);

        // Secondary POS (if different from primary and significant)
        if entries.len() >= 2 && entries[1].0 != pos1 {
            let (pos2, freq2) = entries[1];
            let qfreq2 = ((freq2 as f64 / max_freq as f64) * 65535.0).round().max(1.0).min(65535.0) as u64;
            value |= (pos2 as u64) << 24;
            value |= qfreq2 << 32;
            dual_count += 1;
        } else {
            value |= 0xFF_u64 << 24; // sentinel: no secondary
        }

        fst_entries.push((word, value));
    }

    eprintln!("Building FST from {} entries (max_freq={})...", total, max_freq);
    eprintln!("  Dual-POS entries: {} ({:.1}%)", dual_count, dual_count as f64 / total as f64 * 100.0);

    // Build FST map
    let mut builder = fst::MapBuilder::memory();
    for (word, value) in &fst_entries {
        builder.insert(word.as_bytes(), *value).expect("FST insert failed");
    }
    let fst_bytes = builder.into_inner().expect("FST build failed");

    eprintln!("FST size: {} KB", fst_bytes.len() / 1024);

    // Build Dict v2 binary
    let mut output = Vec::new();
    output.extend_from_slice(b"GARU");
    output.extend_from_slice(&2u32.to_le_bytes());
    output.extend_from_slice(&(fst_bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(&fst_bytes);

    fs::write(output_path, &output).expect("Cannot write output file");
    eprintln!("Output: {} ({} KB)", output_path, output.len() / 1024);
}
