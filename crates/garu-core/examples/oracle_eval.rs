use garu_core::model::Analyzer;
use garu_core::types::Token;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};

#[derive(Default, Clone)]
struct Counts {
    tp: usize,
    fp: usize,
    fn_: usize,
    sentences: usize,
    exact: usize,
    oracle_exact: usize,
    improved: usize,
    no_exact_candidate: usize,
}

#[derive(Clone, Copy)]
struct Prf {
    tp: usize,
    fp: usize,
    fn_: usize,
    f1: f32,
}

fn parse_gold(value: &Value) -> Vec<(String, String)> {
    value["morphemes"]
        .as_array()
        .expect("morphemes must be an array")
        .iter()
        .map(|item| {
            let pair = item.as_array().expect("morpheme must be [form, pos]");
            (
                pair[0].as_str().expect("form must be a string").to_string(),
                pair[1].as_str().expect("pos must be a string").to_string(),
            )
        })
        .collect()
}

fn token_pairs(tokens: &[Token]) -> Vec<(String, String)> {
    tokens
        .iter()
        .map(|t| (t.text.clone(), t.pos.as_str().to_string()))
        .collect()
}

fn multiset(tokens: &[(String, String)]) -> HashMap<(String, String), usize> {
    let mut map = HashMap::new();
    for token in tokens {
        *map.entry(token.clone()).or_insert(0) += 1;
    }
    map
}

fn score(pred: &[(String, String)], gold: &[(String, String)]) -> Prf {
    let pred_counts = multiset(pred);
    let gold_counts = multiset(gold);
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;

    let keys: HashSet<_> = pred_counts.keys().chain(gold_counts.keys()).collect();
    for key in keys {
        let p = *pred_counts.get(key).unwrap_or(&0);
        let g = *gold_counts.get(key).unwrap_or(&0);
        let m = p.min(g);
        tp += m;
        fp += p - m;
        fn_ += g - m;
    }

    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_) as f32
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Prf { tp, fp, fn_, f1 }
}

fn exact(pred: &[(String, String)], gold: &[(String, String)]) -> bool {
    pred == gold
}

fn add_score(counts: &mut Counts, prf: Prf) {
    counts.tp += prf.tp;
    counts.fp += prf.fp;
    counts.fn_ += prf.fn_;
}

fn f1(counts: &Counts) -> (f32, f32, f32) {
    let precision = if counts.tp + counts.fp == 0 {
        0.0
    } else {
        counts.tp as f32 / (counts.tp + counts.fp) as f32
    };
    let recall = if counts.tp + counts.fn_ == 0 {
        0.0
    } else {
        counts.tp as f32 / (counts.tp + counts.fn_) as f32
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    (precision, recall, f1)
}

fn print_counts(label: &str, counts: &Counts) {
    let (p, r, f) = f1(counts);
    println!(
        "{:<12} sent={:<5} P={:.4} R={:.4} F1={:.4} exact={}/{} oracle_exact={} improved={} no_exact_candidate={}",
        label,
        counts.sentences,
        p,
        r,
        f,
        counts.exact,
        counts.sentences,
        counts.oracle_exact,
        counts.improved,
        counts.no_exact_candidate
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: oracle_eval <gold.jsonl> [top_k] [limit]");
        std::process::exit(2);
    }
    let gold_path = &args[1];
    let top_k = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);
    let limit = args.get(3).and_then(|s| s.parse::<usize>().ok());
    let dump_candidates = std::env::var("GARU_ORACLE_DUMP").is_ok();

    let model_path =
        std::env::var("GARU_MODEL").unwrap_or_else(|_| "models/codebook.gmdl".to_string());
    let cnn_path = std::env::var("GARU_CNN").unwrap_or_else(|_| "models/cnn2.bin".to_string());
    let model_data = fs::read(&model_path).expect("failed to read model");
    let cnn_data = fs::read(&cnn_path).expect("failed to read CNN model");
    let analyzer = Analyzer::from_bytes(&model_data, &cnn_data).expect("failed to load analyzer");

    let file = fs::File::open(gold_path).expect("failed to open gold jsonl");
    let reader = BufReader::new(file);

    let mut final_counts = Counts::default();
    let mut oracle_counts = Counts::default();
    let mut domains: HashMap<String, (Counts, Counts)> = HashMap::new();

    for (idx, line) in reader.lines().enumerate() {
        if let Some(limit) = limit {
            if idx >= limit {
                break;
            }
        }
        let line = line.expect("failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).expect("invalid json line");
        let text = value["text"].as_str().expect("text must be a string");
        let domain = value["domain"].as_str().unwrap_or("unknown").to_string();
        let gold = parse_gold(&value);

        let final_result = analyzer.analyze(text);
        let final_pairs = token_pairs(&final_result.tokens);
        let final_score = score(&final_pairs, &gold);
        let final_exact = exact(&final_pairs, &gold);

        let candidates = analyzer.analyze_topn(text, top_k);
        let mut best_score = final_score;
        let mut oracle_exact = false;
        if dump_candidates {
            println!();
            println!("TEXT: {}", text);
            println!("GOLD: {:?}", gold);
            println!("FINAL: {:?} f1={:.4}", final_pairs, final_score.f1);
        }
        for (rank, candidate) in candidates.into_iter().enumerate() {
            let pairs = token_pairs(&candidate.tokens);
            let candidate_score = score(&pairs, &gold);
            if dump_candidates {
                println!(
                    "#{:<2} score={:<8.3} f1={:.4} {:?}",
                    rank + 1,
                    candidate.score,
                    candidate_score.f1,
                    pairs
                );
            }
            if exact(&pairs, &gold) {
                oracle_exact = true;
            }
            if candidate_score.f1 > best_score.f1 {
                best_score = candidate_score;
            }
        }

        final_counts.sentences += 1;
        oracle_counts.sentences += 1;
        add_score(&mut final_counts, final_score);
        add_score(&mut oracle_counts, best_score);
        if final_exact {
            final_counts.exact += 1;
            oracle_counts.exact += 1;
        }
        if oracle_exact {
            final_counts.oracle_exact += 1;
            oracle_counts.oracle_exact += 1;
        }
        if best_score.f1 > final_score.f1 {
            final_counts.improved += 1;
            oracle_counts.improved += 1;
        }
        if !final_exact && !oracle_exact {
            final_counts.no_exact_candidate += 1;
            oracle_counts.no_exact_candidate += 1;
        }

        let (domain_final, domain_oracle) = domains.entry(domain).or_default();
        domain_final.sentences += 1;
        domain_oracle.sentences += 1;
        add_score(domain_final, final_score);
        add_score(domain_oracle, best_score);
        if final_exact {
            domain_final.exact += 1;
            domain_oracle.exact += 1;
        }
        if oracle_exact {
            domain_final.oracle_exact += 1;
            domain_oracle.oracle_exact += 1;
        }
        if best_score.f1 > final_score.f1 {
            domain_final.improved += 1;
            domain_oracle.improved += 1;
        }
        if !final_exact && !oracle_exact {
            domain_final.no_exact_candidate += 1;
            domain_oracle.no_exact_candidate += 1;
        }
    }

    println!("Top-K oracle evaluation: k={}", top_k);
    print_counts("final", &final_counts);
    print_counts("oracle", &oracle_counts);

    let mut domain_names: Vec<_> = domains.keys().cloned().collect();
    domain_names.sort();
    println!();
    println!("By domain:");
    for domain in domain_names {
        let (domain_final, domain_oracle) = domains.get(&domain).unwrap();
        let (_, _, final_f1) = f1(domain_final);
        let (_, _, oracle_f1) = f1(domain_oracle);
        println!(
            "{:<12} final={:.4} oracle={:.4} gain={:+.4} exact={}/{} oracle_exact={} improved={} no_exact_candidate={}",
            domain,
            final_f1,
            oracle_f1,
            oracle_f1 - final_f1,
            domain_final.exact,
            domain_final.sentences,
            domain_final.oracle_exact,
            domain_final.improved,
            domain_final.no_exact_candidate
        );
    }
}
