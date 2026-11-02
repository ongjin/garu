//! Section 18 탐색 비용 학습 (native 전체 경로 averaged perceptron).
//!
//!   GARU_MODEL=m.gmdl search_train gold ROWS.jsonl > GOLD.jsonl
//!       문장마다 정답 형태소열과 같은 아크 경로를 찾아 후보 후처리까지 적용한 토큰을 쓴다
//!       ({"i", "tokens"}; 경로가 없으면 tokens=null). 정답과 같은지는 Python(ep_norm)이 확인한다.
//!   GARU_MODEL=m.gmdl search_train train ROWS.jsonl VERIFIED.txt OUT_PREFIX EPOCHS SEED
//!       VERIFIED의 문장만 학습. 매 epoch 평균 가중치를 OUT_PREFIX.e<k>.tsv(bucket\tweight)로 쓴다.
//!
//! ROWS는 training/ambiguity/prepare.py 형식({text, morphemes, ...})이다.

use garu_core::model::Analyzer;
use garu_core::search::DIM_LOG2;
use garu_core::types::Pos;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};

fn load_rows(path: &str) -> Vec<(String, Vec<(String, Pos)>)> {
    let file = fs::File::open(path).expect("rows");
    io::BufReader::new(file).lines().map(|line| {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let text = row["text"].as_str().unwrap().trim().to_string();
        let gold = row["morphemes"].as_array().unwrap().iter().filter_map(|m| {
            let pos = Pos::from_str(m[1].as_str()?)?;
            Some((m[0].as_str()?.to_string(), pos))
        }).collect();
        (text, gold)
    }).collect()
}

fn gold_mode(analyzer: &Analyzer, rows: &str) {
    let book = analyzer.codebook();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for (i, (text, gold)) in load_rows(rows).iter().enumerate() {
        let lat = book.lab_lattice(text);
        let tokens = book.lab_gold(text, &lat, gold).map(|path| {
            let mut tokens = book.lab_process(text, &lat, &path);
            Analyzer::lab_rerank_rules(&mut tokens);
            tokens.iter().map(|t| (t.text.clone(), t.pos.as_str())).collect::<Vec<_>>()
        });
        serde_json::to_writer(&mut out, &serde_json::json!({"i": i, "tokens": tokens})).unwrap();
        writeln!(out).unwrap();
    }
}

/// xorshift64* — 학습 순서 셔플용.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn shuffle(&mut self, items: &mut [usize]) {
        for i in (1..items.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

fn train_mode(analyzer: &Analyzer, rows: &str, verified: &str, prefix: &str, epochs: usize, seed: u64) {
    let book = analyzer.codebook();
    let rows = load_rows(rows);
    let mut order: Vec<usize> = fs::read_to_string(verified).unwrap()
        .lines().map(|l| l.trim().parse().unwrap()).collect();
    let dim = 1usize << DIM_LOG2;
    let mut w = vec![0f32; dim];
    let mut wa = vec![0f64; dim];
    let mut step: f64 = 0.0;
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

    for epoch in 1..=epochs {
        rng.shuffle(&mut order);
        let mut updates = 0usize;
        for &i in &order {
            step += 1.0;
            let (text, gold) = &rows[i];
            let mut lat = book.lab_lattice(text);
            let Some(gold_path) = book.lab_gold(text, &lat, gold) else { continue; };
            let pred = book.lab_best(text, &mut lat, &w, 1.0);
            if pred == gold_path {
                continue;
            }
            updates += 1;
            for (path, sign) in [(&pred, 1.0f32), (&gold_path, -1.0f32)] {
                for &arc in path.iter() {
                    for &f in lat.features(arc) {
                        w[f as usize] += sign;
                        wa[f as usize] += step * sign as f64;
                    }
                }
            }
        }
        let path = format!("{prefix}.e{epoch}.tsv");
        let mut out = BufWriter::new(fs::File::create(&path).unwrap());
        let mut nnz = 0usize;
        for f in 0..dim {
            let avg = w[f] as f64 - wa[f] / step;
            if avg != 0.0 {
                writeln!(out, "{f}\t{avg}").unwrap();
                nnz += 1;
            }
        }
        eprintln!("epoch {epoch}: updates {updates}/{} nnz {nnz} -> {path}", order.len());
    }
}

fn main() {
    let model_path = std::env::var("GARU_MODEL").unwrap_or_else(|_| "models/codebook.gmdl".into());
    let analyzer = Analyzer::from_bytes(&fs::read(model_path).unwrap()).unwrap();
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gold") => gold_mode(&analyzer, &args[2]),
        Some("train") => train_mode(&analyzer, &args[2], &args[3], &args[4],
                                    args[5].parse().unwrap(), args[6].parse().unwrap()),
        _ => panic!("usage: search_train gold ROWS | train ROWS VERIFIED PREFIX EPOCHS SEED"),
    }
}
