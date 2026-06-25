//! BitNet b1.58 2B4T greedy generation CLI.
//!
//! Usage:
//!   cargo run --release -- "The capital of France is" 20
//!
//! Defaults to the model in ./bitnet-b1.58-2B-4T/. This is a faithful port of
//! the verified numpy reference (reference/), which generates
//!   "The capital of France is" -> " Paris. Paris is a city".

use std::env;
use std::time::Instant;
use tokenizers::Tokenizer;

use bitnet_int8_rs::model::Model;

fn main() {
    let model_path = env::var("BITNET_MODEL")
        .unwrap_or_else(|_| "bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf".to_string());
    let tok_path = env::var("BITNET_TOKENIZER")
        .unwrap_or_else(|_| "bitnet-b1.58-2B-4T/tokenizer.json".to_string());

    let args: Vec<String> = env::args().collect();
    let prompt = args.get(1).cloned().unwrap_or_else(|| "The capital of France is".to_string());
    let n_new: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    eprintln!("loading tokenizer {tok_path}");
    let tok = Tokenizer::from_file(&tok_path).expect("load tokenizer.json");
    eprintln!("loading model {model_path}");
    let t0 = Instant::now();
    let mut model = Model::load(&model_path).expect("load model");
    eprintln!("loaded in {:.1?}", t0.elapsed());

    // prepend BOS, like the reference
    let enc = tok.encode(prompt.as_str(), false).expect("encode");
    let mut ids: Vec<u32> = vec![model.cfg.bos];
    ids.extend(enc.get_ids().iter().copied());

    print!("{prompt}");
    let t0 = Instant::now();
    let mut logits = vec![];
    for &t in &ids {
        logits = model.forward(t as usize);
    }
    let mut produced = 0usize;
    for _ in 0..n_new {
        let next = argmax(&logits) as u32;
        if next == model.cfg.eos { break; }
        let piece = tok.decode(&[next], false).unwrap_or_default();
        print!("{piece}");
        use std::io::Write;
        std::io::stdout().flush().ok();
        produced += 1;
        logits = model.forward(next as usize);
    }
    let dt = t0.elapsed();
    eprintln!(
        "\n[{} tokens in {:.1?}, {:.2} tok/s]",
        produced, dt, produced as f64 / dt.as_secs_f64()
    );
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv { bv = x; bi = i; }
    }
    bi
}
