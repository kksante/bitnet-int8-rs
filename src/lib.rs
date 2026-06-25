//! BitNet b1.58 2B4T inference in pure Rust (W1.58A8).
//!
//! This crate is a faithful port of the numpy reference in `reference/`, which
//! was verified to generate coherent text from the real Microsoft weights
//! (`bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf`). It supersedes the earlier
//! heuristic "scale-tracking" engine, which had several correctness bugs
//! (wrong I2_S unpacking, ignored embedded per-tensor scale, 24 vs 30 layers,
//! no GQA). See `CLAUDE.md` and `reference/FINDINGS.md`.

pub mod gguf;
pub mod model;

pub use gguf::Gguf;
pub use model::{Config, Model};
