# CLAUDE.md

Guidance for working in this repository.

## Project Overview

BitNet b1.58 2B4T (Microsoft's native 1.58-bit LLM, W1.58A8) inference in pure
Rust, plus a verified numpy reference used as the correctness oracle and as a
research harness for characterizing **fully-integer** inference.

Two things live here:

1. `src/` — a clean Rust implementation (faithful float path: ternary weights +
   per-token int8 activation quant + integer-exact matmul).
2. `reference/` — a numpy reference (float oracle **and** a fully-integer path)
   that was used to (a) reverse-engineer the I2_S weight format, (b) verify
   coherent generation, and (c) characterize where the integer path holds/breaks.
   See `reference/FINDINGS.md`.

> **History:** an earlier version used a heuristic "scale-tracking" integer
> engine that never produced coherent text. It had real bugs: wrong I2_S
> unpacking, a discarded embedded scale, 24 (vs 30) layers, and no GQA. That
> approach was replaced after the format and forward were verified end-to-end.

## Verified model facts (BitNet b1.58 2B4T)

- 30 layers, d_model 2560, FFN 6912, **GQA: 20 query / 5 KV heads**, head_dim 128.
- RoPE base 500000 (HF split-half / `rotate_half`, **no** weight permute).
- Squared-ReLU FFN: `down( sub_norm( relu(gate(x))^2 * up(x) ) )`.
- subln: `attn_sub_norm` before `attn_output`, `ffn_sub_norm` before `ffn_down`.
- Tied embeddings (token_embd, F16) for the LM head; no separate output weight.
- rms_eps 1e-5, vocab 128256, context 4096. LLaMA-3 tokenizer.
- Per-token **absmax** int8 activation quant; per-tensor **absmean** ternary scale.

### I2_S weight format (the non-obvious part)

Each I2_S tensor on disk is:

```
[ ceil(ne/4) bytes of 2-bit codes ] [ one little-endian f32 scale ] [ pad to 32B ]
```

Codes are **128-element block interleaved**, not sequential. For byte `j` in
0..32 and 2-bit slot `s` in 0..4 (LSB-first), the element index within the
32-byte / 128-element block is `(3 - s) * 32 + j`. Decoded value = `code - 1`
(`0->-1, 1->0, 2->+1`). Dequantized weight = `(code - 1) * scale`.
ggml tensor dims are `[in_features, out_features]`.

Getting this wrong (sequential packing, or the non-reversed group order)
reproduces the correct ternary *distribution* but scrambles positions →
real-but-incoherent tokens. The reference loader is the source of truth
(`reference/gguf_loader.py::unpack_i2s`, mirrored in `src/gguf.rs::unpack_i2s`).

## Build & run (Rust)

```bash
cargo build --release
cargo run --release -- "The capital of France is" 20
# -> The capital of France is Paris. Paris is a city ...
```

Model files expected in `bitnet-b1.58-2B-4T/` (`ggml-model-i2_s.gguf`,
`tokenizer.json`). Override with `BITNET_MODEL` / `BITNET_TOKENIZER`.

The Rust port mirrors the reference op-for-op; the I2_S unpacking is checked
bit-for-bit against the reference. (The sandbox this was built in had no Rust
toolchain, so compile/run verification is done on your machine.)

## Reference / research harness (numpy)

```bash
cd reference
pip install numpy tokenizers
python3 gguf_loader.py            # validate I2_S load (distributions, scales)
python3 prepare_weights.py        # one-time unpack -> int8 memmap (~2GB, gitignored)
python3 _oracle.py                # cache the float-oracle reference generation
python3 run_configs.py 0 1 2 3    # integer-path sweep (see FINDINGS.md)
```

Key files: `gguf_loader.py` (I2_S), `bitnet_float.py` (simple float oracle),
`bitnet.py` (`FloatRunner` + `IntRunner` with precision knobs), `characterize.py`
/ `run_configs.py` / `run_mult.py` (the sweep), `FINDINGS.md` (the write-up).

## Code map (Rust)

```
src/
├── lib.rs      # crate root: gguf + model
├── gguf.rs     # GGUF parser + I2_S unpack (block interleave) + per-tensor scale
├── model.rs    # Config, BitLinear, Layer, Model::forward (GQA, RoPE, relu^2, KV cache)
└── main.rs     # greedy generation CLI (uses the `tokenizers` crate)
```

Legacy modules under `src/model/`, `src/weights/`, `src/bin/`, etc. are
superseded and not part of the build (`autobins = false`, not referenced by
`lib.rs`). They can be deleted.

## Headline findings (see reference/FINDINGS.md)

- A fully-integer forward (int8 activations, int8 KV, integer norm/RoPE/softmax)
  is **token-exact** vs the float oracle (final-layer cosine 0.9997).
- Integer softmax is cheap: a **16-entry** exp LUT still generates identical text.
- KV-cache: by held-out corpus perplexity (WikiText-2, 10k tok, ref 27.4), the
  limiter is quantizer **granularity + zero-point**, not bit-width. int8 KV is
  free; 2-bit per-vector **and** per-channel-symmetric both collapse (+264–266);
  **only per-channel _asymmetric_ rescues 2-bit (+10)**. (On a small mini-corpus
  per-channel-symmetric alone looks sufficient; that reverses on WikiText, so the
  robust claim is the granularity+zero-point conjunction.) The single-prompt
  "3-bit collapse" was greedy-decoding brittleness.
- Generalizes across architectures: the same surface holds on Falcon3-1B-1.58
  (int8 free, only per-channel-asym survives 2-bit, robust scaling catastrophic;
  Key per-channel bias 0.79σ in both models).
- **absmax activation scaling is load-bearing**: mean/median (outlier-robust)
  scaling raises perplexity 3–6 orders of magnitude, because activations are
  heavy-tailed (absmax/absmean median 7.8, p90 ≈ 106) and the outliers carry the
  signal. This is the single most load-bearing thing in the integer path.
