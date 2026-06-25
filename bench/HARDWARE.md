# Hardware measurements (run on your machine)

The paper's `tab:hw` numbers were measured in an aarch64 Linux sandbox. To get the
headline number on **your Apple-silicon Mac** (the better "edge-class" datapoint),
run one command and paste the output into `tab:hw` in `paper/main.tex`.

## 1. Integer-kernel + memory microbenchmark (no deps, ~5 s)

```bash
cc -O3 -o bench/kernels bench/kernels.c && ./bench/kernels
```

(`cc` is Apple clang on macOS; `-O3` turns on NEON auto-vectorization. Optionally add
`-mcpu=apple-m1` / `-mcpu=native`.) It prints, measured on your CPU:

- sustained memory read bandwidth (GB/s) — the operative bound for memory-bound decode;
- the 2-bit-packed weight size and the weight-stream latency floor (ms/token);
- KV-cache footprint and per-token KV memory traffic at int8 vs 2-bit (the 4× win).

Drop those into Table `tab:hw`. That is the one real silicon number the systems
framing needs; everything else in the paper is already done.

## 2. (Optional, stronger) End-to-end decode latency from the Rust port

```bash
cargo build --release
# decode throughput (the port already prints tok/s):
cargo run --release -- "The capital of France is" 100
# peak resident memory on macOS:
/usr/bin/time -l ./target/release/bitnet-int8-rs "The capital of France is" 100 \
  2>&1 | grep "maximum resident"
```

Report tok/s and peak RSS. This is the end-to-end wall-clock the Limitations section
flags as the remaining systems step; with it, the "edge" claim is fully demonstrated,
not inferred.

## 3. (Optional) bitnet.cpp cross-check

Build `microsoft/BitNet` (bitnet.cpp) and time its CPU decode on the same model; it
reports ~29 ms/token for BitNet 2B, which our `tab:hw` weight-stream floor (11 ms,
single-thread bandwidth-bound) is consistent with.

## What changes in the paper

Only the six numbers in `tab:hw` and the two sentences in §\ref{sec:hw} that quote
them. Everything else is corpus/perplexity work that's already complete. If your Mac
bandwidth differs (Apple silicon is typically 60–100+ GB/s vs the sandbox's 47.5),
the weight-stream floor and KV-traffic numbers scale inversely with it.
```
