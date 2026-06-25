# Where does a fully-integer BitNet b1.58 inference path hold, and where does it break?

> **The formal write-up is the paper at [`../paper/main.tex`](../paper/main.tex)**
> (compiled `main.pdf`), which adds: a real WikiText-2 slice, bootstrap 95% CIs,
> the granularity×asymmetry KV ablation, the KV channel-bias mechanism, the
> activation recovery curve, and figures. This file is the informal lab notebook.
>
> Headline additions since the first draft: on WikiText-2, 2-bit **per-channel
> asymmetric** KV costs only +7.6 PPL vs +241 for per-vector (the rescue is the
> zero-point absorbing a measured per-channel Key bias of ~0.79σ, 2.5× the Value
> cache); robust activation scaling reaches 4.6×10⁵ PPL. All deltas have CIs.


A characterization of integer-path inference (int8 activations, ternary weights,
integer RMSNorm / RoPE / softmax / KV-cache) for **BitNet b1.58 2B4T**, measured
against a faithful floating-point oracle on the real Microsoft weights
(`ggml-model-i2_s.gguf`).

All numbers below are from `reference/` (pure numpy, no framework), verified by
the float oracle generating coherent text from the real model.

---

## 1. Setup and the honest baseline

**Model.** BitNet b1.58 2B4T: 30 layers, d=2560, FFN=6912, GQA with 20 query /
5 KV heads (head_dim 128), RoPE base 5e5, squared-ReLU FFN, subln normalization,
tied (F16) embeddings. Weights are native ternary {-1,0,+1} with a per-tensor
absmean scale; activations are 8-bit (W1.58A8). This is a *quantization-aware
trained* model — the int8 activation path is part of the model, not a post-hoc
approximation.

**Float oracle.** Dequantize ternary weights to float, apply the model's own
per-token int8 absmax activation quantization, and run everything else
(RMSNorm, RoPE, softmax, residual) in float64. This reproduces the intended
model: `"The capital of France is"` → `" Paris. Paris is a city"`. This is the
ground truth every integer configuration is compared against.

**Integer path.** Identical computation graph, but every operation is integer:

| op | integer realization |
|---|---|
| BitLinear | int8 activations × ternary weights → **exact** int32 accumulation |
| RMSNorm | data-adaptive fixed-point, integer sum-of-squares, integer `isqrt`, fixed-point reciprocal |
| RoPE | Q15 fixed-point sin/cos, integer rotate |
| attention scores | int8 query × int KV → int32 dot |
| softmax | fixed-point `exp` LUT + integer normalization |
| KV-cache | quantized to `kv_bits` per vector |

The ternary×int8 matmul is **lossless** (sums of int8 fit in int32), so it
contributes *zero* error. All integer-path error therefore comes from the
*non-matmul* ops — which is exactly the interesting question.

**Metrics.** `tokAgree` = fraction of greedily-decoded tokens matching the
oracle; `LfinCos` = cosine of the final-layer hidden state (last prompt token)
vs oracle; `logitCos` = cosine of the logit vectors. Generation agreement is the
headline pass/fail; the cosines are the rigorous diagnostic.

---

## 2. Result: the integer path is faithful at int8

| config | tokAgree | logitCos | LfinCos | sample |
|---|---|---|---|---|
| **baseline** (absmax, KV8, exp-LUT 12b, softmax 15b) | **1.00** | 0.9993 | 0.9997 | `Paris. Paris is a city` |
| KV 16-bit | 1.00 | 0.9996 | 0.9998 | `Paris. Paris is a city` |

A fully-integer forward — including integer attention and an int8 KV-cache —
reproduces the float oracle **token-for-token**, with final-layer cosine 0.9997.
Per-layer cosine stays ≥ 0.998 across all 30 layers (no error compounding at
int8). **Integer attention holds, cleanly, at the model's native precision.**

---

## 3. Result: integer softmax is remarkably cheap

| config | tokAgree | logitCos | LfinCos |
|---|---|---|---|
| exp-LUT 8-bit (256 entries) | 1.00 | 0.9994 | 0.9997 |
| exp-LUT 6-bit (64 entries) | 1.00 | 0.9974 | 0.9996 |
| exp-LUT 4-bit (**16 entries**) | 1.00 | 0.9926 | 0.9972 |
| softmax prob 8-bit | 1.00 | 0.9995 | 0.9997 |
| softmax prob 4-bit | 1.00 | 0.9772 | 0.9982 |

The fixed-point softmax is not the bottleneck. A **16-entry** exp lookup table
and 4-bit probability accumulation still generate identical text. The attention
distribution is smooth enough that coarse integer exp/normalize is harmless.
*Systems consequence:* integer attention needs no expensive exp unit — a tiny
LUT and an integer divide suffice.

---

## 4. Result: the KV-cache bit-width is the real cliff

| KV bits | tokAgree | logitCos | LfinCos | sample |
|---|---|---|---|---|
| 8 | 1.00 | 0.9993 | 0.9997 | `Paris. Paris is a city` |
| 4 | 1.00 | 0.9511 | 0.9920 | `Paris. Paris is a city` |
| **3** | 0.17 | 0.0334 | 0.9668 | `Paris, not Paris, France` |
| 2 | 0.17 | −0.60 | 0.8425 | `a capital a capital a capital` |

Under **single-prompt greedy decoding**, KV precision degrades gracefully to
**4 bits**, then generation collapses at **3 bits** even though the final-layer
hidden cosine is still 0.967. But greedy decoding is brittle — one flipped argmax
derails the whole sample — so this overstates the damage. The corpus-perplexity
study in §9 shows the *distribution* is barely changed at 3-bit (and that the
cliff, and where it sits, depends on KV quantization **granularity**, not just
bit-width). Read §9 as the authoritative version of this result.

---

## 5. Result (the headline): absmax is load-bearing — mean/median scaling breaks it

The activation quantization scale is the one knob where the choice is genuinely
contested in the literature (robust statistics suggest mean- or median-based
scales). For this trained W1.58A8 model:

| activation scale | tokAgree | logitCos | LfinCos | sample |
|---|---|---|---|---|
| absmax (max-based, native) | 1.00 | 0.9993 | 0.9997 | `Paris. Paris is a city` |
| absmean ×4 | 0.00 | 0.64 | 0.57 | `stostostMaterial…` |
| median ×8 | 0.00 | 0.39 | 0.48 | `" " " " "` |

Mean- and median-based scaling **destroy generation**. Sweeping the clip
multiplier shows why and when they recover:

| absmean × multiplier | tokAgree | LfinCos | sample |
|---|---|---|---|
| ×8 | 0.00 | 0.76 | `the for for. etc etc` |
| ×16 | 0.00 | 0.89 | `is the the capital of France` |
| ×32 | 0.33 | 0.92 | `Paris, Paris, Paris,` |
| ×64 | **1.00** | 0.95 | `Paris. Paris is a city` |

Mean-based scaling only matches absmax once the multiplier is ≈ **64** — i.e.
once it stops clipping. Measured directly, the per-tensor activation
absmax/absmean ratio is heavy-tailed: **median 7.8, p90 ≈ 106, max ≈ 1460**.

**The activation outliers carry the signal.** Any "robust" scale whose purpose
is to ignore outliers throws away exactly the information a quantization-aware
trained BitNet relies on. This is a clean negative result for robust activation
scaling in native-1-bit models, and it explains why BitNet specifies absmax.

At **corpus scale** (§9) the effect is not subtle: replacing absmax with absmean
(×4) takes perplexity from **11.2 → 2.7×10⁴**, and median (×8) to **2.8×10⁶** —
three to six orders of magnitude. The single-prompt collapse was not a fluke of
one sentence.

A secondary methodological point: at ×64 `logitCos` is only 0.12 while
generation is *identical* (tokAgree 1.00). Logit cosine is a misleadingly harsh
metric — the logit vector has a large near-degenerate tail — so **token/generation
agreement, not logit similarity, is the right gate**, exactly as one would want.

---

## 6. Summary of where it holds / breaks

- **Holds:** fully-integer forward at the model's native int8 precision is
  token-exact vs float; integer attention and int8 KV-cache are not a problem.
- **Cheap:** integer softmax — a 16-entry exp LUT is enough.
- **KV bit-width is mostly a granularity question (§9):** at corpus perplexity,
  per-vector KV is free down to **3-bit** and only collapses at 2-bit (+65 PPL);
  **per-channel asymmetric** quantization makes even **2-bit KV nearly free
  (+0.78 PPL)**. The "3-bit cliff" seen under single-prompt greedy decoding is a
  decoding-brittleness artifact, not a distributional one.
- **The one real cliff is activation scaling:** replacing absmax with mean/median
  raises perplexity by 3–6 orders of magnitude — activations are heavy-tailed and
  the outliers are load-bearing. This is the load-bearing, non-obvious result.

## 7. Honest limitations

- The corpus (§9) is a **mini-corpus** — 8 multi-genre held-out passages, 196
  tokens — not WikiText/C4. It is enough to separate "single-sentence fluke" from
  "real distributional effect" and to rank the KV-granularity options, but the
  absolute perplexities should not be quoted as benchmark numbers. Swapping in a
  real held-out set is a drop-in change (feed token-id lists to
  `ppl.corpus_nll`).
- In the perplexity harness, RMSNorm/RoPE/softmax are computed in float (the §3
  result showed their integer versions are token-exact, so the PPL deltas are
  attributable to the two knobs under study). The fully-integer norm/rope/softmax
  live in `bitnet.py::IntRunner` and are validated separately (§2–§3).
- The integer scales are simulated bit-accurately in numpy floats; a true
  fixed-point ALU would also fix the *scale* multiplies to int×shift, which this
  study does not vary.
- `act_mult` for mean/median was swept coarsely; the recovery point (~64) is
  approximate.

## 8. Reproduce

```bash
cd reference
pip install numpy tokenizers
python3 prepare_weights.py          # one-time: unpack I2_S -> int8 memmap (~2GB)
python3 _oracle.py                  # cache the float-oracle reference generation
python3 run_configs.py 0 1 2 3 4 5 6 7 8 9 10 11 12 13   # single-prompt sweep (§2-§5)
python3 run_mult.py absmean:8 absmean:16 absmean:32 absmean:64
python3 run_ppl.py 1 2 3 4 5 6 7 8 9 10 11               # corpus perplexity (§9)
```

---

## 9. Corpus perplexity and KV-quantization granularity

The single-prompt probe (§1–§5) is sharp but fragile: greedy decoding turns a tiny
logit perturbation into a derailed sample. To separate "one sentence flipped" from
"the model is actually worse", I score a held-out **mini-corpus** (8 multi-genre
passages, 196 tokens) by teacher-forced perplexity, and use it to ask the obvious
follow-up: does KV-quantization **granularity** move the cliff?

Reference (absmax activations, float KV): **PPL = 11.22**.

**KV-cache quantization (ΔPPL vs reference):**

| bits | per-vector | per-channel (sym) | per-channel (asym) |
|---|---|---|---|
| 8 | −0.21 | — | — |
| 4 | +0.31 | −0.02 | — |
| 3 | +0.19 | +0.17 | **−0.20** |
| 2 | **+65.2** | +7.57 | **+0.78** |

Two results fall out:

1. **The 3-bit "collapse" is a decoding artifact, not a distributional one.**
   Per-vector 3-bit KV costs only +0.19 PPL — the distribution is essentially
   intact; greedy decoding from a single prompt just happened to flip an argmax.
   The honest cliff for per-vector KV is **2-bit** (+65 PPL).

2. **Granularity moves the cliff almost entirely.** At 2-bit, per-vector KV is
   catastrophic (+65), per-channel-symmetric is degraded but alive (+7.6), and
   **per-channel-asymmetric is nearly free (+0.78)**. The KV cache carries
   channel-wise location/scale structure (per-channel means and ranges) that a
   single per-vector scale cannot represent; an asymmetric per-channel codebook
   captures it and recovers 2-bit. *Systems consequence:* a 2-bit KV cache (4×
   smaller than int8) is viable for this model **if** the quantizer is
   per-channel asymmetric — the bit-width alone is not the limiter.

**Activation scaling (float KV, ΔPPL vs reference):**

| scale | PPL |
|---|---|
| absmax (native) | 11.22 |
| absmean ×4 | 2.67×10⁴ |
| median ×8 | 2.85×10⁶ |

The §5 result is confirmed at corpus scale and is, if anything, starker: robust
(outlier-suppressing) activation scales raise perplexity by 3–6 orders of
magnitude. **For a quantization-aware-trained W1.58A8 model, the activation
outliers are the single most load-bearing thing in the integer path** — far more
than KV bit-width, and the opposite of where robust-statistics intuition points.
