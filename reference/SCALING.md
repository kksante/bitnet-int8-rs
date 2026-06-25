# Scaling the study locally (full corpora + multiple model sizes)

The in-repo results were produced under a constrained sandbox (numpy, ~45s/command,
~3.4 GB RAM), so they use small corpus slices and one model. Everything is built to
scale on your machine with no code changes — only environment variables.

## 0. One-time setup

```bash
cd reference
pip install numpy tokenizers matplotlib
python3 prepare_weights.py            # unpack I2_S -> int8 memmap for the 2B model
```

## 1. Full WikiText-2 (or any corpus)

Download WikiText-2 raw test (a few MB) and point `BN_CORPUS` at it. Drop the
`BN_MAX_CHUNKS` cap to score the whole thing; raise `BN_CHUNK_TOKENS` to the model
context if you like.

```bash
# full corpus, all configs, results -> _ppl_wiki_full/
BN_PPL_OUT=_ppl_wiki_full \
BN_CORPUS=/path/to/wiki.test.raw \
BN_CHUNK_TOKENS=512 \
python3 run_ppl2.py 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14

python3 analyze_ci.py _ppl_wiki_full      # perplexities + bootstrap 95% CIs
```

`run_ppl2.py` writes per-token NLL arrays, so CIs and any later analysis are free.

## 2. Multiple model sizes / architectures

Any BitNet-family GGUF in `I2_S` format works — the loader reads the architecture
and dimensions from metadata, and the forward auto-detects an **architecture
profile** from the tensor list:

* **FFN activation**: squared-ReLU for BitNet b1.58 2B4T; SwiGLU/SiLU otherwise
  (Falcon3, Llama-1.58).
* **sub-layer norms** (`attn_sub_norm`/`ffn_sub_norm`): used if present, skipped
  if absent (Falcon3/Llama have none).
* **LM head**: tied embeddings, or a separate `output.weight` (kept F16) if untied.

So e.g. the Falcon3-1.58bit series drops in directly:

```bash
huggingface-cli download tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF --local-dir models/falcon3-1b
BITNET_GGUF=models/falcon3-1b/ggml-model-i2_s.gguf BN_CACHE=/tmp/cache_falcon3_1b \
  python3 prepare_weights.py
BN_CACHE=/tmp/cache_falcon3_1b BITNET_TOKENIZER=models/falcon3-1b/tokenizer.json \
  BN_PPL_OUT=_ppl_falcon3_1b BN_CORPUS=/path/wiki.test.raw \
  python3 run_ppl2.py 0 1 4 9 11 12 13
python3 analyze_ci.py _ppl_falcon3_1b
```

For each model, prepare a cache, then run with that cache and the model's tokenizer:

```bash
for M in bitnet-2B bitnet-700M bitnet-3B; do
  BITNET_GGUF=/models/$M/ggml-model-i2_s.gguf BN_CACHE=/tmp/cache_$M \
    python3 prepare_weights.py
  BN_CACHE=/tmp/cache_$M \
  BITNET_TOKENIZER=/models/$M/tokenizer.json \
  BN_PPL_OUT=_ppl_$M \
  BN_CORPUS=/path/to/wiki.test.raw \
    python3 run_ppl2.py 0 4 9 11 12 13      # headline cells (or 0..14 for all)
  python3 analyze_ci.py _ppl_$M
done
```

`sweep_local.sh` wraps this loop.

## 3. Figures

`make_figures.py` reads `_ppl/mech.npz` and the measured sweep values. Point it at
your full-corpus numbers (edit the hardcoded sweep dict, or extend it to read the
`_ppl_*/` arrays) and re-run to regenerate `figures/*.pdf` for the paper.

## 4. Notes / fairness

* Perplexity is teacher-forced; chunks are non-overlapping windows (standard).
* For a strict fixed-point study, also vary the integer scale-multiply precision
  (currently exact); the hooks are in `bitnet.py::IntRunner`.
* KV quantization granularity is the dominant KV knob (see the paper); per-channel
  asymmetric, optionally pre-RoPE for Keys, is the recommended setting.
