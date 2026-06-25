#!/usr/bin/env bash
# Local full-scale sweep: WikiText-2 (or any corpus) across one or more BitNet
# GGUF models. Usage:
#   BN_CORPUS=/path/wiki.test.raw ./sweep_local.sh /models/a.gguf:/models/a/tokenizer.json [more...]
# Each arg is  GGUF_PATH:TOKENIZER_PATH .
set -euo pipefail
CORPUS="${BN_CORPUS:?set BN_CORPUS to a text file}"
CHUNK="${BN_CHUNK_TOKENS:-512}"
CONFIGS="${CONFIGS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14}"

for spec in "$@"; do
  GGUF="${spec%%:*}"; TOK="${spec##*:}"
  name="$(basename "$(dirname "$GGUF")")"
  cache="/tmp/cache_${name}"; out="_ppl_${name}"
  echo "=== $name ==="
  BITNET_GGUF="$GGUF" BN_CACHE="$cache" python3 prepare_weights.py
  BN_CACHE="$cache" BITNET_TOKENIZER="$TOK" BN_PPL_OUT="$out" \
    BN_CORPUS="$CORPUS" BN_CHUNK_TOKENS="$CHUNK" \
    python3 run_ppl2.py $CONFIGS
  python3 analyze_ci.py "$out"
done
