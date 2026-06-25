"""Compute per-token NLL for each config and cache it to _ppl/<label>.npy so we
can compute perplexity and bootstrap confidence intervals offline.

Usage: python3 run_ppl2.py 0 1 2   (config indices; see CONFIGS)
"""
import sys, os, time, numpy as np
import ppl as P
from bitnet import Weights
from tokenizers import Tokenizer
from corpus import CORPUS

OUT = os.environ.get("BN_PPL_OUT", "_ppl"); os.makedirs(OUT, exist_ok=True)
P.w = Weights(); w = P.w
import os as _os
tok = Tokenizer.from_file(_os.environ.get(
    'BITNET_TOKENIZER', '../bitnet-b1.58-2B-4T/tokenizer.json'))
from wikitext_loader import corpus_from_env
CORPUS_IDS, CORPUS_NAME = corpus_from_env(tok, w.cfg.get('bos'))
print(f"corpus: {CORPUS_NAME}", flush=True)

CONFIGS = [
    ("ref",                 dict(act_scale="absmax")),
    ("pv8",                 dict(act_scale="absmax", kv_bits=8, kv_mode="per_vector")),
    ("pv4",                 dict(act_scale="absmax", kv_bits=4, kv_mode="per_vector")),
    ("pv3",                 dict(act_scale="absmax", kv_bits=3, kv_mode="per_vector")),
    ("pv2",                 dict(act_scale="absmax", kv_bits=2, kv_mode="per_vector")),
    ("pv_asym3",            dict(act_scale="absmax", kv_bits=3, kv_mode="per_vector_asym")),
    ("pv_asym2",            dict(act_scale="absmax", kv_bits=2, kv_mode="per_vector_asym")),
    ("pc4",                 dict(act_scale="absmax", kv_bits=4, kv_mode="per_channel")),
    ("pc3",                 dict(act_scale="absmax", kv_bits=3, kv_mode="per_channel")),
    ("pc2",                 dict(act_scale="absmax", kv_bits=2, kv_mode="per_channel")),
    ("pc_asym3",            dict(act_scale="absmax", kv_bits=3, kv_mode="per_channel_asym")),
    ("pc_asym2",            dict(act_scale="absmax", kv_bits=2, kv_mode="per_channel_asym")),
    ("pc_asym2_prerope",    dict(act_scale="absmax", kv_bits=2, kv_mode="per_channel_asym", kv_pre_rope=True)),
    ("act_absmean",         dict(act_scale="absmean")),
    ("act_median",          dict(act_scale="median")),
]

# save sentence ids once
sent_path = os.path.join(OUT, "sent.npy")
for i in [int(s) for s in sys.argv[1:]]:
    label, knobs = CONFIGS[i]
    t0 = time.time()
    nlls, sent = P.corpus_nll_tokens(CORPUS_IDS, knobs)
    np.save(os.path.join(OUT, f"{label}.npy"), nlls)
    np.save(sent_path, sent)  # always refresh (corpus may have changed)
    ppl = float(np.exp(nlls.mean()))
    print(f"{label:18s} ppl={ppl:9.3f}  ntok={nlls.size}  ({time.time()-t0:.1f}s)", flush=True)
