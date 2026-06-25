"""Can we push activations to 2-bit / 1-bit on the shipped model (no retraining)?

Tests the granularity lever (per-group activation scaling) and mixed precision
(keep a few outlier channels at int8), with/without the QuaRot rotation, at low
bit-widths. Uses the BitNet cache by default (BN_CACHE). Short corpus for speed.

  BN_CACHE=$(pwd)/_cache python3 run_lowbit.py
"""
import numpy as np
import ppl as P
from bitnet import Weights
from tokenizers import Tokenizer
from corpus import CORPUS

P.w = Weights(); w = P.w
import os
tok = Tokenizer.from_file(os.environ.get('BITNET_TOKENIZER', '../bitnet-b1.58-2B-4T/tokenizer.json'))
SHORT = CORPUS[:6]
bos = w.cfg.get('bos'); pre = [bos] if bos is not None else []
IDS = [pre + tok.encode(s, add_special_tokens=False).ids for s in SHORT]
ref_nll, n = P.corpus_nll(IDS, dict(act_scale="absmax"))
ref = float(np.exp(ref_nll / n))
print(f"REF (native int8) ppl={ref:.3f}  ntok={n}\n", flush=True)

def run(label, knobs):
    nll, _ = P.corpus_nll(IDS, knobs)
    print(f"  {label:38s} ppl={np.exp(nll/n):12.3f}", flush=True)

import sys
which = sys.argv[1] if len(sys.argv) > 1 else "2"
for b in ([2] if which == "2" else [1]):
    print(f"=== {b}-bit activations ===", flush=True)
    run(f"per-token, no rot", dict(act_scale="absmax", act_bits=b))
    run(f"per-token + rot", dict(act_scale="absmax", act_bits=b, act_rotate=True))
    run(f"per-group(128) + rot", dict(act_scale="absmax", act_bits=b, act_rotate=True, act_group=128))
    run(f"per-group(32) + rot", dict(act_scale="absmax", act_bits=b, act_rotate=True, act_group=32))
    run(f"per-group(32) no rot", dict(act_scale="absmax", act_bits=b, act_group=32))
    run(f"rot + keep 8 outliers int8", dict(act_scale="absmax", act_bits=b, act_rotate=True, act_keep=8))
    run(f"per-group(32)+rot+keep8", dict(act_scale="absmax", act_bits=b, act_rotate=True, act_group=32, act_keep=8))
