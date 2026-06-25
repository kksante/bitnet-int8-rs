"""Sub-int8 activation study on the SHIPPED model (no retraining, weights untouched).

Compares naive per-token uniform activation quantization against the same in a
rotated (QuaRot-style Hadamard) basis, sweeping activation bit-width. Question:
can a cheap post-hoc rotation push b1.58 activations below int8?

  python3 run_actbits.py 8 6 5 4 3
"""
import sys, time, numpy as np
import ppl as P
from bitnet import Weights
from tokenizers import Tokenizer
from corpus import CORPUS

P.w = Weights(); w = P.w
tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
SHORT = CORPUS[:6]
IDS = [[w.cfg['bos']] + tok.encode(s, add_special_tokens=False).ids for s in SHORT]
ref_nll, n = P.corpus_nll(IDS, dict(act_scale="absmax"))
ref = float(np.exp(ref_nll / n))
print(f"REF (native int8 absmax) ppl={ref:.3f}  ntok={n}", flush=True)
for b in [int(x) for x in sys.argv[1:]]:
    t0 = time.time()
    nll_n, _ = P.corpus_nll(IDS, dict(act_scale="absmax", act_bits=b, act_rotate=False))
    nll_r, _ = P.corpus_nll(IDS, dict(act_scale="absmax", act_bits=b, act_rotate=True))
    pn, pr = np.exp(nll_n / n), np.exp(nll_r / n)
    line = f"act {b}-bit: naive ppl={pn:11.3f}   rotated ppl={pr:11.3f}   ({time.time()-t0:.1f}s)"
    open('_ppl/actbits.txt', 'a').write(line + "\n")
    print(line, flush=True)
