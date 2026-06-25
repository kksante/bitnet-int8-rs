"""Activation-scale recovery curve: perplexity of absmean activation quantization
as the clip multiplier grows (it must grow to ~absmax/absmean before it stops
clipping outliers and recovers). Short corpus for speed. Appends to _ppl/recovery.txt.
"""
import sys, numpy as np
import ppl as P
from bitnet import Weights
from tokenizers import Tokenizer
from corpus import CORPUS

P.w = Weights(); w = P.w
tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
SHORT = CORPUS[:6]
IDS = [[w.cfg['bos']] + tok.encode(s, add_special_tokens=False).ids for s in SHORT]
ref_nll, ntok = P.corpus_nll(IDS, dict(act_scale="absmax"))
ref_ppl = np.exp(ref_nll / ntok)
print(f"REF(absmax) ppl={ref_ppl:.3f} ntok={ntok}", flush=True)
for m in [float(x) for x in sys.argv[1:]]:
    nll, n = P.corpus_nll(IDS, dict(act_scale="absmean", act_mult=m))
    pp = float(np.exp(nll / n))
    line = f"absmean x{m:<5g} ppl={pp:.4g}"
    open('_ppl/recovery.txt', 'a').write(line + f"  ref={ref_ppl:.3f}\n")
    print(line, flush=True)
