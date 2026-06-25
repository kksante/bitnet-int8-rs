"""Clean activation-bit table: naive vs +rotation vs +rotation+per-group(32) vs
kitchen-sink, across bit-widths. Appends to _ppl/lowbit.txt."""
import sys, numpy as np
import ppl as P
from bitnet import Weights
from tokenizers import Tokenizer
from corpus import CORPUS
P.w = Weights(); w = P.w
tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
IDS = [tok.encode(s, add_special_tokens=False).ids for s in CORPUS[:6]]
n = sum(len(i) - 1 for i in IDS)
ref = float(np.exp(P.corpus_nll(IDS, dict(act_scale='absmax'))[0] / n))
def ppl(k): return float(np.exp(P.corpus_nll(IDS, k)[0] / n))
methods = {
    'naive':        lambda b: dict(act_scale='absmax', act_bits=b),
    'rot':          lambda b: dict(act_scale='absmax', act_bits=b, act_rotate=True),
    'rot+g32':      lambda b: dict(act_scale='absmax', act_bits=b, act_rotate=True, act_group=32),
    'rot+g16+keep64': lambda b: dict(act_scale='absmax', act_bits=b, act_rotate=True, act_group=16, act_keep=64),
}
print(f"ref={ref:.3f} ntok={n}", flush=True)
for arg in sys.argv[1:]:
    b, meth = arg.split(':')
    b = int(b)
    pp = ppl(methods[meth](b))
    line = f"act{b}b {meth:16s} ppl={pp:12.3f}  dPPL={pp-ref:+.3f}"
    open('_ppl/lowbit.txt', 'a').write(line + "\n")
    print(line, flush=True)
