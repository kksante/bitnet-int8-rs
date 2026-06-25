import sys, time, numpy as np
import ppl as P
from bitnet import Weights
from tokenizers import Tokenizer
from corpus import CORPUS

P.w = Weights(); w = P.w
tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
CORPUS_IDS = [[w.cfg['bos']] + tok.encode(s, add_special_tokens=False).ids for s in CORPUS]
NTOK = sum(len(x) - 1 for x in CORPUS_IDS)

CONFIGS = [
    ("ref(absmax,KVfloat)",        dict(act_scale="absmax")),
    ("kv8 per_vector",             dict(act_scale="absmax", kv_bits=8, kv_mode="per_vector")),
    ("kv4 per_vector",             dict(act_scale="absmax", kv_bits=4, kv_mode="per_vector")),
    ("kv3 per_vector",             dict(act_scale="absmax", kv_bits=3, kv_mode="per_vector")),
    ("kv2 per_vector",             dict(act_scale="absmax", kv_bits=2, kv_mode="per_vector")),
    ("kv4 per_channel",            dict(act_scale="absmax", kv_bits=4, kv_mode="per_channel")),
    ("kv3 per_channel",            dict(act_scale="absmax", kv_bits=3, kv_mode="per_channel")),
    ("kv2 per_channel",            dict(act_scale="absmax", kv_bits=2, kv_mode="per_channel")),
    ("kv3 per_channel_asym",       dict(act_scale="absmax", kv_bits=3, kv_mode="per_channel_asym")),
    ("kv2 per_channel_asym",       dict(act_scale="absmax", kv_bits=2, kv_mode="per_channel_asym")),
    ("act_absmean(KVfloat)",       dict(act_scale="absmean")),
    ("act_median(KVfloat)",        dict(act_scale="median")),
]

RES = '/tmp/ppl_results.txt'

def ppl_of(knobs):
    nll, n = P.corpus_nll(CORPUS_IDS, knobs)
    return float(np.exp(nll / n)), nll

# reference ppl (cache so deltas are comparable across calls)
import os, json
if os.path.exists('/tmp/ref_ppl.json'):
    ref = json.load(open('/tmp/ref_ppl.json'))
    ref_ppl, ref_nll = ref['ppl'], ref['nll']
else:
    ref_ppl, ref_nll = ppl_of(CONFIGS[0][1])
    json.dump(dict(ppl=ref_ppl, nll=ref_nll, ntok=NTOK), open('/tmp/ref_ppl.json', 'w'))
    with open(RES, 'a') as f:
        f.write(f"corpus tokens={NTOK}  REF ppl={ref_ppl:.3f}\n")
    print(f"REF ppl={ref_ppl:.3f} over {NTOK} tokens", flush=True)

for i in [int(s) for s in sys.argv[1:]]:
    label, knobs = CONFIGS[i]
    t0 = time.time()
    pp, nll = ppl_of(knobs)
    dppl = pp - ref_ppl
    line = f"{label:26s} ppl={pp:8.3f}  dPPL={dppl:+8.3f}  ({time.time()-t0:.1f}s)"
    with open(RES, 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)
