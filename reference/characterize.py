"""
characterize.py — where does the fully-integer path stop tracking the float oracle?

Sweeps the integer-attention / quantization knobs and reports, per config:
  * tok_agree : fraction of greedily-decoded tokens that match the float oracle
  * logit_cos : cosine(int logits, float logits) at the first decode step
  * Lfin_cos  : final-layer hidden-state cosine (last prompt token)
  * sample    : the decoded continuation (coherence eye-test)

The float oracle generation is computed once and used as ground truth.
"""
import numpy as np, sys, time
from bitnet import Weights, FloatRunner, IntRunner, IntConfig
from tokenizers import Tokenizer

PROMPT = "The capital of France is"
N_NEW = 6

tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
w = Weights()
ids = [w.cfg['bos']] + tok.encode(PROMPT, add_special_tokens=False).ids

def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ---- float oracle reference generation ----
fr = FloatRunner(w); capf = {}
lf = None
for j, t in enumerate(ids):
    lf = fr.step(t, capture=capf if j == len(ids) - 1 else None)
ref_tokens = []
lg = lf
for _ in range(N_NEW):
    nx = int(np.argmax(lg)); ref_tokens.append(nx); lg = fr.step(nx)
print("ORACLE:", repr(tok.decode(ref_tokens)), flush=True)
print(f"{'config':46s} {'tokAgree':8s} {'logitCos':8s} {'LfinCos':8s}  sample", flush=True)

def run(cfg, label):
    ir = IntRunner(w, cfg); capi = {}
    li = None
    for j, t in enumerate(ids):
        li = ir.step(t, capture=capi if j == len(ids) - 1 else None)
    lcos = cos(lf, li)
    fcos = cos(capf[w.cfg['n_layer'] - 1]['x'], capi[w.cfg['n_layer'] - 1]['x'])
    gen = []
    lg = li
    for _ in range(N_NEW):
        nx = int(np.argmax(lg)); gen.append(nx); lg = ir.step(nx)
    agree = np.mean([a == b for a, b in zip(gen, ref_tokens)])
    print(f"{label:46s} {agree:8.2f} {lcos:8.4f} {fcos:8.4f}  {tok.decode(gen)!r}", flush=True)

configs = [
    (IntConfig(), "baseline(absmax,kv8,exp12,sm15)"),
    # KV-cache bit-width
    (IntConfig(kv_bits=16), "kv16"),
    (IntConfig(kv_bits=4), "kv4"),
    (IntConfig(kv_bits=3), "kv3"),
    (IntConfig(kv_bits=2), "kv2"),
    # integer softmax precision (exp LUT resolution)
    (IntConfig(exp_lut_bits=8), "exp_lut8"),
    (IntConfig(exp_lut_bits=6), "exp_lut6"),
    (IntConfig(exp_lut_bits=4), "exp_lut4"),
    (IntConfig(softmax_bits=8), "sm_bits8"),
    (IntConfig(softmax_bits=4), "sm_bits4"),
    # activation scaling: mean vs median vs absmax
    (IntConfig(act_scale="absmean"), "act_absmean"),
    (IntConfig(act_scale="median"), "act_median"),
    # interaction: aggressive KV + robust scaling
    (IntConfig(kv_bits=3, act_scale="median"), "kv3+median"),
    (IntConfig(kv_bits=3, act_scale="absmean"), "kv3+absmean"),
]
sel = sys.argv[1:] or [str(i) for i in range(len(configs))]
for i in [int(s) for s in sel]:
    t0 = time.time()
    cfg, label = configs[i]
    run(cfg, label)
print("ALLDONE", flush=True)
