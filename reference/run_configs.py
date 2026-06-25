import numpy as np, sys, time, os
from bitnet import Weights, IntRunner, IntConfig
from tokenizers import Tokenizer

tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
w = Weights()
o = np.load('/tmp/oracle.npz')
ids = list(o['ids']); ref = list(o['ref']); lf = o['logits']; xfin = o['xfin']
N_NEW = len(ref)

def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

CONFIGS = [
    (IntConfig(), "baseline(absmax,kv8,exp12,sm15)"),
    (IntConfig(kv_bits=16), "kv16"),
    (IntConfig(kv_bits=4), "kv4"),
    (IntConfig(kv_bits=3), "kv3"),
    (IntConfig(kv_bits=2), "kv2"),
    (IntConfig(exp_lut_bits=8), "exp_lut8"),
    (IntConfig(exp_lut_bits=6), "exp_lut6"),
    (IntConfig(exp_lut_bits=4), "exp_lut4"),
    (IntConfig(softmax_bits=8), "sm_bits8"),
    (IntConfig(softmax_bits=4), "sm_bits4"),
    (IntConfig(act_scale="absmean"), "act_absmean"),
    (IntConfig(act_scale="median"), "act_median"),
    (IntConfig(kv_bits=3, act_scale="median"), "kv3+median"),
    (IntConfig(kv_bits=2, act_scale="median"), "kv2+median"),
]

RESULTS = '/tmp/char_results.txt'

def run(cfg, label):
    ir = IntRunner(w, cfg); capi = {}
    li = None
    for j, t in enumerate(ids):
        li = ir.step(t, capture=capi if j == len(ids) - 1 else None)
    lcos = cos(lf, li); fcos = cos(xfin, capi[w.cfg['n_layer'] - 1]['x'])
    first_match = int(np.argmax(li) == ref[0])
    gen = []; lg = li
    for _ in range(N_NEW):
        nx = int(np.argmax(lg)); gen.append(nx); lg = ir.step(nx)
    agree = float(np.mean([a == b for a, b in zip(gen, ref)]))
    line = f"{label:34s} tokAgree={agree:.2f} firstTok={first_match} logitCos={lcos:.4f} LfinCos={fcos:.4f} sample={tok.decode(gen)!r}"
    with open(RESULTS, 'a') as f:
        f.write(line + "\n")
    print(line, flush=True)

for i in [int(s) for s in sys.argv[1:]]:
    t0 = time.time(); cfg, label = CONFIGS[i]; run(cfg, label)
    print(f"  ({time.time()-t0:.1f}s)", flush=True)
