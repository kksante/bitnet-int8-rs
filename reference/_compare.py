import numpy as np, time, sys
from bitnet import Weights, FloatRunner, IntRunner, IntConfig
from tokenizers import Tokenizer

tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
w = Weights()
ids = [w.cfg['bos']] + tok.encode('The capital of France is', add_special_tokens=False).ids

def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# prefill both, capturing last-token per-layer x
fr = FloatRunner(w); ir = IntRunner(w, IntConfig())
capf = {}; capi = {}
t0 = time.time()
for j, t in enumerate(ids):
    lf = fr.step(t, capture=capf if j == len(ids) - 1 else None)
    li = ir.step(t, capture=capi if j == len(ids) - 1 else None)
print(f'prefill done {time.time()-t0:.1f}s', flush=True)
print('float next:', repr(tok.decode([int(np.argmax(lf))])),
      ' int next:', repr(tok.decode([int(np.argmax(li))])), flush=True)
print('logit cosine:', round(cos(lf, li), 4), flush=True)
print('per-layer last-token cosine (int vs float):', flush=True)
for l in range(0, w.cfg['n_layer'], 3):
    print(f'  L{l:2d}: {cos(capf[l]["x"], capi[l]["x"]):.4f}', flush=True)
print(f'  L29: {cos(capf[29]["x"], capi[29]["x"]):.4f}', flush=True)
print('DONE', flush=True)
