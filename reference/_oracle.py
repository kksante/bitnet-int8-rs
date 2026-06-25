import numpy as np, time
from bitnet import Weights, FloatRunner
from tokenizers import Tokenizer
PROMPT = "The capital of France is"; N_NEW = 6
tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
w = Weights()
ids = [w.cfg['bos']] + tok.encode(PROMPT, add_special_tokens=False).ids
fr = FloatRunner(w); capf = {}
t0 = time.time(); lf = None
for j, t in enumerate(ids):
    lf = fr.step(t, capture=capf if j == len(ids) - 1 else None)
ref = []; lg = lf
for _ in range(N_NEW):
    nx = int(np.argmax(lg)); ref.append(nx); lg = fr.step(nx)
np.savez('/tmp/oracle.npz', ids=np.array(ids), ref=np.array(ref),
         logits=lf.astype(np.float32), xfin=capf[w.cfg['n_layer']-1]['x'].astype(np.float32))
print('oracle saved', time.time()-t0, 's; gen=', repr(tok.decode(ref)))
