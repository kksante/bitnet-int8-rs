import sys, numpy as np
sys.path.insert(0, '.')
from bitnet_float import BitNetFloat
from tokenizers import Tokenizer

tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
m = BitNetFloat('../bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf')

def gen(prompt, n=12):
    ids = [m.cfg['bos']] + tok.encode(prompt, add_special_tokens=False).ids
    for _ in range(n):
        lg = m.forward(ids)
        nx = int(np.argmax(lg))
        ids.append(nx)
        if nx == m.cfg['eos']:
            break
    print(f'  {prompt!r} -> {tok.decode(ids[1:])!r}', flush=True)

print('=== generation (hf rope, no perm, actq) ===', flush=True)
gen('The capital of France is')
gen('Paris Paris Paris Paris Paris')
gen('1 2 3 4 5 6')
print('DONE', flush=True)
