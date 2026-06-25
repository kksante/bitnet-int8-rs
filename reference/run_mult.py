import numpy as np, sys, time
from bitnet import Weights, IntRunner, IntConfig
from tokenizers import Tokenizer
tok = Tokenizer.from_file('../bitnet-b1.58-2B-4T/tokenizer.json')
w = Weights(); o = np.load('/tmp/oracle.npz')
ids = list(o['ids']); ref = list(o['ref']); lf = o['logits']; xfin = o['xfin']
def cos(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64);return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def run(cfg,label):
    ir=IntRunner(w,cfg);capi={};li=None
    for j,t in enumerate(ids): li=ir.step(t,capture=capi if j==len(ids)-1 else None)
    lcos=cos(lf,li);fcos=cos(xfin,capi[w.cfg['n_layer']-1]['x'])
    gen=[];lg=li
    for _ in range(len(ref)): nx=int(np.argmax(lg));gen.append(nx);lg=ir.step(nx)
    ag=float(np.mean([a==b for a,b in zip(gen,ref)]))
    print(f"{label:26s} tokAgree={ag:.2f} logitCos={lcos:.4f} LfinCos={fcos:.4f} {tok.decode(gen)!r}",flush=True)
specs=[]
for m in sys.argv[1:]:
    scale,mult=m.split(':'); specs.append((scale,float(mult)))
for scale,mult in specs:
    run(IntConfig(act_scale=scale,act_mult=mult), f"{scale}x{mult}")
