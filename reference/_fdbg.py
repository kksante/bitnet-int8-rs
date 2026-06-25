import numpy as np, itertools, sys
import ppl as P
from bitnet import Weights
from tokenizers import Tokenizer
P.w = Weights(cache='/tmp/cache_f1b'); w = P.w
tok = Tokenizer.from_file('../models/falcon3-1b/tokenizer.json')
c = w.cfg
d, hd, nh, nkv = c['d_model'], c['head_dim'], c['n_head'], c['n_kv']
eps, base, rep = c['rms_eps'], c['rope_base'], nh // nkv
ids = tok.encode('The capital of France is', add_special_tokens=False).ids
T = len(ids)
cosv = np.cos(np.outer(np.arange(T), 1.0/(base**(np.arange(0,hd,2)/hd))))
sinv = np.sin(np.outer(np.arange(T), 1.0/(base**(np.arange(0,hd,2)/hd))))
mask = np.triu(np.full((T, T), -np.inf, np.float32), 1); scl = 1.0/np.sqrt(hd)

def lin(name, x, sign, aq):
    W, sc = w.W(name); Wf = (sign*W).T.astype(np.float32)
    if aq:
        s = 127.0/np.maximum(np.abs(x).max(1, keepdims=True), 1e-5)
        xq = np.clip(np.round(x*s), -128, 127)
        return (xq @ Wf)*(sc/s)
    return (x @ Wf)*sc

def rope(x):
    x1, x2 = x[..., :hd//2], x[..., hd//2:]
    cc = cosv[:, None, :]; ss = sinv[:, None, :]
    return np.concatenate([x1*cc - x2*ss, x2*cc + x1*ss], -1)

def fwd(ffn='silu', sign=1, aq=True):
    x = w.embed[np.asarray(ids)].astype(np.float32)
    for li in range(c['n_layer']):
        p = f'blk.{li}.'
        h = P._rms(x, w.norms[p+'attn_norm'], eps)
        q = lin(p+'attn_q.weight', h, sign, aq).reshape(T, nh, hd)
        k = lin(p+'attn_k.weight', h, sign, aq).reshape(T, nkv, hd)
        v = lin(p+'attn_v.weight', h, sign, aq).reshape(T, nkv, hd)
        q = rope(q); k = rope(k)
        kk = np.repeat(k, rep, 1); vv = np.repeat(v, rep, 1)
        o = np.empty((T, nh, hd), np.float32)
        for hh in range(nh):
            sc = (q[:, hh]@kk[:, hh].T)*scl + mask; sc -= sc.max(1, keepdims=True)
            e = np.exp(sc); o[:, hh] = (e/e.sum(1, keepdims=True))@vv[:, hh]
        x = x + lin(p+'attn_output.weight', o.reshape(T, d), sign, aq)
        h2 = P._rms(x, w.norms[p+'ffn_norm'], eps)
        g = lin(p+'ffn_gate.weight', h2, sign, aq); u = lin(p+'ffn_up.weight', h2, sign, aq)
        if ffn == 'silu': act = (g/(1+np.exp(-g)))*u
        elif ffn == 'silu_swap': act = (u/(1+np.exp(-u)))*g
        elif ffn == 'relu2': act = np.square(np.maximum(g, 0))*u
        elif ffn == 'gelu': act = (0.5*g*(1+np.tanh(0.7978845608*(g+0.044715*g**3))))*u
        x = x + lin(p+'ffn_down.weight', act, sign, aq)
    hn = P._rms(x, w.norms['output_norm'], eps)
    return w.lm_head(hn[-1])

for ffn, sign, aq in itertools.product(['silu','silu_swap','relu2','gelu'], [1,-1], [True]):
    lg = fwd(ffn, sign, aq); t = int(np.argmax(lg))
    print(f'ffn={ffn:9s} sign={sign:+d} aq={aq}: {tok.decode([t])!r}')
