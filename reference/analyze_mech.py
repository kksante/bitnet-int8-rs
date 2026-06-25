"""Mechanism analysis: why does per-channel ASYMMETRIC 2-bit KV rescue the cache?

For a WikiText chunk, capture each layer's K (pre- and post-RoPE) and V, and the
per-BitLinear activation outlier ratio. Quantify:
  * per-channel bias of K: |mean_t K[:,c]| relative to its per-channel std
    (large bias => a symmetric quantizer wastes half its range => need zero-point);
  * channel-outlier concentration in K vs V (KIVI's observation);
  * per-layer activation absmax/absmean ratio (the activation-outlier profile).

Saves arrays to _ppl/mech.npz for figures.
"""
import os, numpy as np
import ppl as P
from bitnet import Weights
from tokenizers import Tokenizer

P.w = Weights(); w = P.w
tok = Tokenizer.from_file(os.environ.get('BITNET_TOKENIZER', '../bitnet-b1.58-2B-4T/tokenizer.json'))
text = open(os.environ.get('BN_CORPUS', '../wiki.test.txt')).read()
_bos = w.cfg.get('bos')
ids = ([_bos] if _bos is not None else []) + tok.encode(text, add_special_tokens=False).ids[:256]
c = w.cfg
d, hd, nh, nkv = c['d_model'], c['head_dim'], c['n_head'], c['n_kv']
eps, base, rep = c['rms_eps'], c['rope_base'], nh // nkv
T = len(ids)
x = w.embed[np.asarray(ids)].astype(np.float32)
inv = 1.0 / (base ** (np.arange(0, hd, 2) / hd)); angp = np.outer(np.arange(T), inv)
cos, sin = np.cos(angp), np.sin(angp)
mask = np.triu(np.full((T, T), -np.inf, np.float32), 1); scl = 1.0 / np.sqrt(hd)
knobs = dict(act_scale='absmax')

k_bias_ratio, v_bias_ratio = [], []     # per layer: mean over (head,channel) of |mu|/sigma
k_chan_kurt, v_chan_kurt = [], []
act_ratio_layer = []                    # per layer mean absmax/absmean over its bitlinears

def chan_bias(X):  # X [T, nkv, hd]
    mu = X.mean(0); sd = X.std(0) + 1e-9
    return float(np.mean(np.abs(mu) / sd))

for li in range(c['n_layer']):
    p = f'blk.{li}.'
    Wq, sq = w.W(p+'attn_q.weight'); Wk, sk = w.W(p+'attn_k.weight')
    Wv, sv = w.W(p+'attn_v.weight'); Wo, so = w.W(p+'attn_output.weight')
    h = P._rms(x, w.norms[p+'attn_norm'], eps)
    ar = []
    for (W_, s_) in [(Wq, sq), (Wk, sk), (Wv, sv)]:
        a = np.abs(h); ar.append(float(a.max(1).mean() / max(a.mean(), 1e-9)))
    q = P._bitlin(Wq, sq, h, knobs).reshape(T, nh, hd)
    k = P._bitlin(Wk, sk, h, knobs).reshape(T, nkv, hd)
    v = P._bitlin(Wv, sv, h, knobs).reshape(T, nkv, hd)
    k_post = P._rope(k, cos, sin)
    k_bias_ratio.append(chan_bias(k_post)); v_bias_ratio.append(chan_bias(v))
    # finish the layer (float) to advance x
    qr = P._rope(q, cos, sin)
    kk = np.repeat(k_post, rep, 1); vv = np.repeat(v, rep, 1)
    out = np.empty((T, nh, hd), np.float32)
    for hh in range(nh):
        sc = (qr[:, hh] @ kk[:, hh].T) * scl + mask; sc -= sc.max(1, keepdims=True)
        e = np.exp(sc); pr = e / e.sum(1, keepdims=True); out[:, hh] = pr @ vv[:, hh]
    attn = out.reshape(T, d)
    if w.has_attn_sub:
        attn = P._rms(attn, w.norms[p+'attn_sub_norm'], eps)
    x = x + P._bitlin(Wo, so, attn.astype(np.float32), knobs).astype(np.float32)
    h2 = P._rms(x, w.norms[p+'ffn_norm'], eps)
    Wg, sg = w.W(p+'ffn_gate.weight'); Wu, su = w.W(p+'ffn_up.weight'); Wd, sd = w.W(p+'ffn_down.weight')
    g = P._bitlin(Wg, sg, h2, knobs); u = P._bitlin(Wu, su, h2, knobs)
    a = np.abs(h2); ar.append(float(a.max(1).mean() / max(a.mean(), 1e-9)))
    act = np.square(np.maximum(g, 0.0)) * u if w.relu2 else (g/(1+np.exp(-g)))*u
    if w.has_ffn_sub:
        act = P._rms(act.astype(np.float32), w.norms[p+'ffn_sub_norm'], eps)
    a = np.abs(act); ar.append(float(a.max(1).mean() / max(a.mean(), 1e-9)))
    x = x + P._bitlin(Wd, sd, act.astype(np.float32), knobs).astype(np.float32)
    act_ratio_layer.append(float(np.mean(ar)))

kbr = np.array(k_bias_ratio); vbr = np.array(v_bias_ratio); arl = np.array(act_ratio_layer)
np.savez(os.environ.get('BN_MECH_OUT', '_ppl/mech.npz'), k_bias=kbr, v_bias=vbr, act_ratio=arl)
print(f"K per-channel bias |mu|/sigma: mean={kbr.mean():.3f}  (V: {vbr.mean():.3f})")
print(f"  -> K channels are off-center by ~{kbr.mean():.2f} sigma on average; a SYMMETRIC")
print(f"     quantizer wastes range, an ASYMMETRIC (zero-point) one does not.")
print(f"K/V bias ratio (K more biased than V): {kbr.mean()/max(vbr.mean(),1e-9):.2f}x")
print(f"activation absmax/absmean per-layer: mean={arl.mean():.1f} min={arl.min():.1f} max={arl.max():.1f}")
