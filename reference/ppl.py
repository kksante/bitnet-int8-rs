"""
ppl.py — corpus perplexity for the BitNet integer path, with KV-quantization
modes (per-vector / per-channel / per-channel-asymmetric).

This extends the single-prompt probe in characterize.py to a held-out mini-corpus
and a proper metric (perplexity / NLL), and isolates the two contested knobs:

  * activation quantization scale: absmax vs absmean vs median (and multiplier);
  * KV-cache quantization: bit-width AND granularity (per-vector vs per-channel,
    symmetric vs asymmetric).

Design choices (stated for honesty):
  * Forward is *batched / teacher-forced* (full causal attention over each
    sequence at once) so scoring a corpus is feasible under tight compute.
  * The A8 part is always real: per-token int8 activation quant + integer-exact
    ternary matmul. RMSNorm / RoPE / softmax are computed in float here, because
    the single-prompt sweep (FINDINGS.md §3) showed the integer versions are
    token-exact even at a 16-entry exp LUT — so the perplexity deltas below are
    attributable to the two knobs under study, not to norm/rope/softmax.
"""
import os, sys, numpy as np
from bitnet import Weights

w = None  # lazy global


def _rms(x, g, eps):
    v = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(v + eps) * g).astype(np.float32)


def _act_quant_rows(x, mode, mult):
    a = np.abs(x)
    if mode == "absmax":
        amax = np.maximum(a.max(axis=1, keepdims=True), 1e-5)
    elif mode == "absmean":
        amax = np.maximum(a.mean(axis=1, keepdims=True) * (mult or 4.0), 1e-5)
    elif mode == "median":
        amax = np.maximum(np.median(a, axis=1, keepdims=True) * (mult or 8.0), 1e-5)
    else:
        raise ValueError(mode)
    s = 127.0 / amax
    q = np.clip(np.round(x * s), -128, 127)
    return q, s  # real = q / s


# --- QuaRot-style exact orthonormal rotation for sub-int8 activation study ---
# For a length-d activation we factor d = p*m with p the largest power of two
# dividing d, and use R = H_p (x) O_m: a normalized Sylvester Hadamard on the
# power-of-two axis (the O(d log d) "free" part) and a fixed small orthonormal
# block O_m on the remainder. R is exactly orthonormal, so x -> R x -> quantize
# -> R^T is a valid post-hoc activation quantizer that requires no retraining and
# leaves the ternary weights untouched.
_ROT = {}
def _make_rot(d):
    p = 1
    while d % (p * 2) == 0:
        p *= 2
    m = d // p
    H = np.array([[1.0]])
    while H.shape[0] < p:
        H = np.block([[H, H], [H, -H]])
    H = (H / np.sqrt(p)).astype(np.float32)
    rng = np.random.default_rng(0)
    O, _ = np.linalg.qr(rng.standard_normal((m, m)))
    return (H, O.astype(np.float32), p, m)

def _rot(x, d, inverse=False):
    if d not in _ROT:
        _ROT[d] = _make_rot(d)
    H, O, p, m = _ROT[d]
    T = x.shape[0]
    xr = x.reshape(T, p, m).astype(np.float32)
    if not inverse:
        xr = xr @ O                 # O on m-axis  [T,p,m]
        xr = np.matmul(H, xr)       # H on p-axis (BLAS, H broadcast over T)
    else:
        xr = np.matmul(H, xr)       # H symmetric -> H^T = H
        xr = xr @ O.T
    return xr.reshape(T, d)

def _quant_uniform(z, bits, asym=False, nonuniform=False):
    """Quant of z over its last axis. asym = per-row(-group) zero-point; nonuniform
    = Lloyd-Max optimal levels (a few k-means iters per row/group)."""
    if nonuniform and bits >= 2:
        return _quant_lloyd(z, bits)
    if asym and bits >= 2:
        lo = z.min(axis=-1, keepdims=True); hi = z.max(axis=-1, keepdims=True)
        rng = np.maximum(hi - lo, 1e-9); L = (1 << bits) - 1
        s = L / rng
        return np.clip(np.round((z - lo) * s), 0, L) / s + lo
    amax = np.maximum(np.abs(z).max(axis=-1, keepdims=True), 1e-5)
    if bits <= 1:
        return np.sign(z) * amax                       # 1-bit sign {-amax,+amax}
    hi = (1 << (bits - 1)) - 1
    s = hi / amax
    return np.clip(np.round(z * s), -hi - 1, hi) / s


def _quant_lloyd(z, bits, iters=8):
    """Lloyd-Max (k-means) scalar quantizer along the last axis, per row/group."""
    K = 1 << bits
    zf = z.reshape(-1, z.shape[-1])
    out = np.empty_like(zf)
    for r in range(zf.shape[0]):
        v = zf[r]
        c = np.quantile(v, (np.arange(K) + 0.5) / K)   # init at quantiles
        for _ in range(iters):
            d = np.abs(v[:, None] - c[None, :])
            a = d.argmin(1)
            for k in range(K):
                m = a == k
                if m.any():
                    c[k] = v[m].mean()
        out[r] = c[np.abs(v[:, None] - c[None, :]).argmin(1)]
    return out.reshape(z.shape)


def _act_quant_bits(x, bits, rotate, group=None, keep=0, asym=False, nonuniform=False):
    """Activation quant to `bits` bits for the sub-int8 study.

    rotate : apply the QuaRot-style orthonormal rotation before quantizing.
    group  : per-group scaling (group_size channels share a scale) instead of
             per-token; the granularity lever that rescued the KV cache.
    keep   : mixed precision -- keep the `keep` largest-|.| channels (per token)
             at int8 and quantize the rest to `bits`. Outliers preserved exactly.
    Returns the dequantized (float) activation for the ternary matmul.
    """
    d = x.shape[1]
    xr = _rot(x, d) if rotate else x
    # mixed precision: pull out the top-`keep` channels per token, keep at int8
    hi_mask = None
    if keep > 0:
        idx = np.argpartition(np.abs(xr), -keep, axis=1)[:, -keep:]
        hi_mask = np.zeros_like(xr, dtype=bool)
        np.put_along_axis(hi_mask, idx, True, axis=1)
        hi_vals = _quant_uniform(np.where(hi_mask, xr, 0.0), 8)
    base = xr
    qkw = dict(asym=asym, nonuniform=nonuniform)
    if group and group < d:
        ng = d // group
        body = base[:, :ng * group].reshape(base.shape[0], ng, group)
        q = _quant_uniform(body, bits, **qkw).reshape(base.shape[0], ng * group)
        if d % group:                                   # remainder group
            tail = _quant_uniform(base[:, ng * group:], bits, **qkw)
            q = np.concatenate([q, tail], axis=1)
        xq = q
    else:
        xq = _quant_uniform(base, bits, **qkw)
    if hi_mask is not None:
        xq = np.where(hi_mask, hi_vals, xq)             # splice outliers back at int8
    return _rot(xq, d, inverse=True) if rotate else xq


def _bitlin(W, sc, x, knobs, name=""):
    ab = knobs.get("act_bits")
    if ab is not None:
        # per-projection mixed precision: projections whose suffix is listed in
        # act_hp ("high precision") stay at int8; others get `act_bits`.
        hp = knobs.get("act_hp")
        bits = 8 if (hp and any(t in name for t in hp)) else ab
        # sub-int8 activation study (PTQ on the shipped model, weights untouched)
        xq = _act_quant_bits(x, bits, knobs.get("act_rotate", False),
                             group=knobs.get("act_group"), keep=knobs.get("act_keep", 0),
                             asym=knobs.get("act_asym", False),
                             nonuniform=knobs.get("act_nonuniform", False))
        return (xq.astype(np.float32) @ W.T.astype(np.float32)).astype(np.float64) * sc
    q, s = _act_quant_rows(x, knobs["act_scale"], knobs.get("act_mult"))
    # ternary x int8 accumulates to <= 6912*127 ~ 8.8e5, exactly representable in
    # float32, so we use BLAS float matmul (fast) and stay bit-exact to int32.
    acc = q.astype(np.float32) @ W.T.astype(np.float32)    # [T,out], integer-valued
    return acc.astype(np.float64) * (sc / s)               # [T,out]


def _kv_quant(X, bits, mode):
    """Quantize K or V [T, nkv, hd] -> dequantized [T,nkv,hd] (lossy).

    Granularity x symmetry grid:
      per_vector        scale per (t, head), symmetric
      per_vector_asym   scale+zero-point per (t, head)
      per_channel       scale per (head, channel), symmetric
      per_channel_asym  scale+zero-point per (head, channel)
    """
    if bits is None:
        return X
    hi = (1 << (bits - 1)) - 1
    levels = (1 << bits) - 1
    if mode == "per_vector":
        amax = np.maximum(np.abs(X).max(axis=2, keepdims=True), 1e-9)
        s = hi / amax
        return np.clip(np.round(X * s), -hi - 1, hi) / s
    if mode == "per_vector_asym":
        lo = X.min(axis=2, keepdims=True); up = X.max(axis=2, keepdims=True)
        rng = np.maximum(up - lo, 1e-9); s = levels / rng
        return np.clip(np.round((X - lo) * s), 0, levels) / s + lo
    if mode == "per_channel":
        amax = np.maximum(np.abs(X).max(axis=0, keepdims=True), 1e-9)
        s = hi / amax
        return np.clip(np.round(X * s), -hi - 1, hi) / s
    if mode == "per_channel_asym":
        lo = X.min(axis=0, keepdims=True); up = X.max(axis=0, keepdims=True)
        rng = np.maximum(up - lo, 1e-9); s = levels / rng
        return np.clip(np.round((X - lo) * s), 0, levels) / s + lo
    raise ValueError(mode)


def _rope(x, cos, sin):
    hd = x.shape[-1]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    c = cos[:, None, :]; s = sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def _rope_il(x, pos, hd, base):
    # GPT-NeoX / interleaved rope: rotate adjacent pairs (2i, 2i+1)
    inv = base ** (-(np.arange(0, hd, 2)) / hd)
    ang = np.outer(pos, inv)              # [T, hd/2]
    c = np.cos(ang)[:, None, :]; s = np.sin(ang)[:, None, :]
    xe = x[..., 0::2]; xo = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


def batched_logits(tokens, knobs):
    """Teacher-forced forward over `tokens`; returns logits [T, vocab]."""
    c = w.cfg
    d, hd, nh, nkv = c["d_model"], c["head_dim"], c["n_head"], c["n_kv"]
    eps, base, rep = c["rms_eps"], c["rope_base"], nh // nkv
    T = len(tokens)
    x = w.embed[np.asarray(tokens)].astype(np.float32)
    inv = 1.0 / (base ** (np.arange(0, hd, 2) / hd))
    ang = np.outer(np.arange(T), inv)
    cos, sin = np.cos(ang), np.sin(ang)
    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    scl = 1.0 / np.sqrt(hd)
    for li in range(c["n_layer"]):
        p = f"blk.{li}."
        Wq, sq = w.W(p + "attn_q.weight"); Wk, sk = w.W(p + "attn_k.weight")
        Wv, sv = w.W(p + "attn_v.weight"); Wo, so = w.W(p + "attn_output.weight")
        h = _rms(x, w.norms[p + "attn_norm"], eps)
        q = _bitlin(Wq, sq, h, knobs, 'attn_q').reshape(T, nh, hd)
        k = _bitlin(Wk, sk, h, knobs, 'attn_k').reshape(T, nkv, hd)
        v = _bitlin(Wv, sv, h, knobs, 'attn_v').reshape(T, nkv, hd)
        pdir = knobs.get("pdir", getattr(w, "pdir", 0))
        if pdir == 1:
            q = q.reshape(T, nh, hd // 2, 2).transpose(0, 1, 3, 2).reshape(T, nh, hd)
            k = k.reshape(T, nkv, hd // 2, 2).transpose(0, 1, 3, 2).reshape(T, nkv, hd)
        elif pdir == 2:
            q = q.reshape(T, nh, 2, hd // 2).transpose(0, 1, 3, 2).reshape(T, nh, hd)
            k = k.reshape(T, nkv, 2, hd // 2).transpose(0, 1, 3, 2).reshape(T, nkv, hd)
        bits = knobs.get("kv_bits"); kmode = knobs.get("kv_mode", "per_vector")
        rk = knobs.get("rope", getattr(w, "rope_kind", "hf"))
        rope = (lambda z: _rope_il(z, np.arange(T), hd, base)) if rk == "il" \
               else (lambda z: _rope(z, cos, sin))
        if knobs.get("kv_pre_rope") and bits is not None:
            k = rope(_kv_quant(k, bits, kmode)); q = rope(q)
        else:
            q = rope(q); k = rope(k)
            k = _kv_quant(k, bits, kmode)
        v = _kv_quant(v, bits, kmode)
        kk = np.repeat(k, rep, axis=1); vv = np.repeat(v, rep, axis=1)
        out = np.empty((T, nh, hd), np.float32)
        for hh in range(nh):
            sca = (q[:, hh] @ kk[:, hh].T) * scl + mask
            sca -= sca.max(axis=1, keepdims=True)
            e = np.exp(sca); pr = e / e.sum(axis=1, keepdims=True)
            out[:, hh] = pr @ vv[:, hh]
        attn = out.reshape(T, d)
        if w.has_attn_sub:                       # BitNet subln; absent in Falcon3/Llama
            attn = _rms(attn, w.norms[p + "attn_sub_norm"], eps)
        x = x + _bitlin(Wo, so, attn.astype(np.float32), knobs, 'attn_output').astype(np.float32)
        Wg, sg = w.W(p + "ffn_gate.weight"); Wu, su = w.W(p + "ffn_up.weight")
        Wd, sd = w.W(p + "ffn_down.weight")
        h2 = _rms(x, w.norms[p + "ffn_norm"], eps)
        g = _bitlin(Wg, sg, h2, knobs, 'ffn_gate'); u = _bitlin(Wu, su, h2, knobs, 'ffn_up')
        if w.relu2:
            act = np.square(np.maximum(g, 0.0)) * u            # squared-ReLU (BitNet 2B4T)
        else:
            act = (g / (1.0 + np.exp(-g))) * u                 # SwiGLU / SiLU (Falcon3/Llama)
        if w.has_ffn_sub:
            act = _rms(act.astype(np.float32), w.norms[p + "ffn_sub_norm"], eps)
        x = x + _bitlin(Wd, sd, act.astype(np.float32), knobs, 'ffn_down').astype(np.float32)
    hn = _rms(x, w.norms["output_norm"], eps)
    # chunked lm_head -> logits [T, vocab]; use the untied output.weight if present
    head = w.lm_w if w.lm_w is not None else w.embed   # [vocab, d]
    V = head.shape[0]
    logits = np.empty((T, V), np.float32)
    hn32 = hn.astype(np.float32)
    step = 16384
    for i in range(0, V, step):
        logits[:, i:i + step] = hn32 @ head[i:i + step].astype(np.float32).T
    return logits


def corpus_nll_tokens(corpus_ids, knobs):
    """Return per-token NLL array and a parallel sentence-id array (for bootstrap)."""
    nlls = []; sent = []
    for si, ids in enumerate(corpus_ids):
        lg = batched_logits(ids, knobs)            # [T, vocab]
        for t in range(len(ids) - 1):
            row = lg[t].astype(np.float64)
            m = row.max()
            lse = m + np.log(np.exp(row - m).sum())
            nlls.append(lse - row[ids[t + 1]])
            sent.append(si)
    return np.array(nlls), np.array(sent)


def corpus_nll(corpus_ids, knobs):
    """Sum NLL and token count over a list of id-sequences (each starts with BOS)."""
    nlls, _ = corpus_nll_tokens(corpus_ids, knobs)
    return float(nlls.sum()), int(nlls.size)
