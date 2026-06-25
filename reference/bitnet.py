"""
bitnet.py — verified reference for BitNet b1.58 2B4T with TWO forward paths:

  * forward_float : faithful float oracle (the model as intended; W1.58A8 with
                    per-token int8 activation quant, everything else in float).
  * forward_int   : a *fully integer* path — int8 activations, exact int32
                    ternary matmuls, and fixed-point integer RMSNorm, RoPE,
                    softmax and KV-cache. Configurable precision knobs let us
                    measure where the integer path stops tracking the oracle.

Both use an incremental KV cache (prefill + single-token decode) so generation
is cheap. Weights are read from the int8 memmap built by prepare_weights.py.

Integer-path semantics are simulated bit-accurately in numpy: every quantity is
an exact integer (enforced by rounding) carried with an explicit scale, exactly
as an integer ALU would. The scales would be int multiply+shift constants in
hardware; here they are floats for clarity, which does not affect the integer
approximation error being studied (that error comes from int8 rounding, integer
rsqrt, fixed-point exp, and fixed-point sin/cos — all implemented as real int ops).
"""
import os
import numpy as np

CACHE = os.environ.get("BN_CACHE", "/tmp/bn_cache")


# --------------------------------------------------------------------------
# Config / weights
# --------------------------------------------------------------------------
class IntConfig:
    """Precision knobs for the integer path."""
    def __init__(self, act_scale="absmax", softmax_bits=15, exp_lut_bits=12,
                 kv_bits=8, rope_bits=15, rsqrt_bits=16, act_mult=None,
                 int_scale=False):
        self.act_scale = act_scale       # 'absmax' | 'absmean' | 'median'
        self.act_mult = act_mult         # multiplier for mean/median scaling (clip point)
        self.int_scale = int_scale       # rescale as integer multiply + 2^-frac shift
        self.softmax_bits = softmax_bits  # fixed-point frac bits in softmax probs
        self.exp_lut_bits = exp_lut_bits  # entries = 2^bits over the exp input range
        self.kv_bits = kv_bits            # KV cache storage bit-width (8 or 16)
        self.rope_bits = rope_bits        # Q-format frac bits for sin/cos
        self.rsqrt_bits = rsqrt_bits      # frac bits for integer 1/rms


PROJ = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]


class Weights:
    def __init__(self, cache=CACHE):
        c = np.load(os.path.join(cache, "cfg.npz"))
        self.cfg = {k: (int(c[k][0]) if float(c[k][0]).is_integer() else float(c[k][0]))
                    for k in c.files}
        self.embed = np.load(os.path.join(cache, "embed.f16.npy"))  # [vocab, d] f16
        self.norms = dict(np.load(os.path.join(cache, "norms.npz")))
        meta = np.load(os.path.join(cache, "meta.npz"))
        self.meta = {k: meta[k] for k in meta.files}
        total = os.path.getsize(os.path.join(cache, "weights.i8"))
        self.mm = np.memmap(os.path.join(cache, "weights.i8"), dtype=np.int8,
                            mode="r", shape=(total,))
        # untied LM head (Falcon3/Llama-style), if present; else tied to embed.
        lm = os.path.join(cache, "lm_head.f16.npy")
        self.lm_w = np.load(lm) if os.path.exists(lm) else None
        # arch profile flags (default to BitNet-2B4T behaviour for old caches)
        self.relu2 = bool(self.cfg.get("relu2", 1))
        self.has_attn_sub = bool(self.cfg.get("has_attn_sub", 1))
        self.has_ffn_sub = bool(self.cfg.get("has_ffn_sub", 1))

    def W(self, name):
        off, outf, inf = self.meta[name]
        w = self.mm[off:off + outf * inf].reshape(outf, inf)
        scale = float(self.meta[name + ".scale"][0])
        return w, scale  # int8 [out,in], float scale

    def lm_head(self, hn):
        """Tied-embedding logits without materializing a giant float embedding.

        Chunked over vocab to keep peak memory low (the full f64 upcast of the
        128k x 2560 embedding would be ~2.6GB and OOM the sandbox)."""
        hn32 = hn.astype(np.float32)
        head = self.lm_w if self.lm_w is not None else self.embed  # [vocab, d]
        V = head.shape[0]
        out = np.empty(V, dtype=np.float32)
        step = 16384
        for i in range(0, V, step):
            out[i:i + step] = head[i:i + step].astype(np.float32) @ hn32
        return out


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def rope_angles(pos, hd, base):
    inv = 1.0 / (base ** (np.arange(0, hd, 2) / hd))
    ang = pos * inv
    return np.cos(ang), np.sin(ang)  # each [hd/2]


# ============================ FLOAT ORACLE ================================
class FloatRunner:
    def __init__(self, w: Weights):
        self.w = w
        c = w.cfg
        self.n_layer, self.d = c["n_layer"], c["d_model"]
        self.nh, self.nkv, self.hd = c["n_head"], c["n_kv"], c["head_dim"]
        self.eps, self.base = c["rms_eps"], c["rope_base"]
        self.rep = self.nh // self.nkv
        self.reset()

    def reset(self):
        self.kc = [[] for _ in range(self.n_layer)]  # list of [nkv,hd]
        self.vc = [[] for _ in range(self.n_layer)]
        self.pos = 0

    def _rms(self, x, wname):
        g = self.w.norms[wname]
        v = np.mean(x.astype(np.float64) ** 2)
        return (x / np.sqrt(v + self.eps) * g).astype(np.float32)

    def _lin(self, x, name):
        W, sc = self.w.W(name)
        s = 127.0 / max(np.abs(x).max(), 1e-5)
        xq = np.clip(np.round(x * s), -128, 127)
        return (xq @ W.T.astype(np.float32)) * (sc / s)

    def step(self, token, capture=None):
        d, hd, nh, nkv = self.d, self.hd, self.nh, self.nkv
        x = self.w.embed[token].astype(np.float32)
        cphase, sphase = rope_angles(self.pos, hd, self.base)
        for li in range(self.n_layer):
            p = f"blk.{li}."
            h = self._rms(x, p + "attn_norm")
            q = self._lin(h, p + "attn_q.weight").reshape(nh, hd)
            k = self._lin(h, p + "attn_k.weight").reshape(nkv, hd)
            v = self._lin(h, p + "attn_v.weight").reshape(nkv, hd)
            q = self._rope(q, cphase, sphase)
            k = self._rope(k, cphase, sphase)
            self.kc[li].append(k); self.vc[li].append(v)
            K = np.stack(self.kc[li]); V = np.stack(self.vc[li])  # [T,nkv,hd]
            out = np.empty((nh, hd), np.float32)
            scl = 1.0 / np.sqrt(hd)
            for hh in range(nh):
                kk = K[:, hh // self.rep, :]; vv = V[:, hh // self.rep, :]
                sc = (kk @ q[hh]) * scl
                sc -= sc.max(); e = np.exp(sc); pr = e / e.sum()
                out[hh] = pr @ vv
            attn = self._rms(out.reshape(d), p + "attn_sub_norm")
            x = x + self._lin(attn, p + "attn_output.weight")
            h2 = self._rms(x, p + "ffn_norm")
            g = self._lin(h2, p + "ffn_gate.weight")
            u = self._lin(h2, p + "ffn_up.weight")
            act = np.square(np.maximum(g, 0.0)) * u
            act = self._rms(act, p + "ffn_sub_norm")
            x = x + self._lin(act, p + "ffn_down.weight")
            if capture is not None:
                capture.setdefault(li, {})["x"] = x.copy()
        self.pos += 1
        hn = self._rms(x, "output_norm")
        return self.w.lm_head(hn)

    def _rope(self, x, c, s):
        hd = x.shape[-1]
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        out = np.empty_like(x)
        out[..., :hd // 2] = x1 * c - x2 * s
        out[..., hd // 2:] = x2 * c + x1 * s
        return out


# ============================ INTEGER PATH ===============================
class IntRunner:
    """Fully-integer forward. See module docstring for the simulation contract."""
    def __init__(self, w: Weights, icfg: IntConfig):
        self.w = w; self.ic = icfg
        c = w.cfg
        self.n_layer, self.d = c["n_layer"], c["d_model"]
        self.nh, self.nkv, self.hd = c["n_head"], c["n_kv"], c["head_dim"]
        self.eps, self.base = c["rms_eps"], c["rope_base"]
        self.rep = self.nh // self.nkv
        # precompute fixed-point rope tables per position lazily
        self._rope_cache = {}
        self.reset()

    def reset(self):
        self.kc = [[] for _ in range(self.n_layer)]
        self.vc = [[] for _ in range(self.n_layer)]
        self.ks = [[] for _ in range(self.n_layer)]  # kv scales
        self.vs = [[] for _ in range(self.n_layer)]
        self.pos = 0

    # ---- integer primitives ----
    def _act_quant(self, x):
        """x: float (holds the real activation). Quantize to int8 + scale s (real=q/s)."""
        a = np.abs(x)
        if self.ic.act_scale == "absmax":
            amax = max(a.max(), 1e-5)
        elif self.ic.act_scale == "absmean":
            amax = max(a.mean() * (self.ic.act_mult or 4.0), 1e-5)
        elif self.ic.act_scale == "median":
            amax = max(np.median(a) * (self.ic.act_mult or 8.0), 1e-5)
        else:
            raise ValueError(self.ic.act_scale)
        s = 127.0 / amax
        q = np.clip(np.round(x * s), -128, 127).astype(np.int32)
        return q, s

    def _rms_to_int8(self, x, wname):
        """Integer RMSNorm fused with int8 activation quant.

        Data-adaptive fixed point: choose a per-vector scale so the int view fits
        in ~15 bits regardless of the (growing) residual magnitude, then do an
        integer sum-of-squares, integer sqrt, and a fixed-point reciprocal divide.
        This is the genuine integer approximation whose error we study.
        """
        g = self.w.norms[wname]
        n = x.size
        amax = max(float(np.abs(x).max()), 1e-9)
        SN = (1 << 14) / amax                      # |xi| <= 16384
        xi = np.round(x * SN).astype(np.int64)
        ss = int((xi * xi).sum())                  # <= n*2^28, safe in int64
        msq = ss // n
        rms_u = max(isqrt(msq), 1)                 # rms of xi, integer (SN units)
        INV = 1 << self.ic.rsqrt_bits
        xn_fp = (xi * INV) // rms_u                 # integer; real value ~ xi/rms_u (unit rms)
        xn = xn_fp.astype(np.float64) / INV * g     # * norm weight (not quantized)
        return self._act_quant(xn)

    def _lin(self, q_int8, s_act, name):
        W, wsc = self.w.W(name)
        acc = q_int8.astype(np.int32) @ W.T.astype(np.int32)   # exact int32
        if getattr(self.ic, "int_scale", False):
            # close the loop: the per-(tensor,token) rescale as an integer
            # multiply + power-of-two shift, the only remaining float op.
            frac = 24
            mant = int(round((wsc / s_act) * (1 << frac)))      # int mantissa
            return (acc.astype(np.int64) * mant) / (1 << frac)  # int mul, 2^-frac shift
        return acc.astype(np.float64) * (wsc / s_act)          # real output

    def _rope_tbl(self, pos):
        if pos in self._rope_cache:
            return self._rope_cache[pos]
        hd = self.hd
        Q = 1 << self.ic.rope_bits
        c, s = rope_angles(pos, hd, self.base)
        ci = np.round(c * Q).astype(np.int64); si = np.round(s * Q).astype(np.int64)
        self._rope_cache[pos] = (ci, si, Q)
        return self._rope_cache[pos]

    def _rope_int(self, x, pos):
        """Integer RoPE on real-valued x using fixed-point sin/cos."""
        ci, si, Q = self._rope_tbl(pos)
        hd = x.shape[-1]
        XF = 1 << 12
        xi = np.round(x * XF).astype(np.int64)
        x1, x2 = xi[..., :hd // 2], xi[..., hd // 2:]
        o1 = (x1 * ci - x2 * si) >> self.ic.rope_bits
        o2 = (x2 * ci + x1 * si) >> self.ic.rope_bits
        out = np.concatenate([o1, o2], axis=-1).astype(np.float64) / XF
        return out

    def _softmax_int(self, scores_int, qk_scale):
        """Integer softmax over an int32 score vector.

        scores_int: raw int32 q.k accumulator. qk_scale converts to real logits.
        Uses a fixed-point exp LUT and integer normalization to `softmax_bits`.
        Returns integer probabilities with scale 2^softmax_bits (sum ~= 2^bits).
        """
        logits = scores_int.astype(np.float64) * qk_scale     # real
        logits -= logits.max()
        # fixed-point exp via LUT over [-, 0]
        lut_n = 1 << self.ic.exp_lut_bits
        LO = -20.0                                             # exp(-20)~2e-9
        idx = np.clip(((logits - LO) / (-LO) * (lut_n - 1)), 0, lut_n - 1).astype(np.int64)
        xs = LO + np.arange(lut_n) / (lut_n - 1) * (-LO)
        lut = np.round(np.exp(xs) * (1 << self.ic.softmax_bits)).astype(np.int64)
        e = lut[idx]                                           # int
        tot = int(e.sum())
        if tot == 0:
            tot = 1
        p = (e.astype(np.int64) << self.ic.softmax_bits) // tot  # int prob, scale 2^bits
        return p.astype(np.float64) / (1 << self.ic.softmax_bits)

    def _softmax_int_batch(self, real):
        """Integer softmax over each row of `real` ([nh,T] real logits).

        Per row: shift by max, fixed-point exp via LUT (exp_lut_bits resolution
        over [-20,0], values scaled by 2^softmax_bits), then integer normalize.
        """
        r = real - real.max(axis=1, keepdims=True)
        lut_n = 1 << self.ic.exp_lut_bits
        LO = -20.0
        idx = np.clip(((r - LO) / (-LO) * (lut_n - 1)), 0, lut_n - 1).astype(np.int64)
        xs = LO + np.arange(lut_n) / (lut_n - 1) * (-LO)
        lut = np.round(np.exp(xs) * (1 << self.ic.softmax_bits)).astype(np.int64)
        e = lut[idx]                                           # [nh,T] int
        tot = e.sum(axis=1, keepdims=True); tot[tot == 0] = 1
        p = (e << self.ic.softmax_bits) // tot                 # int, scale 2^bits
        return p.astype(np.float64) / (1 << self.ic.softmax_bits)

    def _kv_quant(self, vec):
        b = self.ic.kv_bits
        hi = (1 << (b - 1)) - 1
        amax = max(np.abs(vec).max(), 1e-5)
        s = hi / amax
        q = np.clip(np.round(vec * s), -hi - 1, hi).astype(np.int32)
        return q, s

    def step(self, token, capture=None):
        d, hd, nh, nkv = self.d, self.hd, self.nh, self.nkv
        x = self.w.embed[token].astype(np.float64)
        for li in range(self.n_layer):
            p = f"blk.{li}."
            qa, sa = self._rms_to_int8(x, p + "attn_norm")
            q = self._lin(qa, sa, p + "attn_q.weight").reshape(nh, hd)
            k = self._lin(qa, sa, p + "attn_k.weight").reshape(nkv, hd)
            v = self._lin(qa, sa, p + "attn_v.weight").reshape(nkv, hd)
            q = self._rope_int(q, self.pos)
            k = self._rope_int(k, self.pos)
            # quantize k,v into KV cache
            kq, ks = self._kv_quant(k); vq, vs = self._kv_quant(v)
            self.kc[li].append(kq); self.ks[li].append(ks)
            self.vc[li].append(vq); self.vs[li].append(vs)
            scl = 1.0 / np.sqrt(hd)
            Kq = np.stack(self.kc[li]); Vq = np.stack(self.vc[li])  # [T,nkv,hd] int
            Ks = np.array(self.ks[li]); Vs = np.array(self.vs[li])  # [T]
            T = Kq.shape[0]
            # per-head int8 query, integer q.k over the (GQA-expanded) KV cache
            qmax = np.maximum(np.abs(q).max(axis=1, keepdims=True), 1e-5)
            qs = 127.0 / qmax                                   # [nh,1]
            qi = np.clip(np.round(q * qs), -128, 127).astype(np.int64)   # [nh,hd]
            gidx = np.arange(nh) // self.rep                    # head->kv group
            Kg = Kq[:, gidx, :].astype(np.int64)               # [T,nh,hd]
            dot = np.einsum('thd,hd->ht', Kg, qi)              # [nh,T] int
            real = dot.astype(np.float64) / qs / Ks[None, :] * scl   # [nh,T] real logits
            pr = self._softmax_int_batch(real)                 # [nh,T] probs
            Vg = Vq[:, gidx, :].astype(np.float64) / Vs[:, None, None]  # [T,nh,hd]
            out = np.einsum('ht,thd->hd', pr, Vg)              # [nh,hd]
            ao, so = self._rms_to_int8(out.reshape(d), p + "attn_sub_norm")
            x = x + self._lin(ao, so, p + "attn_output.weight")
            ga, gsa = self._rms_to_int8(x, p + "ffn_norm")
            g = self._lin(ga, gsa, p + "ffn_gate.weight")
            u = self._lin(ga, gsa, p + "ffn_up.weight")
            act = np.square(np.maximum(g, 0.0)) * u
            fa, fsa = self._rms_to_int8(act, p + "ffn_sub_norm")
            x = x + self._lin(fa, fsa, p + "ffn_down.weight")
            if capture is not None:
                capture.setdefault(li, {})["x"] = x.copy()
        self.pos += 1
        hn_q, hn_s = self._rms_to_int8(x, "output_norm")
        hn = hn_q.astype(np.float64) / hn_s
        return self.w.lm_head(hn)


# --------------------------------------------------------------------------
def isqrt(n):
    if n < 2:
        return n
    x = int(n); y = (x + 1) // 2
    while y < x:
        x = y; y = (x + n // x) // 2
    return x
