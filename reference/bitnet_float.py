"""
bitnet_float.py — Faithful *float* reference forward pass for BitNet b1.58 2B4T.

This is the ground-truth oracle. It computes exactly what the model defines:
ternary weights dequantized to float, per-token int8 activation quantization
(the model is W1.58A8 — activation quantization is part of the trained model,
not an approximation), and everything else (RMSNorm, RoPE, softmax, residual
accumulation) in float64/float32.

The integer-path implementation (bitnet_int.py) mirrors this op-for-op but in
fixed-point integer arithmetic, so the *only* differences are the integer
approximations of the non-matmul ops. (The ternary x int8 matmul itself is
exactly representable in int32, so it contributes no error.)

Weights are kept packed (2 bits/value) and unpacked per matmul to fit in RAM.
"""

import numpy as np
from gguf_loader import GGUFLoader, unpack_i2s


def rmsnorm(x, w, eps):
    # x: [T, d] float; w: [d]
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    xn = x / np.sqrt(var + eps)
    return (xn * w).astype(np.float32)


def activation_quant(x):
    """Per-token (per-row) absmax int8 quantization. Returns (xq_int8_float, scale).

    scale = 127 / max(|x|) per row; xq = round(x*scale) clamped to [-128,127].
    Dequantized activation = xq / scale.
    """
    s = 127.0 / np.clip(np.abs(x).max(axis=-1, keepdims=True), 1e-5, None)
    xq = np.clip(np.round(x * s), -128, 127)
    return xq, s


class BitLinearF:
    """Float reference of a BitLinear projection (ternary weight x int8 act)."""

    def __init__(self, loader, name):
        self.packed, self.scale, dims = loader.load_i2s_packed(name)
        self.in_f, self.out_f = dims[0], dims[1]
        self.ne = self.in_f * self.out_f

    def W(self):
        # [out, in] int8 ternary
        return unpack_i2s(self.packed, self.ne).reshape(self.out_f, self.in_f)

    def __call__(self, x):
        # x: [T, in] float
        xq, s = activation_quant(x)
        acc = xq @ self.W().T.astype(np.float32)        # [T, out], integer-valued
        return (acc * (self.scale / s)).astype(np.float32)


def rope_tables(positions, head_dim, base):
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))  # [hd/2]
    ang = np.outer(positions, inv_freq)            # [T, hd/2]
    emb = np.concatenate([ang, ang], axis=-1)      # [T, hd]
    return np.cos(emb), np.sin(emb)


def apply_rope(x, cos, sin):
    # x: [T, n_head, hd]; cos/sin: [T, hd]
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    rot = np.concatenate([-x2, x1], axis=-1)
    c = cos[:, None, :]
    s = sin[:, None, :]
    return x * c + rot * s


class BitNetFloat:
    def __init__(self, path, capture=None):
        self.ld = GGUFLoader(path)
        self.cfg = self.ld.config()
        c = self.cfg
        self.n_layer = c["n_layer"]
        self.d = c["d_model"]
        self.n_head = c["n_head"]
        self.n_kv = c["n_kv"]
        self.hd = c["head_dim"]
        self.eps = c["rms_eps"]
        self.base = c["rope_base"]
        self.embed = self.ld.load_f16("token_embd.weight").reshape(c["vocab"], self.d)
        self.out_norm = self.ld.load_f32("output_norm.weight")
        self.layers = []
        for i in range(self.n_layer):
            p = f"blk.{i}."
            self.layers.append(dict(
                attn_norm=self.ld.load_f32(p + "attn_norm.weight"),
                ffn_norm=self.ld.load_f32(p + "ffn_norm.weight"),
                attn_sub=self.ld.load_f32(p + "attn_sub_norm.weight"),
                ffn_sub=self.ld.load_f32(p + "ffn_sub_norm.weight"),
                q=BitLinearF(self.ld, p + "attn_q.weight"),
                k=BitLinearF(self.ld, p + "attn_k.weight"),
                v=BitLinearF(self.ld, p + "attn_v.weight"),
                o=BitLinearF(self.ld, p + "attn_output.weight"),
                gate=BitLinearF(self.ld, p + "ffn_gate.weight"),
                up=BitLinearF(self.ld, p + "ffn_up.weight"),
                down=BitLinearF(self.ld, p + "ffn_down.weight"),
            ))
        self.capture = capture  # optional dict-list to record per-layer activations

    def forward(self, tokens):
        """Full (prefill) forward over a token list. Returns logits for last pos.

        Simple O(T^2) attention over the whole sequence (no incremental cache);
        used for the oracle. The integer path implements the KV-cache variant.
        """
        c = self.cfg
        T = len(tokens)
        x = self.embed[np.asarray(tokens)].astype(np.float32)   # [T, d]
        pos = np.arange(T)
        cos, sin = rope_tables(pos, self.hd, self.base)
        rep = self.n_head // self.n_kv
        cap = self.capture
        for li, L in enumerate(self.layers):
            # ---- attention ----
            h = rmsnorm(x, L["attn_norm"], self.eps)
            q = L["q"](h).reshape(T, self.n_head, self.hd)
            k = L["k"](h).reshape(T, self.n_kv, self.hd)
            v = L["v"](h).reshape(T, self.n_kv, self.hd)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            # GQA expand
            k = np.repeat(k, rep, axis=1)
            v = np.repeat(v, rep, axis=1)
            scale = 1.0 / np.sqrt(self.hd)
            out = np.empty((T, self.n_head, self.hd), dtype=np.float32)
            mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
            for hh in range(self.n_head):
                scores = (q[:, hh, :] @ k[:, hh, :].T) * scale + mask  # [T,T]
                scores -= scores.max(axis=-1, keepdims=True)
                e = np.exp(scores)
                p = e / e.sum(axis=-1, keepdims=True)
                out[:, hh, :] = p @ v[:, hh, :]
            attn = out.reshape(T, self.d)
            attn = rmsnorm(attn, L["attn_sub"], self.eps)
            attn = L["o"](attn)
            x = x + attn
            # ---- ffn ----
            h2 = rmsnorm(x, L["ffn_norm"], self.eps)
            g = L["gate"](h2)
            u = L["up"](h2)
            act = np.square(np.maximum(g, 0.0)) * u
            act = rmsnorm(act, L["ffn_sub"], self.eps)
            d = L["down"](act)
            x = x + d
            if cap is not None:
                cap.append(x[-1].copy())
        hn = rmsnorm(x, self.out_norm, self.eps)
        logits = hn[-1] @ self.embed.T
        return logits


if __name__ == "__main__":
    import sys, time
    from tokenizers import Tokenizer
    path = "../bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    tok = Tokenizer.from_file("../bitnet-b1.58-2B-4T/tokenizer.json")
    model = BitNetFloat(path)
    bos = model.cfg["bos"]
    eos = model.cfg["eos"]
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    ids = [bos] + tok.encode(prompt).ids
    n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(f"prompt: {prompt!r}\nids: {ids}")
    t0 = time.time()
    for step in range(n_new):
        logits = model.forward(ids)
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        if nxt == eos:
            print("[eos]")
            break
        print(f"  step {step:2d}: tok={nxt:6d} {tok.decode([nxt])!r}  "
              f"(top logit {logits[nxt]:.2f})", flush=True)
    print(f"\nGENERATION ({time.time()-t0:.1f}s):")
    print(tok.decode(ids[1:]))
