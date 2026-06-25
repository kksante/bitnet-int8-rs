"""
prepare_weights.py — one-time unpack of all I2_S weights into an int8 memmap.

Forwards then read ternary weights straight from the memmap (OS page cache),
avoiding the ~per-call bit-unpack and keeping resident RAM ~1GB. Run once:

    python3 prepare_weights.py

Produces (next to the gguf, in ./_cache/):
    weights.i8        concatenated int8 ternary [out,in] row-major
    meta.npz          name -> (offset, out_features, in_features, scale)
    embed.f16.npy     token_embd as float16 [vocab, d]
    norms.npz         all f32 norm vectors
"""
import os, numpy as np
from gguf_loader import GGUFLoader, unpack_i2s

# Model-agnostic: override for other BitNet GGUF sizes, e.g.
#   BITNET_GGUF=/path/other.gguf BN_CACHE=/tmp/other_cache python3 prepare_weights.py
GGUF = os.environ.get("BITNET_GGUF", "../bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf")
CACHE = os.environ.get("BN_CACHE", "_cache")

PROJ = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
NORMS = ["attn_norm", "ffn_norm", "attn_sub_norm", "ffn_sub_norm"]


def main():
    os.makedirs(CACHE, exist_ok=True)
    ld = GGUFLoader(GGUF)
    cfg = ld.config()
    np.save(os.path.join(CACHE, "embed.f16.npy"),
            ld.load_f16("token_embd.weight").reshape(cfg["vocab"], cfg["d_model"]).astype(np.float16))
    norms = {"output_norm": ld.load_f32("output_norm.weight").astype(np.float32)}
    use_norms = ["attn_norm", "ffn_norm"]
    if cfg["has_attn_sub"]:
        use_norms.append("attn_sub_norm")
    if cfg["has_ffn_sub"]:
        use_norms.append("ffn_sub_norm")
    for i in range(cfg["n_layer"]):
        for nm in use_norms:
            norms[f"blk.{i}.{nm}"] = ld.load_f32(f"blk.{i}.{nm}.weight").astype(np.float32)
    np.savez(os.path.join(CACHE, "norms.npz"), **norms)

    # untied LM head (Falcon3/Llama-style): a separate output.weight. Keep as F16
    # if stored full-precision; if it is itself I2_S ternary, fall back to tied.
    if not cfg["tied"]:
        ot = ld.tensors["output.weight"]
        if ot["dtype"] == 1:  # F16
            np.save(os.path.join(CACHE, "lm_head.f16.npy"),
                    ld.load_f16("output.weight").reshape(cfg["vocab"], cfg["d_model"]).astype(np.float16))
        else:
            print("note: output.weight is not F16 (dtype", ot["dtype"], "); using tied embeddings")

    # total size
    total = 0
    plan = []
    for i in range(cfg["n_layer"]):
        for nm in PROJ:
            t = ld.tensors[f"blk.{i}.{nm}.weight"]
            inf, outf = t["dims"][0], t["dims"][1]
            plan.append((f"blk.{i}.{nm}.weight", inf, outf))
            total += inf * outf
    print(f"total ternary elements: {total} ({total/1e9:.2f}B), int8 bytes {total}")
    mm = np.memmap(os.path.join(CACHE, "weights.i8"), dtype=np.int8, mode="w+", shape=(total,))
    meta = {}
    off = 0
    for name, inf, outf in plan:
        packed, scale, dims = ld.load_i2s_packed(name)
        w = unpack_i2s(packed, inf * outf)  # flat [out*in], row-major [out,in]
        mm[off:off + inf * outf] = w
        meta[name] = np.array([off, outf, inf], dtype=np.int64)
        meta[name + ".scale"] = np.array([scale], dtype=np.float32)
        off += inf * outf
    mm.flush()
    np.savez(os.path.join(CACHE, "meta.npz"), **meta)
    cfgarr = {k: np.array([v]) for k, v in cfg.items() if isinstance(v, (int, float))}
    np.savez(os.path.join(CACHE, "cfg.npz"), **cfgarr)
    print("done. cache at", os.path.abspath(CACHE))


if __name__ == "__main__":
    main()
