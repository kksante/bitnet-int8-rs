"""
gguf_loader.py — Minimal, verified GGUF reader for BitNet b1.58 2B4T (I2_S).

This is a *reference* loader written to establish ground truth for the Rust
implementation. It is deliberately explicit and dependency-light (numpy only).

Verified facts about this model's on-disk format (confirmed against the bytes):

  * General: GGUF v3, little-endian, 332 tensors, alignment = 32 bytes.
  * Architecture `bitnet-b1.58`: 30 blocks, dim 2560, ffn 6912,
    20 query heads / 5 KV heads (GQA), head_dim 128, rope base 500000,
    rms eps 1e-5, vocab 128256, context 4096.
  * token_embd.weight is F16, the four per-block norms + final norm are F32.
  * The 7 projections per block (q,k,v,o,gate,up,down) are I2_S (type 36).

I2_S layout (per tensor), confirmed empirically:
  [ ceil(n/4) bytes of 2-bit codes, 4 values/byte, LSB-first ]
  [ 1 little-endian f32 scale ]
  [ zero/garbage padding up to 32-byte alignment ]

  Decoded value = code - 1, i.e. 00->-1, 01->0, 10->+1, 11 unused.
  Dequantized weight = (code - 1) * scale.

  This mapping was selected because it yields the symmetric, ~50%-sparse
  ternary distribution (mean ~= 0) characteristic of BitNet; the alternative
  mapping used by the previous Rust code (0->0,1->+1,2->-1) produced a biased,
  25%-sparse distribution and is wrong.
"""

import struct
import numpy as np

GGUF_MAGIC = 0x46554747

# ggml type ids we care about
GGML_F32 = 0
GGML_F16 = 1
GGML_I2_S = 36

_VALSCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
              6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}


class GGUFLoader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.metadata = {}
        self.tensors = {}   # name -> dict(dims, dtype, offset)
        self._parse_header()

    # ---- header parsing -------------------------------------------------
    def _rd(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, self.f.read(n))

    def _rstr(self):
        (l,) = self._rd("Q")
        return self.f.read(l).decode("utf-8", "replace")

    def _rval(self, t):
        if t == 8:
            return self._rstr()
        if t == 9:  # array
            (et,) = self._rd("I")
            (cnt,) = self._rd("Q")
            return [self._rval(et) for _ in range(cnt)]
        return self._rd(_VALSCALAR[t])[0]

    def _parse_header(self):
        f = self.f
        (magic,) = self._rd("I")
        if magic != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        (self.version,) = self._rd("I")
        (n_tensors,) = self._rd("Q")
        (n_meta,) = self._rd("Q")
        for _ in range(n_meta):
            k = self._rstr()
            (t,) = self._rd("I")
            self.metadata[k] = self._rval(t)
        for _ in range(n_tensors):
            name = self._rstr()
            (ndim,) = self._rd("I")
            dims = [self._rd("Q")[0] for _ in range(ndim)]
            (dtype,) = self._rd("I")
            (offset,) = self._rd("Q")
            self.tensors[name] = dict(dims=dims, dtype=dtype, offset=offset)
        pos = f.tell()
        align = self.metadata.get("general.alignment", 32)
        self.data_offset = (pos + align - 1) // align * align

    # ---- config ---------------------------------------------------------
    def config(self):
        m = self.metadata
        # arch-agnostic: derive the metadata key prefix from general.architecture
        # so any BitNet-family GGUF (other sizes) loads without code changes.
        a = m.get("general.architecture", "bitnet-b1.58")
        # architecture profile flags, detected from tensor presence + arch name.
        t = self.tensors
        has_attn_sub = "blk.0.attn_sub_norm.weight" in t
        has_ffn_sub = "blk.0.ffn_sub_norm.weight" in t
        tied = "output.weight" not in t
        # squared-ReLU FFN for BitNet's 2B4T arch; SiLU/SwiGLU for llama/falcon-style.
        relu2 = ("bitnet" in a.lower()) and has_ffn_sub
        return dict(
            arch=a, has_attn_sub=has_attn_sub, has_ffn_sub=has_ffn_sub,
            tied=tied, relu2=relu2,
            **self._dims(a, m),
        )

    def _dims(self, a, m):
        return dict(
            n_layer=m[f"{a}.block_count"],
            d_model=m[f"{a}.embedding_length"],
            d_ff=m[f"{a}.feed_forward_length"],
            n_head=m[f"{a}.attention.head_count"],
            n_kv=m[f"{a}.attention.head_count_kv"],
            head_dim=m[f"{a}.rope.dimension_count"],
            rope_base=m[f"{a}.rope.freq_base"],
            rms_eps=m[f"{a}.attention.layer_norm_rms_epsilon"],
            vocab=m[f"{a}.vocab_size"],
            context=m[f"{a}.context_length"],
            bos=m.get("tokenizer.ggml.bos_token_id"),
            eos=m.get("tokenizer.ggml.eos_token_id"),
        )

    # ---- raw tensor bytes ----------------------------------------------
    def _n_elements(self, t):
        n = 1
        for d in t["dims"]:
            n *= d
        return n

    def _read_bytes(self, t, nbytes):
        self.f.seek(self.data_offset + t["offset"])
        return self.f.read(nbytes)

    # ---- typed tensor access -------------------------------------------
    def load_f32(self, name):
        t = self.tensors[name]
        ne = self._n_elements(t)
        raw = self._read_bytes(t, ne * 4)
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        return arr  # 1-D; caller reshapes if needed

    def load_f16(self, name):
        """Return float32 array (dims are ggml order: dims[0] is innermost)."""
        t = self.tensors[name]
        ne = self._n_elements(t)
        raw = self._read_bytes(t, ne * 2)
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        return arr

    def load_i2s_packed(self, name):
        """Return (packed_uint8, scale_float, dims).

        Keeps weights packed (2 bits/value) to save memory. Use unpack_i2s()
        to expand a row-major [out, in] int8 ternary matrix on demand.
        ggml dims are [in, out] (dims[0] innermost = in_features)."""
        t = self.tensors[name]
        ne = self._n_elements(t)
        code_bytes = (ne + 3) // 4
        self.f.seek(self.data_offset + t["offset"])
        packed = np.frombuffer(self.f.read(code_bytes), dtype=np.uint8).copy()
        (scale,) = struct.unpack("<f", self.f.read(4))
        return packed, float(scale), list(t["dims"])

    def tensor_type(self, name):
        return self.tensors[name]["dtype"]


# I2_S block de-interleave permutation (verified empirically against the model).
#
# Each 32-byte block encodes 128 ternary elements. For byte j in 0..31 and
# 2-bit slot s in 0..3 (slot s occupies bits [2s, 2s+1], LSB-first), the decoded
# element index within the block is:
#
#     element = (3 - s) * 32 + j
#
# i.e. the 4 groups of 32 are stored MSB-slot-first. This exact scheme is the
# one (out of the plausible SIMD interleavings) that makes the model generate
# coherent text; sequential packing or the non-reversed group order produce
# the correct ternary *distribution* but scrambled positions -> gibberish.
_I2S_PERM = np.empty(128, dtype=np.int64)
for _j in range(32):
    for _s in range(4):
        _I2S_PERM[(3 - _s) * 32 + _j] = 4 * _j + _s


def unpack_i2s(packed, n):
    """Expand packed 2-bit I2_S codes to int8 ternary {-1,0,+1}.

    Handles the 128-element / 32-byte block interleaving. value = code - 1.
    Returns a flat int8 array of length n (n must be a multiple of 128 up to
    the final partial block, which is fine for this model: all dims % 128 == 0).
    """
    nb = packed.size // 32
    b = packed[: nb * 32].reshape(nb, 32).astype(np.int16)
    seq = np.empty((nb, 128), dtype=np.int16)
    for s in range(4):
        seq[:, s::4] = (b >> (2 * s)) & 0b11   # seq[:, 4j + s]
    el = seq[:, _I2S_PERM].reshape(-1)[:n]
    return (el.astype(np.int8) - 1)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "../bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    ld = GGUFLoader(path)
    cfg = ld.config()
    print("config:", cfg)
    # validate a few ternary tensors
    for name in ["blk.0.attn_q.weight", "blk.0.ffn_down.weight",
                 "blk.15.attn_v.weight", "blk.29.ffn_up.weight"]:
        packed, scale, dims = ld.load_i2s_packed(name)
        ne = int(np.prod(dims))
        codes = unpack_i2s(packed, ne)
        frac_zero = float((codes == 0).mean())
        print(f"{name:28s} dims={dims} scale={scale:.4f} "
              f"mean={codes.mean():+.4f} frac_zero={frac_zero:.3f} "
              f"min={codes.min()} max={codes.max()}")
