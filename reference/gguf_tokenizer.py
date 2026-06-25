"""Build a HuggingFace `tokenizers` BPE tokenizer from the vocab+merges embedded
in a GGUF (gpt2 byte-level BPE), for models whose GGUF repo ships no tokenizer.json
(e.g. the Falcon3-1.58bit GGUFs). Saves <out> and prints bos/eos.

  python3 gguf_tokenizer.py ../models/falcon3-1b/ggml-model-i2_s.gguf ../models/falcon3-1b/tokenizer.json
"""
import sys
from gguf_loader import GGUFLoader
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors


def build(gguf_path, out_path):
    ld = GGUFLoader(gguf_path)
    m = ld.metadata
    model = m.get("tokenizer.ggml.model")
    tokens = m["tokenizer.ggml.tokens"]
    merges_raw = m.get("tokenizer.ggml.merges", [])
    assert model == "gpt2", f"only gpt2 BPE supported here, got {model!r}"
    vocab = {tok: i for i, tok in enumerate(tokens)}
    merges = []
    for s in merges_raw:
        parts = s.split(" ")
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))
    bpe = models.BPE(vocab=vocab, merges=merges, fuse_unk=False,
                     byte_fallback=False)
    tk = Tokenizer(bpe)
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.ByteLevel(trim_offsets=True)
    tk.save(out_path)
    bos = m.get("tokenizer.ggml.bos_token_id")
    eos = m.get("tokenizer.ggml.eos_token_id")
    print(f"saved {out_path}  vocab={len(vocab)} merges={len(merges)} bos={bos} eos={eos}")
    return tk, bos, eos


if __name__ == "__main__":
    g, o = sys.argv[1], sys.argv[2]
    tk, bos, eos = build(g, o)
    # round-trip sanity
    s = "The capital of France is Paris."
    enc = tk.encode(s, add_special_tokens=False)
    print("ids:", enc.ids[:16])
    print("roundtrip:", repr(tk.decode(enc.ids)))
