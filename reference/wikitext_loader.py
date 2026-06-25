"""Load a real held-out text corpus (e.g. WikiText-2 raw) from a local file into
token-id chunks for ppl.corpus_nll. No network needed — point it at a file you
downloaded into the repo.

  WikiText-2 raw test set is typically `wikitext-2-raw/wiki.test.raw`.
  Download it on your machine (it is a few MB) and drop it in the repo, then:

    BN_CORPUS=wikitext-2-raw/wiki.test.raw python3 run_ppl2.py 0 4 9 11

By default chunks the stream into non-overlapping windows of `chunk_tokens`
tokens (each prefixed with BOS). For a quick in-sandbox slice use BN_MAX_CHUNKS.
"""
import os


def load_text_corpus(path, tokenizer, bos, chunk_tokens=256, max_chunks=None,
                     chunk_offset=0, min_chars=1):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # WikiText keeps blank lines and " = Heading = " markers; keep it simple and
    # score the raw stream (standard practice is to concatenate and window it).
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    all_chunks = [ids[i:i + chunk_tokens] for i in range(0, len(ids), chunk_tokens)]
    all_chunks = [c for c in all_chunks if len(c) >= 2]
    sel = all_chunks[chunk_offset:]
    if max_chunks:
        sel = sel[:max_chunks]
    pre = [bos] if bos is not None else []
    return [pre + c for c in sel]


def corpus_from_env(tokenizer, bos):
    """Return (corpus_ids, name). Uses BN_CORPUS file if set, else the synthetic
    mini-corpus from corpus.py."""
    path = os.environ.get("BN_CORPUS")
    if path:
        ct = int(os.environ.get("BN_CHUNK_TOKENS", "256"))
        mc = os.environ.get("BN_MAX_CHUNKS")
        off = int(os.environ.get("BN_CHUNK_OFFSET", "0"))
        chunks = load_text_corpus(path, tokenizer, bos, chunk_tokens=ct,
                                  max_chunks=int(mc) if mc else None, chunk_offset=off)
        return chunks, f"{os.path.basename(path)} ({len(chunks)} chunks x {ct} tok, off={off})"
    from corpus import CORPUS
    pre = [bos] if bos is not None else []
    return ([pre + tokenizer.encode(s, add_special_tokens=False).ids for s in CORPUS],
            "synthetic-mini-corpus")
