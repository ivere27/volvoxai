#!/usr/bin/env python3
"""
Export a HuggingFace byte-level BPE tokenizer (GPT-2 / GPT-Neo family) into the two
files VolvoxAI's tokenizers consume:

  * vocab.bin  - the binary format read by native/tokenizer.c and js/Tokenizer.js:
                     int32  vocab_size
                     repeat vocab_size times:
                         int32  len
                         byte[len]   (the UTF-8 bytes of the token string, id order)
  * merges.txt - the standard GPT-2 merges file (used to enable proper BPE merging).

Usage:
    python3 tools/export_tokenizer.py <hf-model-id> <out-dir>
    python3 tools/export_tokenizer.py roneneldan/TinyStories-1M models/tinystories_1m
"""
import struct
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    model_id, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    print(f"[Tokenizer] Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    vocab = tok.get_vocab()  # token_str -> id
    size = max(vocab.values()) + 1
    id_to_tok = [""] * size
    for text, idx in vocab.items():
        id_to_tok[idx] = text

    vocab_bin = out_dir / "vocab.bin"
    with open(vocab_bin, "wb") as f:
        f.write(struct.pack("<i", size))
        for text in id_to_tok:
            b = text.encode("utf-8")
            f.write(struct.pack("<i", len(b)))
            f.write(b)
    print(f"[Tokenizer] Wrote {vocab_bin} ({size} tokens)")

    # save_vocabulary emits vocab.json + merges.txt in GPT-2 format.
    saved = tok.save_vocabulary(str(out_dir))
    print(f"[Tokenizer] Wrote merges/vocab: {saved}")
    if not (out_dir / "merges.txt").exists():
        print("[Tokenizer] WARNING: merges.txt not produced; BPE merging will be greedy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
