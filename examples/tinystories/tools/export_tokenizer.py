#!/usr/bin/env python3
"""Export the TinyStories byte-level BPE vocabulary for VolvoxAI examples.

Outputs ``vocab.bin`` for VolvoxAI tokenizers and the tokenizer's GPT-2-format
``merges.txt``/``vocab.json`` assets.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer


def export_tokenizer(model_id_or_path: str, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Tokenizer] Loading {model_id_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    vocabulary = tokenizer.get_vocab()
    size = max(vocabulary.values()) + 1
    id_to_token = [""] * size
    for text, token_id in vocabulary.items():
        id_to_token[token_id] = text

    vocabulary_path = output_dir / "vocab.bin"
    with vocabulary_path.open("wb") as output:
        output.write(struct.pack("<i", size))
        for text in id_to_token:
            encoded = text.encode("utf-8")
            output.write(struct.pack("<i", len(encoded)))
            output.write(encoded)
    print(f"[Tokenizer] Wrote {vocabulary_path} ({size} tokens)")

    saved = tokenizer.save_vocabulary(str(output_dir))
    print(f"[Tokenizer] Wrote merges/vocab: {saved}")
    if not (output_dir / "merges.txt").exists():
        print("[Tokenizer] WARNING: merges.txt not produced; BPE merging will be greedy.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model ID or directory")
    parser.add_argument("out_dir", type=Path, help="Output package directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export_tokenizer(args.model, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
