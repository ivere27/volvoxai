#!/usr/bin/env python3
"""Project a prepared independent H oracle into browser-readable inputs.

This changes container format only. It never recomputes model outputs, changes
budgets, or loads HF/ORT inside the candidate browser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from safetensors.numpy import save_file


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for value in iter(lambda: stream.read(1 << 20), b""): digest.update(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "use a fresh receipt"
    corpus = json.loads((args.bundle / "corpus.json").read_text())
    assert corpus["status"] == "REFERENCE_PREPARED"
    assert sha(args.bundle / "references.npz") == corpus["references_sha256"]
    with np.load(args.bundle / "references.npz", allow_pickle=False) as source:
        arrays = {name: np.ascontiguousarray(source[name]) for name in source.files}
    assert len(arrays) == 58, "qualified independent H corpus has 58 arrays"
    assert all(value.dtype in (np.dtype("float32"), np.dtype("int32")) for value in arrays.values())
    receipt = {"format": "volvox-browser-multimodal-input-projection/v1", "status": "PASS",
        "corpus_sha256": sha(args.bundle / "corpus.json"), "source_reference_sha256": corpus["references_sha256"],
        "array_count": len(arrays), "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in arrays.items()},
        "scope": "Original independent HF/ORT arrays projected without numerical changes; raw F32 PCM retained."}
    assert receipt["arrays"]["audio_pcm"]["sha256"] == corpus["audio"]["pcm_sha256"]
    # Verify before replacing, and preserve an already identical projection.
    with tempfile.TemporaryDirectory(prefix=".browser-projection-", dir=args.bundle) as temporary:
        work = Path(temporary)
        save_file(arrays, str(work / "browser-references.safetensors"))
        (work / "audio.pcm.f32").write_bytes(arrays["audio_pcm"].tobytes())
        for name in ("browser-references.safetensors", "audio.pcm.f32"):
            source, target = work / name, args.bundle / name
            if target.exists(): assert sha(target) == sha(source), "existing browser projection differs"
            else: source.replace(target)
            receipt[name] = {"bytes": target.stat().st_size, "sha256": sha(target)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "arrays": len(arrays), "receipt": str(args.out)}))


if __name__ == "__main__": main()
