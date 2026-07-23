#!/usr/bin/env python3
"""Model-level PyTorch oracle for TinyStories-1M: load the *source* HF checkpoint
(roneneldan/TinyStories-1M, the GPT-Neo the VolvoxAI package was exported from),
run the exact token fixture, and emit the row-4 logits signature the harness reads.
This is the genuine "PyTorch vs VolvoxAI" whole-model correctness check.

Writes tests/parity/out/tinystories_1m.torch.json.
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PARITY))
sys.path.insert(0, HERE)
from sigutil import signature, write_json_atomic  # noqa: E402

def main():
    policy = json.load(open(os.path.join(PARITY, "policy.json")))
    dst = os.path.join(PARITY, "out", "tinystories_1m.torch.json")
    try:
        os.unlink(dst)
    except FileNotFoundError:
        pass
    spec = policy["models"]["tinystories_1m"]["outputs"][0]
    row_index = spec["row"]
    tokens = np.fromfile(os.path.join(ROOT, "models/tinystories_1m/tokens.i32"), dtype=np.int32).reshape(1, -1)
    external = policy["models"]["tinystories_1m"]["external"]
    model = AutoModelForCausalLM.from_pretrained(
        external["repo"], revision=external["revision"]
    )
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=torch.from_numpy(tokens.astype(np.int64))).logits  # [1, S, vocab]
    row = logits[0, row_index, :].to(torch.float64).numpy()
    sig = signature(
        row,
        topk=spec.get("topk", 10),
        shape=spec["shape"],
        sample_axes=spec.get("sampleAxes", []),
    )
    write_json_atomic(
        dst,
        {"model": "tinystories_1m", "backend": "torch", "ms": None, "sigs": {"logits": sig}},
    )
    print(f"pytorch-oracle tinystories_1m: row {row_index} logits[{row.shape[0]}] -> {os.path.relpath(dst, ROOT)}")


if __name__ == "__main__":
    main()
