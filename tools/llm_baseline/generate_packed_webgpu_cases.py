#!/usr/bin/env python3
"""Independent packed projection corpus for the public WebGPU release runner."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.tests.test_packed_linear import packed_words, reference


def tensor(name, value, oracle=False):
    value = np.asarray(value, dtype="<f8" if oracle else "<f4", order="C")
    return {"name": name, "shape": list(value.shape), "dtype": "F64" if oracle else "F32",
            "data": base64.b64encode(value.tobytes()).decode("ascii")}


def shard(values):
    header, payload = {}, bytearray()
    for name, value in values.items():
        data = np.ascontiguousarray(value).tobytes()
        header[name] = {"dtype": "I32" if value.dtype == np.int32 else "F32",
                        "shape": list(value.shape), "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return base64.b64encode(struct.pack("<Q", len(encoded)) + encoded + payload).decode("ascii")


def fixture(rows, width, outputs, group, bits, arithmetic=8, pattern="finite"):
    rng = np.random.default_rng(471 + width + bits)
    x = rng.normal(0, .5, (rows, width)).astype(np.float32)
    x[0].fill(0)
    if rows > 1:
        x[1] = ((np.arange(width) % 2) * 2 - 1) * ((np.arange(width) % 31) + .5)
        x[1, 0] = 127
    if pattern == "nonfinite":
        x[-1, -1] = np.nan
    limit = 1 << (bits - 1)
    codes = rng.integers(-limit, limit, (outputs, width), dtype=np.int8)
    scales = rng.uniform(.001, .1, (outputs, (width + group - 1) // group)).astype(np.float32)
    bias = rng.normal(0, .25, outputs).astype(np.float32)
    expected = reference(x, codes, scales, bias, arithmetic, group)
    graph = {"format": "volvox-graph/v1", "dimensions": {},
             "inputs": {"input": {"shape": [rows, width], "dtype": "float32"}},
             "nodes": [{"id": "projection", "opType": "PackedLinear",
                        "inputs": {name: name for name in ("input", "weight", "scale", "bias")},
                        "outputs": {"out": {"tensor": "out", "shape": [rows, outputs], "dtype": "float32"}},
                        "params": {"group_size": group, "weight_bits": bits}}],
             "outputs": ["out"],
             "execution": {"entrypoints": [{"name": "evaluate", "effect": 1, "allowed_ranges": [1],
                                              "inputs": ["input"], "outputs": ["out"]}], "stateless": {}}}
    return {"name": f"packed-shared/M{rows}-K{width}-N{outputs}-G{group}-W{bits}-A{arithmetic}-{pattern}",
            "graph": graph, "inputs": [tensor("input", x)], "expected": [tensor("out", expected, True)],
            "weight_shards": [shard({"weight": packed_words(codes, bits), "scale": scales, "bias": bias})],
            "packed_arithmetic": {8: 1, 16: 2, 32: 3}[arithmetic]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    cases = [fixture(rows, width, outputs, group, bits, pattern=pattern)
             for bits in (4, 8) for group in (32, 64, 128, 256)
             for rows, width, outputs, pattern in ((3, group + 5, 64, "finite"), (4, 513, 128, "nonfinite"))]
    cases += [fixture(3, 35, 64, 32, bits, arithmetic) for bits in (4, 8) for arithmetic in (16, 32)]
    cases += [fixture(402, 320, 320, 32, bits) for bits in (4, 8)]
    result = {"format": "volvox-modern-primitives-wasm-oracles/v1", "count": len(cases), "cases": cases,
              "tolerance": {"absolute": 1e-5, "relative": 1e-5},
              "expected_source": "independent NumPy integer dots and ordered F32 group sums, widened for transport",
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256((ROOT / "tools/tests/test_packed_linear.py").read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(f"Wrote {len(cases)} independent public-API fixtures: {args.out}")


if __name__ == "__main__":
    main()
