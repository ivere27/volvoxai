#!/usr/bin/env python3
"""Generate self-contained test cases for PackedLinear, PackedLSTM, and QConv2D (weight-only)

Each test directory contains:
- graph.json
- model.safetensors
- x.f32 (raw input tensor)
- y_ref.f32 (golden reference output computed on CPU)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from safetensors.numpy import save

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from tools.exporter.packed_linear import pack_weight, pack_lstm_weights
from python.tests.test_packed_lstm import make_lstm_fixture


def make_packed_linear_case(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(101)
    in_features = 64
    out_features = 128
    batch = 4
    x = rng.normal(0, 1, (1, batch, in_features)).astype(np.float32)
    w = rng.normal(0, 0.5, (out_features, in_features)).astype(np.float32)
    bias = rng.normal(0, 0.1, (out_features,)).astype(np.float32)

    words, scales = pack_weight(w, weight_bits=4, group_size=32)

    graph = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1, batch, in_features], "dtype": "float32"}},
        "nodes": [
            {
                "id": "linear",
                "opType": "PackedLinear",
                "inputs": {"input": "x", "weight": "w", "scale": "s", "bias": "b"},
                "outputs": {"out": {"tensor": "y", "shape": [1, batch, out_features], "dtype": "float32"}},
                "params": {"weight_bits": 4, "group_size": 32},
            }
        ],
        "outputs": ["y"],
        "execution": {
            "stateless": {},
            "entrypoints": [
                {
                    "name": "evaluate",
                    "effect": 1,
                    "allowed_ranges": [1],
                    "inputs": ["x"],
                    "outputs": ["y"],
                }
            ],
        },
    }
    weights = {"w": words, "s": scales, "b": bias}

    (out_dir / "graph.json").write_text(json.dumps(graph, indent=2))
    (out_dir / "model.safetensors").write_bytes(save(weights))
    (out_dir / "x.f32").write_bytes(x.tobytes())

    # Compute CPU golden reference
    cmd = [
        str(ROOT / "native/volvoxai"), "run", str(out_dir),
        "--cpu", "--input", f"x={out_dir}/x.f32",
        "--output", f"y={out_dir}/y_ref.f32",
        "--report-json", f"{out_dir}/report_ref.json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("PackedLinear CPU reference generated. Output shape:", [1, batch, out_features])


def make_packed_lstm_case(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len, batch, width, hidden, direction = 4, 2, 16, 8, 2
    graph, weights, x = make_lstm_fixture(
        seq_len=seq_len, batch=batch, width=width, hidden=hidden, direction=direction
    )
    packed_graph, packed_weights = pack_lstm_weights(graph, weights, weight_bits=4, group_size=32)

    (out_dir / "graph.json").write_text(json.dumps(packed_graph, indent=2))
    (out_dir / "model.safetensors").write_bytes(save(packed_weights))
    (out_dir / "x.f32").write_bytes(x.tobytes())

    # Compute CPU golden reference
    cmd = [
        str(ROOT / "native/volvoxai"), "run", str(out_dir),
        "--cpu", "--input", f"x={out_dir}/x.f32",
        "--output", f"y={out_dir}/y_ref.f32",
        "--report-json", f"{out_dir}/report_ref.json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("PackedLSTM CPU reference generated. Output shape:", [seq_len, direction, batch, hidden])


def make_qconv2d_case(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(202)
    out_ch, kh, kw, in_ch, groups = 4, 1, 3, 24, 1
    terms = kh * kw * (in_ch // groups)  # 72 -> 3 groups of 32
    codes = rng.integers(-8, 8, size=(out_ch, terms), dtype=np.int8)
    packed = np.full((out_ch, (terms + 1) // 2), 0x88, dtype=np.uint8)
    packed[:, :] = (codes[:, ::2] + 8).astype(np.uint8) | (packed & 0xF0)
    packed[:, : terms // 2] = (
        ((codes[:, 1::2] + 8).astype(np.uint8) << 4)
        | (packed[:, : terms // 2] & 0x0F)
    )
    scales = rng.uniform(0.02, 0.25, size=(out_ch, 3)).astype(np.float32)
    bias = rng.uniform(-0.1, 0.1, size=(out_ch,)).astype(np.float32)
    image = rng.normal(0.0, 0.5, size=(1, 1, 10, in_ch)).astype(np.float32)

    qgraph = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 1, 10, in_ch], "dtype": "float32"}},
        "nodes": [
            {
                "id": "qconv",
                "opType": "QConv2D",
                "inputs": {
                    "input": "x",
                    "weight": "w",
                    "weight_scale": "ws",
                    "bias": "b",
                },
                "outputs": {
                    "out": {"tensor": "y", "shape": [1, 1, 10, out_ch], "dtype": "float32"}
                },
                "params": {
                    "data_layout": "NHWC",
                    "weight_layout": "OHWI",
                    "weight_only": True,
                    "weight_bits": 4,
                    "kernel_size": [kh, kw],
                    "groups": groups,
                    "stride": [1, 1],
                    "dilation": [1, 1],
                    "pads": [0, 1, 0, 1],
                },
            }
        ],
        "outputs": ["y"],
        "execution": {
            "stateless": {},
            "entrypoints": [
                {
                    "name": "evaluate",
                    "effect": 1,
                    "allowed_ranges": [1],
                    "inputs": ["x"],
                    "outputs": ["y"],
                }
            ],
        },
    }

    (out_dir / "graph.json").write_text(json.dumps(qgraph, indent=2))
    (out_dir / "model.safetensors").write_bytes(save({"w": packed, "ws": scales, "b": bias}))
    (out_dir / "x.f32").write_bytes(image.tobytes())

    # Compute CPU golden reference
    cmd = [
        str(ROOT / "native/volvoxai"), "run", str(out_dir),
        "--cpu", "--input", f"x={out_dir}/x.f32",
        "--output", f"y={out_dir}/y_ref.f32",
        "--report-json", f"{out_dir}/report_ref.json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("QConv2D weight-only CPU reference generated. Output shape:", [1, 1, 10, out_ch])


def main():
    base_dir = ROOT / "build/gpu_packed_test/cases"
    make_packed_linear_case(base_dir / "packed_linear")
    make_packed_lstm_case(base_dir / "packed_lstm")
    make_qconv2d_case(base_dir / "qconv2d")
    print("All fixtures generated successfully in:", base_dir)


if __name__ == "__main__":
    main()
