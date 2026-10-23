#!/usr/bin/env python3
"""Measure storage-group tradeoffs through completed public PackedLinear calls.

Synthetic reconstruction/output errors are diagnostic, not language-model
quality. Run this same script with baseline/candidate libraries and --groups
128 to isolate implementation regression from changes in quantized weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import volvoxai as vx
from tools.exporter.packed_linear import pack_weight
from tools.remote_benchmark_runner import distribution, sha256_file


def fixture(m, n, k, bits, group):
    rng = np.random.default_rng(20261023)
    weight = rng.normal(0, .1, (n, k)).astype(np.float32)
    # Identical sparse outliers for every G make scale-sharing loss visible.
    weight[:, ::113] *= 8
    x = rng.normal(0, .25, (m, k)).astype(np.float32)
    bias = rng.normal(0, .05, n).astype(np.float32)
    words, scales = pack_weight(weight, weight_bits=bits, group_size=group)
    lanes, offset = 32 // bits, 1 << (bits - 1)
    indices = np.arange(k)
    codes = ((words.view(np.uint32)[:, indices // lanes] >>
              (indices % lanes * bits).astype(np.uint32)) & (2**bits - 1)).astype(np.int32) - offset
    decoded = codes.astype(np.float32) * scales[:, indices // group]
    graph = {"format": "volvox-graph/v1", "dimensions": {},
        "inputs": {"x": {"dtype": "float32", "shape": [m, k]}},
        "nodes": [{"id": "projection", "opType": "PackedLinear",
            "inputs": {"input": "x", "weight": "w", "scale": "s", "bias": "b"},
            "outputs": {"out": {"tensor": "y", "dtype": "float32", "shape": [m, n]}},
            "params": {"weight_bits": bits, "group_size": group}}], "outputs": ["y"],
        "execution": {"stateless": {}, "entrypoints": [{"name": "evaluate", "effect": 1,
            "allowed_ranges": [1], "inputs": ["x"], "outputs": ["y"]}]}}
    original = x.astype(np.float64) @ weight.astype(np.float64).T + bias
    quantized = x.astype(np.float64) @ decoded.astype(np.float64).T + bias
    info = {"weight_mse": float(np.mean((decoded.astype(np.float64) - weight)**2)),
        "packed_weight_bytes": words.nbytes, "scale_bytes": scales.nbytes,
        "weight_scale_sha256": hashlib.sha256(words.tobytes() + scales.tobytes()).hexdigest(),
        "input_sha256": hashlib.sha256(x.tobytes()).hexdigest()}
    return graph, {"w": words, "s": scales, "b": bias}, x, original, quantized, info


def nrms(actual, expected):
    return float(np.sqrt(np.mean((actual.astype(np.float64) - expected)**2) /
                         np.mean(expected**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "vulkan", "opengl"])
    parser.add_argument("--groups", nargs="+", type=int, default=[32, 64, 128, 256], choices=[32, 64, 128, 256])
    parser.add_argument("--shapes", nargs="+", default=["1,1024,2048", "32,256,512", "512,256,2048"],
                        help="M,N,K triples")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    shapes = [tuple(map(int, value.split(","))) for value in args.shapes]
    if any(len(s) != 3 or min(s) < 1 for s in shapes):
        parser.error("shapes must be positive M,N,K triples")
    report = {"format": "volvox-packed-groups-benchmark/v1", "backend": args.backend,
        "seed": 20261023, "warmup": args.warmup, "repeats": args.repeats,
        "timing_scope": "completed public run, including transport and copies; warmed reusable context",
        "quality_scope": "synthetic weight reconstruction and projection output; no task-quality claim",
        "script_sha256": sha256_file(__file__), "cases": []}
    library = os.environ.get("VOLVOXAI_LIBRARY")
    if library:
        report["library_sha256"] = sha256_file(library)
    for m, n, k in shapes:
        for bits in (4, 8):
            for group in args.groups:
                graph, weights, x, original, quantized, info = fixture(m, n, k, bits, group)
                with tempfile.TemporaryDirectory(prefix="volvox-packed-groups-") as tmp:
                    path = Path(tmp)
                    (path / "graph.json").write_text(json.dumps(graph))
                    save_file(weights, path / "model.safetensors")
                    for arithmetic in ("f32", "f32-fast", "f16", "dynamic-i8"):
                        with vx.InferenceSession(path, backend=args.backend, cpu_threads=1,
                                                 packed_arithmetic=arithmetic) as session:
                            assert session.backend == args.backend
                            assert not session.report.fallback.operator_fallback_used
                            for _ in range(args.warmup):
                                session.run({"x": x})
                            durations = []
                            for _ in range(args.repeats):
                                start = time.perf_counter_ns()
                                result = session.run({"x": x})["y"]
                                durations.append((time.perf_counter_ns() - start) / 1e6)
                            with session.trace(detail="nodes", device_timing=args.backend == "cuda") as trace:
                                traced = session.run({"x": x})["y"]
                            np.testing.assert_array_equal(result, traced)
                            programs = sorted({event.name for event in trace.events if event.HasField("program")})
                            case = {"shape_mnk": [m, n, k], "weight_bits": bits, "group_size": group,
                                "arithmetic": arithmetic, **info, "latency_ms": distribution(durations),
                                "output_nrms_vs_float_weights": nrms(result, original),
                                "output_nrms_vs_decoded_weights": nrms(result, quantized),
                                "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
                                "programs": programs}
                            report["cases"].append(case)
                            print(f"M{m} N{n} K{k} W{bits} G{group} {arithmetic}: "
                                  f"{case['latency_ms']['p50']:.3f} ms {programs}", flush=True)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
