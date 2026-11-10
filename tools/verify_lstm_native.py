#!/usr/bin/env python3
"""Check live operator requests against ONNX and same-backend baseline bits."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.tts_common.artifacts import sha256
from examples.tts_common.benchmark_native import Native


def arrays(tensors):
    return {item["name"]: np.frombuffer(base64.b64decode(item["data"]),
        dtype=np.float32 if item["dtype"] == "F32" else np.int32).copy().reshape(item["shape"])
        for item in tensors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--backend", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--operator", default="lstm", choices=("lstm", "conv"))
    parser.add_argument("--baseline-outputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    outputs_path = args.out.with_suffix(".outputs.npz")
    baseline = None
    if args.baseline_outputs:
        with np.load(args.baseline_outputs, allow_pickle=False) as values:
            baseline = dict(values)
    snapshots = {}
    report = {"format": f"volvoxai-{args.operator}-numerical-validation/v1", "status": "failed",
              "backend": args.backend, "library_sha256": sha256(args.library),
              "corpus_sha256": sha256(args.corpus), "cases": []}
    if baseline is not None:
        report["baseline_outputs_sha256"] = sha256(args.baseline_outputs)
    try:
        for case_index, case in enumerate(json.loads(args.corpus.read_text())["cases"]):
            with tempfile.TemporaryDirectory(prefix="volvox-lstm-case-") as directory:
                root = Path(directory)
                (root / "graph.json").write_text(json.dumps(case["graph"]))
                weight_paths = []
                for index, encoded in enumerate(case["weight_shards"]):
                    path = root / f"weights-{index}.safetensors"
                    path.write_bytes(base64.b64decode(encoded))
                    weight_paths.append(path.name)
                contract = {"graph": "graph.json", "weights": weight_paths,
                    "inputs": {item["name"]: item["name"] for item in case["inputs"]},
                    "outputs": {item["name"]: item["name"] for item in case["expected"]}}
                runner = Native(root, contract, args.backend, 4, args.library, 120)
                row = {"name": case["name"], "requests": []}
                try:
                    for request_index, request in enumerate(case.get("requests", [case])):
                        runner.prepare(arrays(request["inputs"]))
                        actual, _ = runner.run()
                        expected = arrays(request["expected"])
                        maximum = 0.
                        for name, value in actual.items():
                            np.testing.assert_allclose(value, expected[name], rtol=1e-4, atol=1e-5)
                            maximum = max(maximum, float(np.max(np.abs(value - expected[name]))))
                            key = f"case_{case_index}_request_{request_index}_{name}"
                            snapshots[key] = value.copy()
                            if baseline is not None:
                                np.testing.assert_array_equal(value.view(np.uint32), baseline[key].view(np.uint32))
                        row["requests"].append({"index": request_index, "max_onnx_error": maximum,
                                                "baseline_bitwise_equal": baseline is not None})
                    row["route"] = runner.route
                    row["status"] = "pass"
                    report["cases"].append(row)
                finally:
                    runner.close()
            print("PASS", args.backend, case["name"], flush=True)
        if baseline is not None and snapshots.keys() != baseline.keys():
            raise ValueError("baseline request/output set differs")
        report["status"] = "pass"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(outputs_path, **snapshots)
        report["outputs_sha256"] = sha256(outputs_path)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
