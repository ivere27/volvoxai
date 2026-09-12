#!/usr/bin/env python3
"""WASM/native parity over the committed model corpus.

The ONNX oracle compares each backend against onnxruntime on small synthetic
graphs. This gate is the complementary axis: real committed models, hundreds of
nodes, compared WASM-against-native directly. No external oracle exists for
these packages, so native CPU is the reference and WASM must match it.

Inputs are generated from a fixed seed and written next to the run, so a rerun
on the same corpus reproduces the same bytes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from onnx_oracle import RAW_SUFFIXES, fail, run_command, run_native  # noqa: E402

WASM_RUNNER = HERE / "run_wasm.mjs"
SEED = 20260905
# Symbolic extents are resolved to one fixed profile so native and WASM receive
# byte-identical inputs. The value is deliberately odd to exercise tail paths.
SYMBOLIC_EXTENTS = {"S": 7}
NUMPY_DTYPES = {
    "float32": np.float32,
    "int32": np.int32,
    "int8": np.int8,
    "uint8": np.uint8,
}


def resolve_shape(name: str, shape: list) -> list[int]:
    resolved = []
    for extent in shape:
        if isinstance(extent, int):
            resolved.append(extent)
            continue
        if extent not in SYMBOLIC_EXTENTS:
            fail(f"input {name!r} has unbound symbolic extent {extent!r}")
        resolved.append(SYMBOLIC_EXTENTS[extent])
    return resolved


def generate_input(rng, name: str, shape: list[int], dtype: str, graph: dict) -> np.ndarray:
    elements = int(np.prod(shape)) if shape else 1
    if dtype == "float32":
        return rng.standard_normal(elements, dtype=np.float32).reshape(shape)
    if dtype == "uint8":
        return rng.integers(0, 256, size=elements, dtype=np.uint8).reshape(shape)
    if dtype == "int8":
        return rng.integers(-128, 128, size=elements, dtype=np.int8).reshape(shape)
    if dtype == "int32":
        # Token-like inputs must stay inside the embedding table, and position
        # inputs must stay inside the trained context window.
        limit = embedding_limit(graph, name)
        if name.startswith("position"):
            return np.arange(elements, dtype=np.int32).reshape(shape)
        return rng.integers(0, limit, size=elements, dtype=np.int32).reshape(shape)
    fail(f"input {name!r} has unsupported dtype {dtype!r}")


def embedding_limit(graph: dict, name: str) -> int:
    """Smallest embedding row count that consumes this input, or a safe default."""
    limits = []
    for node in graph.get("nodes", []):
        if node.get("opType") not in {"Embedding", "QEmbedding"}:
            continue
        if name not in (node.get("inputs") or {}).values():
            continue
        for key in ("numEmbeddings", "num_embeddings", "vocabSize", "vocab_size"):
            value = (node.get("parameters") or {}).get(key)
            if isinstance(value, int) and value > 0:
                limits.append(value)
    return min(limits) if limits else 128


def write_inputs(package: Path, destination: Path) -> Path:
    graph = json.loads((package / "graph.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    destination.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for name, spec in graph["inputs"].items():
        dtype = spec["dtype"]
        if dtype not in NUMPY_DTYPES:
            fail(f"input {name!r} has unsupported dtype {dtype!r}")
        shape = resolve_shape(name, spec["shape"])
        array = generate_input(rng, name, shape, dtype, graph)
        path = destination / f"{name}{RAW_SUFFIXES[dtype]}"
        path.write_bytes(array.tobytes())
        descriptors.append({
            "name": name, "shape": shape, "dtype": dtype, "file": os.fspath(path),
        })
    manifest = destination / "inputs.json"
    manifest.write_text(
        json.dumps({"schema": "volvoxai.model-corpus-inputs", "version": 1,
                    "seed": SEED, "inputs": descriptors}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_wasm(bundle: Path, wasm: Path, package: Path,
             manifest: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    run_command(
        ["node", os.fspath(WASM_RUNNER),
         "--bundle", os.fspath(bundle), "--wasm", os.fspath(wasm),
         "--package", os.fspath(package), "--inputs", os.fspath(manifest),
         "--out", os.fspath(destination)],
        cwd=ROOT,
    )
    return json.loads((destination / "outputs.json").read_text(encoding="utf-8"))


def load(destination: Path, descriptor: dict) -> np.ndarray:
    dtype = descriptor["dtype"]
    if dtype not in NUMPY_DTYPES:
        fail(f"output {descriptor['name']!r} has unsupported dtype {dtype!r}")
    path = destination / descriptor["file"]
    return np.frombuffer(path.read_bytes(), dtype=NUMPY_DTYPES[dtype])


def compare(native_dir: Path, native_meta: dict,
            wasm_dir: Path, wasm_meta: dict, rtol: float, atol: float) -> list[dict]:
    native_by_name = {d["name"]: d for d in native_meta["outputs"]}
    wasm_by_name = {d["name"]: d for d in wasm_meta["outputs"]}
    if sorted(native_by_name) != sorted(wasm_by_name):
        fail(f"output names differ: native {sorted(native_by_name)} wasm {sorted(wasm_by_name)}")
    comparisons = []
    for name in sorted(native_by_name):
        reference = load(native_dir, native_by_name[name])
        candidate = load(wasm_dir, wasm_by_name[name])
        if reference.shape != candidate.shape:
            fail(f"output {name!r} element count differs: "
                 f"native {reference.size} wasm {candidate.size}")
        if native_by_name[name]["dtype"] != wasm_by_name[name]["dtype"]:
            fail(f"output {name!r} dtype differs")
        exact = native_by_name[name]["dtype"] != "float32"
        if exact:
            mismatches = int(np.count_nonzero(reference != candidate))
            if mismatches:
                fail(f"output {name!r} is not byte-identical: {mismatches} of "
                     f"{reference.size} elements differ")
            comparisons.append({"name": name, "mode": "exact",
                                "elements": int(reference.size),
                                "max_abs": 0.0, "max_rel": 0.0})
            continue
        left = reference.astype(np.float64)
        right = candidate.astype(np.float64)
        delta = np.abs(left - right)
        max_abs = float(delta.max()) if delta.size else 0.0
        scale = np.maximum(np.abs(left), np.abs(right))
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(scale > 0, delta / scale, 0.0)
        max_rel = float(relative.max()) if relative.size else 0.0
        if not np.allclose(left, right, rtol=rtol, atol=atol):
            fail(f"output {name!r} exceeds tolerance: max_abs={max_abs:g} max_rel={max_rel:g}")
        comparisons.append({"name": name, "mode": "tolerance",
                            "elements": int(reference.size), "rtol": rtol, "atol": atol,
                            "max_abs": max_abs, "max_rel": max_rel})
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--wasm", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=ROOT / "native" / "volvoxai")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--model", action="append", default=[],
                        help="Run one model directory name; repeatable")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--work-dir", type=Path, help="Preserve artifacts in a new directory")
    parser.add_argument("--report", type=Path, help="Write the final JSON report")
    args = parser.parse_args()

    for label, path in (("inference bundle", args.bundle), ("release WASM", args.wasm),
                        ("native runner", args.native)):
        if not path.is_file():
            fail(f"{label} is missing: {path}")

    packages = sorted(
        entry for entry in args.models_dir.iterdir()
        if entry.is_dir() and (entry / "graph.json").is_file()
        and (entry / "model.safetensors").is_file()
    )
    if args.model:
        wanted = set(args.model)
        packages = [entry for entry in packages if entry.name in wanted]
        missing = wanted - {entry.name for entry in packages}
        if missing:
            fail(f"unknown model(s): {', '.join(sorted(missing))}")
    if not packages:
        fail("model corpus is empty")

    if args.work_dir:
        if args.work_dir.exists():
            fail(f"--work-dir already exists: {args.work_dir}")
        root = args.work_dir
        root.mkdir(parents=True)
        temporary = None
    else:
        temporary = tempfile.mkdtemp(prefix="volvoxai-model-corpus-")
        root = Path(temporary)

    cases = []
    try:
        for package in packages:
            print(f"model-corpus {package.name}: running", flush=True)
            case_root = root / package.name
            manifest = write_inputs(package, case_root / "inputs")
            native_dir = case_root / "native-cpu"
            native_meta = run_native(args.native, package, manifest, native_dir, "cpu")
            wasm_dir = case_root / "wasm"
            wasm_meta = run_wasm(args.bundle, args.wasm, package, manifest, wasm_dir)
            comparisons = compare(native_dir, native_meta, wasm_dir, wasm_meta,
                                  args.rtol, args.atol)
            graph = json.loads((package / "graph.json").read_text(encoding="utf-8"))
            cases.append({
                "id": package.name,
                "nodes": len(graph.get("nodes", [])),
                "reference": "native-cpu",
                "candidate": "wasm",
                "comparisons": comparisons,
            })
            worst = max((c["max_abs"] for c in comparisons), default=0.0)
            print(f"model-corpus {package.name}: PASS (max_abs={worst:g})", flush=True)
    finally:
        if temporary and not args.work_dir:
            shutil.rmtree(temporary, ignore_errors=True)

    report = {"schema": "volvoxai.model-corpus-report", "version": 1,
              "seed": SEED, "symbolicExtents": SYMBOLIC_EXTENTS,
              "rtol": args.rtol, "atol": args.atol, "cases": cases}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
