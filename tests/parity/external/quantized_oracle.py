#!/usr/bin/env python3
"""Byte-exact project QLinear seam for portable and strict GPU backends."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from safetensors.numpy import save_file

from onnx_oracle import (
    HarnessError,
    compare,
    fail,
    load_candidate_arrays,
    run_native,
    run_wasm,
    run_webgpu,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE_VERSION = json.loads(
    (ROOT / "package.json").read_text(encoding="utf-8")
)["version"]

ROWS = np.asarray([
    [-1, -1, -1],
    [-3, -3, -3],
], dtype=np.int8)
EXPECTED = np.asarray([
    [4, 4, 127, -128],
    [2, 2, -128, 127],
], dtype=np.int8)

OPTIMIZED_ROWS = 2
OPTIMIZED_K = 33
OPTIMIZED_N = 36


def per_tensor(scale: str, zero_point: str) -> dict[str, str]:
    return {
        "scheme": "per_tensor",
        "scale_tensor": scale,
        "zero_point_tensor": zero_point,
    }


def write_package(destination: Path) -> None:
    destination.mkdir(parents=True)
    graph = {
        "format": "volvox-graph/v1",
        "dimensions": {"B": {"min": 1, "max": 2, "multiple_of": 1}},
        "inputs": {"x": {"shape": ["B", 3], "dtype": "int8"}},
        "nodes": [{
            "id": "qlinear_boundary",
            "opType": "QLinear",
            "inputs": {"input": "x", "weight": "weight", "bias": "bias"},
            "outputs": {
                "out": {"tensor": "y", "shape": ["B", 4], "dtype": "int8"},
            },
            "params": {},
        }],
        "outputs": ["y"],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "x": per_tensor("__quant__.x.scale", "__quant__.x.zero_point"),
                "weight": {
                    "scheme": "per_axis",
                    "axis": 0,
                    "scale_tensor": "__quant__.weight.scale",
                    "zero_point_tensor": "__quant__.weight.zero_point",
                },
                "y": per_tensor("__quant__.y.scale", "__quant__.y.zero_point"),
            },
        },
    }
    (destination / "graph.json").write_text(
        json.dumps(graph, indent=2) + "\n", encoding="utf-8",
    )
    save_file({
        "weight": np.asarray([
            [2, 1, 1],
            [-3, -2, -3],
            [127, 127, 127],
            [-128, -128, -128],
        ], dtype=np.int8),
        "bias": np.zeros(4, dtype=np.int32),
        "__quant__.x.scale": np.asarray([1.0], dtype=np.float32),
        "__quant__.x.zero_point": np.asarray([-2], dtype=np.int8),
        "__quant__.weight.scale": np.asarray([0.5, 0.5, 1.0, 1.0], dtype=np.float32),
        "__quant__.weight.zero_point": np.asarray([1, -3, 2, -1], dtype=np.int8),
        "__quant__.y.scale": np.asarray([1.0], dtype=np.float32),
        "__quant__.y.zero_point": np.asarray([3], dtype=np.int8),
    }, os.fspath(destination / "model.safetensors"))


def write_inputs(destination: Path, batch: int) -> Path:
    inputs = destination / "inputs"
    inputs.mkdir(parents=True)
    payload = inputs / "x.i8"
    payload.write_bytes(ROWS[:batch].tobytes(order="C"))
    manifest = destination / "inputs.json"
    manifest.write_text(json.dumps({
        "schema": "volvoxai.quantized-oracle-inputs",
        "version": 1,
        "inputs": [{
            "name": "x",
            "dtype": "int8",
            "shape": [batch, 3],
            "file": os.fspath(payload.resolve()),
        }],
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def _optimized_fixture(
    case_id: str,
    input_dtype: str,
    weight_dtype: str,
    output_dtype: str,
    seed: int,
) -> dict[str, object]:
    dtype = {"int8": np.int8, "uint8": np.uint8}
    input_zero_point = -3 if input_dtype == "int8" else 129
    output_zero_point = -5 if output_dtype == "int8" else 131
    weight_zero_points = np.asarray([
        (-4 + channel % 7) if weight_dtype == "int8" else (120 + channel % 7)
        for channel in range(OPTIMIZED_N)
    ], dtype=dtype[weight_dtype])
    input_values = np.asarray([
        input_zero_point + ((index * 7 + seed) % 17) - 8
        for index in range(OPTIMIZED_ROWS * OPTIMIZED_K)
    ], dtype=dtype[input_dtype]).reshape(OPTIMIZED_ROWS, OPTIMIZED_K)
    weight_values = np.empty((OPTIMIZED_N, OPTIMIZED_K), dtype=dtype[weight_dtype])
    for channel in range(OPTIMIZED_N):
        zero_point = int(weight_zero_points[channel])
        weight_values[channel] = np.asarray([
            zero_point + ((term * 5 + channel * 3 + seed) % 15) - 7
            for term in range(OPTIMIZED_K)
        ], dtype=dtype[weight_dtype])
    weight_scales = np.asarray([
        0.125 if channel % 2 == 0 else 0.25
        for channel in range(OPTIMIZED_N)
    ], dtype=np.float32)
    bias = np.asarray([
        ((channel * 11 + seed) % 41) - 20
        for channel in range(OPTIMIZED_N)
    ], dtype=np.int32)
    fixture: dict[str, object] = {
        "id": case_id,
        "input_dtype": input_dtype,
        "weight_dtype": weight_dtype,
        "output_dtype": output_dtype,
        "input": input_values,
        "weight": weight_values,
        "bias": bias,
        "input_scale": np.float32(0.25),
        "input_zero_point": input_zero_point,
        "weight_scales": weight_scales,
        "weight_zero_points": weight_zero_points,
        "output_scale": np.float32(0.5),
        "output_zero_point": output_zero_point,
    }
    fixture["expected"] = _portable_c_qlinear(fixture)
    return fixture


def _portable_c_qlinear(fixture: dict[str, object]) -> np.ndarray:
    """Mirror the portable-C W8A8 accumulator and two-step F32 transform."""

    input_values = np.asarray(fixture["input"])
    weight_values = np.asarray(fixture["weight"])
    bias = np.asarray(fixture["bias"], dtype=np.int32)
    weight_scales = np.asarray(fixture["weight_scales"], dtype=np.float32)
    weight_zero_points = np.asarray(fixture["weight_zero_points"])
    input_scale = np.float32(fixture["input_scale"])
    output_scale = np.float32(fixture["output_scale"])
    input_zero_point = int(fixture["input_zero_point"])
    output_zero_point = int(fixture["output_zero_point"])
    output_dtype = str(fixture["output_dtype"])
    minimum, maximum = (-128, 127) if output_dtype == "int8" else (0, 255)
    output = np.empty((input_values.shape[0], weight_values.shape[0]), dtype=output_dtype)
    for row in range(input_values.shape[0]):
        for channel in range(weight_values.shape[0]):
            accumulator = int(bias[channel])
            weight_zero_point = int(weight_zero_points[channel])
            for term in range(input_values.shape[1]):
                accumulator += (
                    (int(input_values[row, term]) - input_zero_point)
                    * (int(weight_values[channel, term]) - weight_zero_point)
                )
            product = np.float32(input_scale * weight_scales[channel])
            multiplier = np.float32(product / output_scale)
            scaled = np.float32(np.float32(accumulator) * multiplier)
            transformed = np.float32(scaled + np.float32(output_zero_point))
            quantized = int(np.rint(transformed))
            output[row, channel] = min(max(quantized, minimum), maximum)
    return output


def optimized_fixtures() -> tuple[dict[str, object], ...]:
    return (
        _optimized_fixture("qlinear_tiled_i8_u8_i8", "int8", "uint8", "int8", 3),
        _optimized_fixture("qlinear_tiled_u8_i8_u8", "uint8", "int8", "uint8", 11),
    )


def write_optimized_package(destination: Path, fixture: dict[str, object]) -> None:
    destination.mkdir(parents=True)
    input_dtype = str(fixture["input_dtype"])
    output_dtype = str(fixture["output_dtype"])
    graph = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [OPTIMIZED_ROWS, OPTIMIZED_K], "dtype": input_dtype}},
        "nodes": [{
            "id": "qlinear_tiled_odd_tail",
            "opType": "QLinear",
            "inputs": {"input": "x", "weight": "weight", "bias": "bias"},
            "outputs": {
                "out": {
                    "tensor": "y",
                    "shape": [OPTIMIZED_ROWS, OPTIMIZED_N],
                    "dtype": output_dtype,
                },
            },
            "params": {},
        }],
        "outputs": ["y"],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "x": per_tensor("__quant__.x.scale", "__quant__.x.zero_point"),
                "weight": {
                    "scheme": "per_axis",
                    "axis": 0,
                    "scale_tensor": "__quant__.weight.scale",
                    "zero_point_tensor": "__quant__.weight.zero_point",
                },
                "y": per_tensor("__quant__.y.scale", "__quant__.y.zero_point"),
            },
        },
    }
    (destination / "graph.json").write_text(
        json.dumps(graph, indent=2) + "\n", encoding="utf-8",
    )
    byte_dtype = np.int8 if input_dtype == "int8" else np.uint8
    output_byte_dtype = np.int8 if output_dtype == "int8" else np.uint8
    save_file({
        "weight": np.asarray(fixture["weight"]),
        "bias": np.asarray(fixture["bias"], dtype=np.int32),
        "__quant__.x.scale": np.asarray([fixture["input_scale"]], dtype=np.float32),
        "__quant__.x.zero_point": np.asarray([fixture["input_zero_point"]], dtype=byte_dtype),
        "__quant__.weight.scale": np.asarray(fixture["weight_scales"], dtype=np.float32),
        "__quant__.weight.zero_point": np.asarray(fixture["weight_zero_points"]),
        "__quant__.y.scale": np.asarray([fixture["output_scale"]], dtype=np.float32),
        "__quant__.y.zero_point": np.asarray([fixture["output_zero_point"]], dtype=output_byte_dtype),
    }, os.fspath(destination / "model.safetensors"))


def write_optimized_inputs(destination: Path, fixture: dict[str, object]) -> Path:
    inputs = destination / "inputs"
    inputs.mkdir(parents=True)
    dtype = str(fixture["input_dtype"])
    suffix = ".i8" if dtype == "int8" else ".u8"
    payload = inputs / f"x{suffix}"
    array = np.asarray(fixture["input"])
    payload.write_bytes(array.tobytes(order="C"))
    manifest = destination / "inputs.json"
    manifest.write_text(json.dumps({
        "schema": "volvoxai.quantized-oracle-inputs",
        "version": 1,
        "inputs": [{
            "name": "x",
            "dtype": dtype,
            "shape": [OPTIMIZED_ROWS, OPTIMIZED_K],
            "file": os.fspath(payload.resolve()),
        }],
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def optimized_expected_case(fixture: dict[str, object], destination: Path):
    dtype = str(fixture["output_dtype"])
    suffix = ".i8" if dtype == "int8" else ".u8"
    array = np.asarray(fixture["expected"])
    case = {
        "id": fixture["id"],
        "outputs": [{"name": "y", "dtype": dtype, "comparison": {"mode": "exact"}}],
    }
    destination.mkdir(parents=True)
    (destination / f"y{suffix}").write_bytes(array.tobytes(order="C"))
    metadata = [{
        "name": "y",
        "dtype": dtype,
        "shape": [OPTIMIZED_ROWS, OPTIMIZED_N],
        "byteLength": int(array.nbytes),
        "file": f"y{suffix}",
    }]
    return case, {"y": array}, metadata


def expected_case(batch: int, destination: Path):
    case = {
        "id": f"qlinear_i8_b{batch}",
        "outputs": [{
            "name": "y",
            "dtype": "int8",
            "comparison": {"mode": "exact"},
        }],
    }
    array = EXPECTED[:batch].copy()
    destination.mkdir(parents=True)
    (destination / "y.raw").write_bytes(array.tobytes(order="C"))
    metadata = [{
        "name": "y",
        "dtype": "int8",
        "shape": [batch, 4],
        "byteLength": int(array.nbytes),
        "file": "y.raw",
    }]
    return case, {"y": array}, metadata


def optimized_path_evidence(
    backend: str,
    metadata: dict[str, object],
    *,
    require_physical: bool,
) -> dict[str, object] | None:
    predicate = {
        "rows": OPTIMIZED_ROWS,
        "k": OPTIMIZED_K,
        "n": OPTIMIZED_N,
        "oddKTail": True,
    }
    if backend == "webgpu":
        evidence = metadata.get("executionEvidence", {})
        if (
            not isinstance(evidence, dict)
            or evidence.get("provider") != "webgpu"
            or evidence.get("attested") is not True
        ):
            fail("optimized WebGPU QLinear omitted public route attestation")
        return {
            "proof": "public-route-attestation-plus-deterministic-shape-predicate",
            "provider": "webgpu",
            "attested": True,
            "predicate": predicate,
        }
    if not backend.startswith("native-"):
        return None
    native_backend = backend.removeprefix("native-")
    if native_backend == "vulkan":
        device = metadata.get("physicalDevice")
        packed = device.get("packedInt8Dot") if isinstance(device, dict) else None
        if require_physical and packed != "enabled":
            fail("physical Vulkan optimized QLinear requires advertised packed INT8 dot")
        return {
            "proof": "strict-runtime-plus-deterministic-shape-predicate",
            "selectedFamily": "vulkan.qlinear.tiled",
            "packedDotAdvertised": packed == "enabled",
            "predicate": predicate,
        }
    if native_backend == "opengl":
        return {
            "proof": "strict-runtime-plus-deterministic-shape-predicate",
            "selectedFamily": "opengl.qlinear.tiled",
            "predicate": predicate,
        }
    if native_backend == "metal":
        return {
            "proof": "strict-runtime-plus-deterministic-shape-predicate",
            "selectedFamily": "metal.qlinear.tiled",
            "predicate": predicate,
        }
    if native_backend == "cuda":
        device = metadata.get("physicalDevice")
        compute = device.get("compute") if isinstance(device, dict) else None
        if require_physical:
            try:
                major, minor = (int(part) for part in str(compute).split(".", 1))
            except (TypeError, ValueError):
                fail("physical CUDA optimized QLinear omitted compute capability")
            if (major, minor) < (6, 1):
                fail(f"CUDA compute capability {compute} cannot prove DP4A")
        return {
            "proof": "strict-runtime-plus-deterministic-k-and-grid-predicate",
            "selectedFamily": "cuda.qlinear.warp-dp4a",
            "computeCapability": compute,
            "predicate": predicate,
        }
    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path,
        default=ROOT / "dist" / PACKAGE_VERSION / "volvoxai.js",
    )
    parser.add_argument(
        "--wasm", type=Path,
        default=ROOT / "dist" / PACKAGE_VERSION / "volvoxai.wasm",
    )
    parser.add_argument(
        "--full-bundle", type=Path,
        default=ROOT / "dist" / PACKAGE_VERSION / "volvoxai.full.js",
        help="Full-profile JavaScript bundle used only by --webgpu",
    )
    parser.add_argument(
        "--full-wasm", type=Path,
        default=ROOT / "dist" / PACKAGE_VERSION / "volvoxai.full.wasm",
        help="Full-profile WASM sidecar used only by --webgpu",
    )
    parser.add_argument("--native", type=Path, default=ROOT / "native" / "volvoxai")
    parser.add_argument(
        "--native-backend", action="append",
        choices=("cpu", "vulkan", "opengl", "metal", "cuda"), default=[],
        help="Add a strict native candidate to the always-run CPU candidate; repeatable",
    )
    parser.add_argument("--webgpu", action="store_true")
    parser.add_argument("--deno", default="deno")
    parser.add_argument("--require-physical-webgpu", action="store_true")
    parser.add_argument("--require-physical-native", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.require_physical_webgpu and not args.webgpu:
        fail("--require-physical-webgpu requires --webgpu")
    backends = list(dict.fromkeys(["cpu", *args.native_backend]))
    bundle = args.bundle.resolve()
    wasm = args.wasm.resolve()
    full_bundle = args.full_bundle.resolve()
    full_wasm = args.full_wasm.resolve()
    native = args.native.resolve()
    required_artifacts = [
        ("inference bundle", bundle),
        ("inference WASM", wasm),
        ("native runner", native),
    ]
    if args.webgpu:
        required_artifacts.extend((
            ("full bundle", full_bundle),
            ("full WASM", full_wasm),
        ))
    for label, path in required_artifacts:
        if not path.is_file():
            fail(f"{label} is unavailable at {path}")
    if not os.access(native, os.X_OK):
        fail(f"native runner is not executable: {native}")
    if shutil.which("node") is None:
        fail("node is unavailable")
    if args.webgpu and shutil.which(args.deno) is None:
        fail(f"Deno executable is unavailable: {args.deno}")
    if args.report:
        args.report = args.report.resolve()
        if args.report.exists():
            args.report.unlink()

    temporary = None
    if args.work_dir:
        work = args.work_dir.resolve()
        if work.exists():
            fail("--work-dir must not already exist")
        work.mkdir(parents=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="volvoxai-quantized-oracle-")
        work = Path(temporary.name)
    try:
        package = work / "package"
        write_package(package)
        cases = []
        for batch in (1, 2):
            case_dir = work / f"b{batch}"
            case_dir.mkdir()
            manifest = write_inputs(case_dir, batch)
            case, oracle, oracle_meta = expected_case(batch, case_dir / "canonical")
            results = {}

            destination = case_dir / "wasm"
            metadata = run_wasm(bundle, wasm, package, manifest, destination)
            arrays = load_candidate_arrays(case, destination, metadata, oracle_meta)
            results["wasm"] = compare(case, "wasm", oracle, arrays)

            for backend in backends:
                label = f"native-{backend}"
                destination = case_dir / label
                metadata = run_native(
                    native, package, manifest, destination, backend,
                    args.require_physical_native and backend != "cpu",
                )
                arrays = load_candidate_arrays(case, destination, metadata, oracle_meta)
                results[label] = compare(case, label, oracle, arrays)

            if args.webgpu:
                destination = case_dir / "webgpu"
                metadata = run_webgpu(
                    args.deno, full_bundle, full_wasm, package, manifest, destination,
                    args.require_physical_webgpu,
                )
                arrays = load_candidate_arrays(case, destination, metadata, oracle_meta)
                results["webgpu"] = compare(case, "webgpu", oracle, arrays)
            cases.append({
                "id": case["id"],
                "features": [
                    "odd-k-tail", "asymmetric-zero-points", "ties-to-even", "saturation",
                ],
                "backends": results,
            })
            print(f"quantized-oracle {case['id']}: PASS", flush=True)

        for fixture in optimized_fixtures():
            case_id = str(fixture["id"])
            case_dir = work / case_id
            case_dir.mkdir()
            optimized_package = case_dir / "package"
            write_optimized_package(optimized_package, fixture)
            manifest = write_optimized_inputs(case_dir, fixture)
            case, oracle, oracle_meta = optimized_expected_case(
                fixture, case_dir / "canonical",
            )
            results = {}
            paths = {}

            destination = case_dir / "wasm"
            metadata = run_wasm(bundle, wasm, optimized_package, manifest, destination)
            arrays = load_candidate_arrays(case, destination, metadata, oracle_meta)
            results["wasm"] = compare(case, "wasm", oracle, arrays)

            for backend in backends:
                label = f"native-{backend}"
                destination = case_dir / label
                metadata = run_native(
                    native, optimized_package, manifest, destination, backend,
                    args.require_physical_native and backend != "cpu",
                )
                arrays = load_candidate_arrays(case, destination, metadata, oracle_meta)
                results[label] = compare(case, label, oracle, arrays)
                evidence = optimized_path_evidence(
                    label, metadata,
                    require_physical=args.require_physical_native and backend != "cpu",
                )
                if evidence is not None:
                    paths[label] = evidence

            if args.webgpu:
                destination = case_dir / "webgpu"
                metadata = run_webgpu(
                    args.deno, full_bundle, full_wasm,
                    optimized_package, manifest, destination,
                    args.require_physical_webgpu,
                )
                arrays = load_candidate_arrays(case, destination, metadata, oracle_meta)
                results["webgpu"] = compare(case, "webgpu", oracle, arrays)
                paths["webgpu"] = optimized_path_evidence(
                    "webgpu", metadata,
                    require_physical=args.require_physical_webgpu,
                )
            cases.append({
                "id": case_id,
                "features": [
                    "optimized-tiled-shape", "odd-k-tail", "mixed-signedness",
                    "per-axis-weight-affine", "cpu-canonical-byte-exact",
                ],
                "shape": {"rows": OPTIMIZED_ROWS, "k": OPTIMIZED_K, "n": OPTIMIZED_N},
                "dtypes": {
                    "input": fixture["input_dtype"],
                    "weight": fixture["weight_dtype"],
                    "output": fixture["output_dtype"],
                },
                "pathEvidence": paths,
                "backends": results,
            })
            print(f"quantized-oracle {case_id}: PASS", flush=True)

        report = {
            "schema": "volvoxai.quantized-oracle-report",
            "version": 1,
            "status": "pass",
            "authority": "portable-c-canonical-vectors",
            "providers": [
                "wasm", *(f"native-{backend}" for backend in backends),
                *(["webgpu"] if args.webgpu else []),
            ],
            "cases": cases,
        }
        encoded = json.dumps(report, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HarnessError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"quantized-oracle: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
