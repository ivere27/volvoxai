#!/usr/bin/env python3
"""Required source-ONNX parity gate for ORT and strict VolvoxAI backends."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "cases.json"
EXPORTER = ROOT / "tools" / "export_safetensors.py"
WASM_RUNNER = HERE / "run_wasm.mjs"
WEBGPU_RUNNER = HERE / "run_webgpu.mjs"
PACKAGE_VERSION = json.loads(
    (ROOT / "package.json").read_text(encoding="utf-8")
)["version"]
PINNED = {
    "numpy": "2.2.6",
    "onnx": "1.22.0",
    "onnxruntime": "1.23.2",
    "safetensors": "0.8.0",
}
RUNTIME_DTYPES = {
    "float32": np.dtype(np.float32),
    "int32": np.dtype(np.int32),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
}
DATA_TYPE_NAMES = {
    "DATA_TYPE_F32": "float32",
    "DATA_TYPE_I32": "int32",
    "DATA_TYPE_I8": "int8",
    "DATA_TYPE_U8": "uint8",
}
RAW_SUFFIXES = {
    "float32": ".f32",
    "int32": ".i32",
    "int8": ".i8",
    "uint8": ".u8",
}
TENSOR_PROTO = {
    "float32": TensorProto.FLOAT,
    "int32": TensorProto.INT32,
    "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8,
}
SOFTWARE_ADAPTER = re.compile(
    r"\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software rasterizer|"
    r"microsoft basic render|cpu)\b",
    re.IGNORECASE,
)


class HarnessError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise HarnessError(message)


def value(name: str, dtype: str, shape: list[int | str]):
    return helper.make_tensor_value_info(name, TENSOR_PROTO[dtype], shape)


def initializer(name: str, array: Any):
    return numpy_helper.from_array(np.asarray(array), name=name)


def _build_add_broadcast_dynamic_f32(batch: int):
    base = np.asarray([
        [[-2.0, -0.25, 0.5], [1.0, 2.0, 3.0]],
        [[4.0, -4.0, 0.125], [8.0, 0.0, -1.5]],
        [[-8.0, 0.75, 2.5], [0.25, -3.0, 6.0]],
        [[1.25, -6.0, 4.0], [-0.5, 7.0, -2.25]],
    ], dtype=np.float32)
    feeds = OrderedDict([
        ("x", base[:batch]),
    ])
    bias = np.asarray([[[0.5, -1.0, 2.0]]], dtype=np.float32)
    return {
        "nodes": [helper.make_node("Add", ["x", "bias"], ["y"], name="broadcast_add")],
        "inputs": [value("x", "float32", ["B", 2, 3])],
        "outputs": [value("y", "float32", ["B", 2, 3])],
        "initializers": [initializer("bias", bias)],
        "feeds": feeds,
    }


def build_add_broadcast_dynamic_min_f32():
    return _build_add_broadcast_dynamic_f32(1)


def build_add_broadcast_dynamic_max_f32():
    return _build_add_broadcast_dynamic_f32(4)


def build_erf_gelu_f32():
    feeds = OrderedDict([
        ("x", np.asarray([[[-4.0, -1.5, -0.25, 0.0], [0.25, 1.0, 2.5, 5.0]]], dtype=np.float32)),
    ])
    return {
        "nodes": [
            helper.make_node("Add", ["x", "bias"], ["activation_input"], name="activation_input"),
            helper.make_node("Div", ["activation_input", "sqrt_two"], ["normalized"], name="gelu_div"),
            helper.make_node("Erf", ["normalized"], ["erf"], name="gelu_erf"),
            helper.make_node("Add", ["erf", "one"], ["erf_plus_one"], name="gelu_add"),
            helper.make_node("Mul", ["half", "erf_plus_one"], ["scaled_erf"], name="gelu_scale"),
            helper.make_node("Mul", ["activation_input", "scaled_erf"], ["y"], name="gelu"),
        ],
        "inputs": [value("x", "float32", [1, 2, 4])],
        "outputs": [value("y", "float32", [1, 2, 4])],
        "initializers": [
            initializer("bias", np.asarray([0.125, -0.25, 0.5, -0.75], dtype=np.float32)),
            initializer("sqrt_two", np.asarray(np.sqrt(2.0), dtype=np.float32)),
            initializer("one", np.asarray(1.0, dtype=np.float32)),
            initializer("half", np.asarray(0.5, dtype=np.float32)),
        ],
        "feeds": feeds,
    }


def build_layernorm_linear_f32():
    feeds = OrderedDict([
        ("x", np.asarray([
            [[-2.0, -0.5, 0.25, 1.5], [3.0, -1.0, 2.0, 0.5]],
            [[0.125, 0.25, 0.5, 1.0], [-4.0, 2.0, -1.0, 3.0]],
        ], dtype=np.float32)),
    ])
    weight = np.asarray([
        [0.25, -0.5, 0.75],
        [1.0, 0.125, -0.25],
        [-0.75, 0.5, 0.25],
        [0.5, -1.0, 0.125],
    ], dtype=np.float32)
    return {
        "nodes": [
            helper.make_node(
                "LayerNormalization", ["x", "scale", "ln_bias"], ["normalized"],
                name="layer_norm", axis=-1, epsilon=1e-5,
            ),
            helper.make_node("MatMul", ["normalized", "weight"], ["projected"], name="projection"),
            helper.make_node("Add", ["projected", "out_bias"], ["y"], name="projection_bias"),
        ],
        "inputs": [value("x", "float32", [2, 2, 4])],
        "outputs": [value("y", "float32", [2, 2, 3])],
        "initializers": [
            initializer("scale", np.asarray([1.0, 0.75, 1.25, 0.5], dtype=np.float32)),
            initializer("ln_bias", np.asarray([0.0, 0.1, -0.2, 0.3], dtype=np.float32)),
            initializer("weight", weight),
            initializer("out_bias", np.asarray([0.2, -0.1, 0.05], dtype=np.float32)),
        ],
        "feeds": feeds,
    }


def build_conv_relu_nchw_f32():
    x = (np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4) - 15.5) / 8.0
    weight = (np.arange(54, dtype=np.float32).reshape(3, 2, 3, 3) % 11 - 5.0) / 10.0
    return {
        "nodes": [
            helper.make_node(
                "Conv", ["x", "weight", "bias"], ["convolved"], name="conv",
                auto_pad="NOTSET", pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1], group=1,
            ),
            helper.make_node("Relu", ["convolved"], ["y"], name="relu"),
        ],
        "inputs": [value("x", "float32", [1, 2, 4, 4])],
        "outputs": [value("y", "float32", [1, 3, 4, 4])],
        "initializers": [
            initializer("weight", weight.astype(np.float32)),
            initializer("bias", np.asarray([0.25, -0.5, 0.125], dtype=np.float32)),
        ],
        "feeds": OrderedDict([("x", x)]),
    }


def build_gather_i32_indices_f32():
    data = np.asarray([
        [0.25, -1.0, 2.0, 4.0],
        [8.0, 16.0, -2.0, 0.5],
        [-4.0, 3.0, 1.5, -0.25],
    ], dtype=np.float32)
    indices = np.asarray([2, 0], dtype=np.int32)
    return {
        "nodes": [helper.make_node("Gather", ["data", "indices"], ["y"], name="gather", axis=0)],
        "inputs": [value("data", "float32", [3, 4]), value("indices", "int32", [2])],
        "outputs": [value("y", "float32", [2, 4])],
        "initializers": [],
        "feeds": OrderedDict([("data", data), ("indices", indices)]),
    }


def build_argmax_cast_i32():
    logits = np.asarray([
        [-2.0, 1.0, 3.5, 0.0, 2.0],
        [9.0, 1.0, -4.0, 2.0, 8.5],
    ], dtype=np.float32)
    return {
        "nodes": [
            helper.make_node(
                "ArgMax", ["logits"], ["route_i64"], name="argmax",
                axis=1, keepdims=0, select_last_index=0,
            ),
            helper.make_node("Cast", ["route_i64"], ["y"], name="cast_i32", to=TensorProto.INT32),
        ],
        "inputs": [value("logits", "float32", [2, 5])],
        "outputs": [value("y", "int32", [2])],
        "initializers": [],
        "feeds": OrderedDict([("logits", logits)]),
    }


def build_quantize_linear_u8():
    x = np.asarray([[-100.0, -1.11, -0.49, 0.0, 0.49, 1.11, 7.9, 100.0]], dtype=np.float32)
    return {
        "nodes": [helper.make_node(
            "QuantizeLinear", ["x", "scale", "zero"], ["y"], name="quantize",
        )],
        "inputs": [value("x", "float32", [1, 8])],
        "outputs": [value("y", "uint8", [1, 8])],
        "initializers": [
            initializer("scale", np.asarray(0.25, dtype=np.float32)),
            initializer("zero", np.asarray(128, dtype=np.uint8)),
        ],
        "feeds": OrderedDict([("x", x)]),
    }


def build_masked_attention_odd_seq_f32():
    """One-head explicit ONNX attention with a non-causal hard keep mask."""

    batch, sequence, heads, head_width = 1, 3, 1, 4
    q = np.asarray([
        [[0.5, -1.0, 0.25, 2.0], [1.5, 0.0, -0.5, 0.75], [-1.0, 0.5, 2.0, -0.25]],
    ], dtype=np.float32)
    k = np.asarray([
        [[0.25, 1.0, -0.5, 0.75], [-1.5, 0.25, 1.0, 0.5], [0.5, -0.75, 0.25, 1.25]],
    ], dtype=np.float32)
    v = np.asarray([
        [[1.0, -2.0, 0.5, 0.25], [0.0, 1.5, -1.0, 2.0], [-0.5, 0.75, 2.5, -1.5]],
    ], dtype=np.float32)
    keep = np.asarray([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.int32)
    additive_mask = np.where(keep != 0, 0.0, -np.inf).astype(np.float32)
    split_shape = np.asarray([batch, sequence, heads, head_width], dtype=np.int64)
    output_shape = np.asarray([batch, sequence, heads * head_width], dtype=np.int64)
    return {
        "nodes": [
            helper.make_node("Reshape", ["q", "split_shape"], ["q_split"], name="q_split"),
            helper.make_node("Transpose", ["q_split"], ["q_heads"], name="q_heads", perm=[0, 2, 1, 3]),
            helper.make_node("Reshape", ["k", "split_shape"], ["k_split"], name="k_split"),
            helper.make_node("Transpose", ["k_split"], ["k_heads"], name="k_heads", perm=[0, 2, 3, 1]),
            helper.make_node("Reshape", ["v", "split_shape"], ["v_split"], name="v_split"),
            helper.make_node("Transpose", ["v_split"], ["v_heads"], name="v_heads", perm=[0, 2, 1, 3]),
            helper.make_node("MatMul", ["q_heads", "k_heads"], ["scores"], name="qk"),
            helper.make_node("Mul", ["scores", "scale"], ["scaled_scores"], name="scale_scores"),
            helper.make_node("Add", ["scaled_scores", "additive_mask"], ["masked_scores"], name="hard_mask"),
            helper.make_node("Softmax", ["masked_scores"], ["probabilities"], name="probabilities", axis=-1),
            helper.make_node("MatMul", ["probabilities", "v_heads"], ["context_heads"], name="pv"),
            helper.make_node("Transpose", ["context_heads"], ["context_rows"], name="restore_heads", perm=[0, 2, 1, 3]),
            helper.make_node("Reshape", ["context_rows", "output_shape"], ["y"], name="restore_width"),
        ],
        "inputs": [
            value("q", "float32", [batch, sequence, heads * head_width]),
            value("k", "float32", [batch, sequence, heads * head_width]),
            value("v", "float32", [batch, sequence, heads * head_width]),
        ],
        "outputs": [value("y", "float32", [batch, sequence, heads * head_width])],
        "initializers": [
            initializer("split_shape", split_shape),
            initializer("output_shape", output_shape),
            initializer("scale", np.asarray(0.5, dtype=np.float32)),
            initializer("additive_mask", additive_mask.reshape(1, 1, sequence, sequence)),
        ],
        "feeds": OrderedDict((("q", q), ("k", k), ("v", v))),
    }


BUILDERS: dict[str, Callable[[], dict[str, Any]]] = {
    function.__name__.removeprefix("build_"): function
    for function in (
        build_add_broadcast_dynamic_min_f32,
        build_add_broadcast_dynamic_max_f32,
        build_erf_gelu_f32,
        build_layernorm_linear_f32,
        build_conv_relu_nchw_f32,
        build_gather_i32_indices_f32,
        build_argmax_cast_i32,
        build_quantize_linear_u8,
        build_masked_attention_odd_seq_f32,
    )
}


def check_dependencies() -> None:
    mismatches = []
    for package, expected in PINNED.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            mismatches.append(f"{package}=={actual} (expected {expected})")
    if mismatches:
        fail(
            "pinned Python environment mismatch: " + ", ".join(mismatches)
            + f"; install {HERE / 'requirements.txt'} deliberately before running"
        )


def load_config(selected: set[str]) -> tuple[int, list[dict[str, Any]]]:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    if document.get("schema") != "volvoxai.onnx-oracle-cases" or document.get("version") != 1:
        fail("cases.json has an unsupported schema/version")
    opset = document.get("opset")
    cases = document.get("cases")
    if not isinstance(opset, int) or opset < 17 or not isinstance(cases, list) or not cases:
        fail("cases.json must declare opset >= 17 and a non-empty cases array")
    ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if len(ids) != len(cases) or any(not isinstance(case_id, str) or not case_id for case_id in ids):
        fail("every case must have a non-empty string id")
    if len(set(ids)) != len(ids):
        fail("case ids must be unique")
    unknown = selected - set(ids)
    if unknown:
        fail(f"unknown selected case(s): {', '.join(sorted(unknown))}")
    chosen = [case for case in cases if not selected or case["id"] in selected]
    for case in chosen:
        if case.get("builder") not in BUILDERS:
            fail(f"{case['id']}: unknown builder {case.get('builder')!r}")
        if case.get("oracle_class") not in {"onnx-direct", "onnx-decomposition"}:
            fail(f"{case['id']}: executable case must classify its ONNX relationship")
        if not isinstance(case.get("expected_lowered_ops"), list):
            fail(f"{case['id']}: expected_lowered_ops must be an array")
        lifecycle = case.get("webgpu_lifecycle")
        if lifecycle is not None:
            if (
                not isinstance(lifecycle, dict)
                or set(lifecycle) != {"next_builder", "symbol"}
                or lifecycle.get("next_builder") not in BUILDERS
                or not isinstance(lifecycle.get("symbol"), str)
                or lifecycle["symbol"] not in case.get("dimension_bounds", {})
            ):
                fail(f"{case['id']}: malformed WebGPU dynamic lifecycle declaration")
        outputs = case.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            fail(f"{case['id']}: outputs must be a non-empty array")
        for output in outputs:
            if output.get("dtype") not in RUNTIME_DTYPES:
                fail(f"{case['id']}: unsupported output dtype {output.get('dtype')!r}")
            comparison = output.get("comparison", {})
            if comparison.get("mode") == "exact":
                if output["dtype"] not in {"int32", "int8", "uint8"}:
                    fail(f"{case['id']}: exact mode is reserved for integer/quantized outputs")
            elif comparison.get("mode") == "tolerance":
                if output["dtype"] != "float32" or any(
                    not isinstance(comparison.get(name), (int, float)) or comparison[name] < 0
                    for name in ("rtol", "atol")
                ):
                    fail(f"{case['id']}: float tolerance must declare non-negative rtol/atol")
            else:
                fail(f"{case['id']}: unsupported comparison policy")
    return opset, chosen


def write_model(destination: Path, case_id: str, opset: int, specification: dict[str, Any]) -> None:
    graph = helper.make_graph(
        specification["nodes"],
        case_id,
        specification["inputs"],
        specification["outputs"],
        specification["initializers"],
    )
    model = helper.make_model(
        graph,
        producer_name="volvoxai-required-onnx-oracle",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    # IR v10 is sufficient for this suite and accepted by the pinned ORT.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def run_ort(model_path: Path, feeds: OrderedDict[str, np.ndarray], outputs: list[dict[str, Any]], destination: Path):
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_mem_pattern = False
    session = ort.InferenceSession(
        os.fspath(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        fail(f"{model_path.name}: ORT did not select only CPUExecutionProvider")
    source_inputs = session.get_inputs()
    if [entry.name for entry in source_inputs] != list(feeds):
        fail(f"{model_path.name}: ORT input order differs from authored source graph")
    output_names = [output["name"] for output in outputs]
    if [entry.name for entry in session.get_outputs()] != output_names:
        fail(f"{model_path.name}: ORT output names differ from case policy")
    results = session.run(output_names, dict(feeds))
    destination.mkdir(parents=True)
    metadata = []
    arrays = {}
    for policy, result in zip(outputs, results):
        array = np.ascontiguousarray(result)
        dtype = RUNTIME_DTYPES[policy["dtype"]]
        if array.dtype != dtype:
            fail(f"{model_path.name}/{policy['name']}: ORT dtype {array.dtype} != {dtype}")
        if array.size == 0 or not all(isinstance(extent, int) and extent > 0 for extent in array.shape):
            fail(f"{model_path.name}/{policy['name']}: ORT returned an empty/invalid shape {array.shape}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            fail(f"{model_path.name}/{policy['name']}: ORT returned non-finite values")
        filename = f"{policy['name']}.raw"
        (destination / filename).write_bytes(array.tobytes(order="C"))
        metadata.append({
            "name": policy["name"],
            "dtype": policy["dtype"],
            "shape": list(array.shape),
            "byteLength": array.nbytes,
            "file": filename,
        })
        arrays[policy["name"]] = array
    (destination / "outputs.json").write_text(json.dumps({
        "schema": "volvoxai.onnx-oracle-output",
        "version": 1,
        "backend": "onnxruntime-cpu",
        "outputs": metadata,
    }, indent=2) + "\n", encoding="utf-8")
    return arrays, metadata


def export_package(case: dict[str, Any], model_path: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True)
    command = [
        sys.executable,
        os.fspath(EXPORTER),
        "--model", os.fspath(model_path),
        "--out", os.fspath(destination / "model.safetensors"),
        "--weight-dtype", "float32",
        "--target", "portable",
        "--report", os.fspath(destination.parent / "exporter-report.json"),
        "--report-format", "json",
    ]
    for output in case["outputs"]:
        command.extend(("--output-name", output["name"]))
    for symbol, bounds in sorted(case.get("dimension_bounds", {}).items()):
        encoded = f"{symbol}={bounds['min']}:{bounds['max']}:{bounds.get('multiple_of', 1)}"
        command.extend(("--dimension-bound", encoded))
    run_command(command, cwd=ROOT)
    graph = json.loads((destination / "graph.json").read_text(encoding="utf-8"))
    actual_ops = [node.get("opType") for node in graph.get("nodes", [])]
    if actual_ops != case["expected_lowered_ops"]:
        fail(
            f"{case['id']}: lowered op inventory {actual_ops!r} != "
            f"required {case['expected_lowered_ops']!r}"
        )
    if graph.get("outputs") != [output["name"] for output in case["outputs"]]:
        fail(f"{case['id']}: exported output inventory differs from case policy")
    return graph


def write_candidate_inputs(
    case_dir: Path,
    graph: dict[str, Any],
    feeds: OrderedDict[str, np.ndarray],
    *,
    allow_one_past_symbol: str | None = None,
) -> Path:
    graph_inputs = list(graph.get("inputs", {}))
    if len(graph_inputs) != len(feeds):
        fail("exported graph input count differs from source ONNX")
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir()
    records = []
    for index, ((source_name, array), package_name) in enumerate(zip(feeds.items(), graph_inputs)):
        expected_package_name = f"input{index}"
        if package_name != expected_package_name:
            fail(
                f"{case_dir.name}: source input {source_name!r} at index {index} "
                f"lowered to {package_name!r}, expected {expected_package_name!r}"
            )
        descriptor = graph["inputs"][package_name]
        dtype_name = next((name for name, dtype in RUNTIME_DTYPES.items() if dtype == array.dtype), None)
        if dtype_name is None or descriptor.get("dtype") != dtype_name:
            fail(f"{case_dir.name}/{source_name}: exported input dtype differs from source fixture")
        if len(descriptor.get("shape", [])) != array.ndim:
            fail(f"{case_dir.name}/{source_name}: exported input rank differs from source fixture")
        for axis, (declared, actual) in enumerate(zip(descriptor["shape"], array.shape)):
            if isinstance(declared, int):
                valid = declared == actual
            else:
                domain = graph.get("dimensions", {}).get(declared, {})
                multiple = domain.get("multiple_of", 1)
                valid = (
                    isinstance(domain.get("min"), int)
                    and isinstance(domain.get("max"), int)
                    and domain["min"] <= actual <= domain["max"]
                    and actual % multiple == 0
                )
                if (
                    not valid
                    and declared == allow_one_past_symbol
                    and isinstance(domain.get("max"), int)
                    and actual == domain["max"] + multiple
                ):
                    valid = True
            if not valid:
                fail(
                    f"{case_dir.name}/{source_name}: fixture axis {axis}={actual} "
                    f"is outside exported dimension {declared!r}"
                )
        filename = f"input{index}{RAW_SUFFIXES[dtype_name]}"
        path = inputs_dir / filename
        path.write_bytes(np.ascontiguousarray(array).tobytes(order="C"))
        records.append({
            "name": package_name,
            "sourceName": source_name,
            "dtype": dtype_name,
            "shape": list(array.shape),
            "file": os.fspath(path.resolve()),
        })
    manifest = case_dir / "inputs.json"
    manifest.write_text(json.dumps({
        "schema": "volvoxai.onnx-oracle-inputs",
        "version": 1,
        "inputs": records,
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def _changed_fixture(feeds: OrderedDict[str, np.ndarray]) -> OrderedDict[str, np.ndarray]:
    changed = OrderedDict((name, np.asarray(array).copy()) for name, array in feeds.items())
    for array in changed.values():
        if not array.size:
            continue
        if np.issubdtype(array.dtype, np.floating):
            array.reshape(-1)[0] = np.asarray(array.reshape(-1)[0] + 0.75, dtype=array.dtype)
        elif np.issubdtype(array.dtype, np.signedinteger):
            value = int(array.reshape(-1)[0])
            bounds = np.iinfo(array.dtype)
            array.reshape(-1)[0] = value + 1 if value < bounds.max else value - 1
        elif np.issubdtype(array.dtype, np.unsignedinteger):
            value = int(array.reshape(-1)[0])
            bounds = np.iinfo(array.dtype)
            array.reshape(-1)[0] = value + 1 if value < bounds.max else value - 1
        else:
            continue
        return changed
    fail("WebGPU lifecycle fixture has no mutable numerical input")


def write_webgpu_lifecycle_inputs(
    case: dict[str, Any],
    case_dir: Path,
    graph: dict[str, Any],
    first_feeds: OrderedDict[str, np.ndarray],
) -> dict[str, Path] | None:
    lifecycle = case.get("webgpu_lifecycle")
    if lifecycle is None:
        return None
    next_feeds = BUILDERS[lifecycle["next_builder"]]()["feeds"]
    if list(next_feeds) != list(first_feeds):
        fail(f"{case['id']}: lifecycle min/max source input order differs")
    symbol = lifecycle["symbol"]
    bounds = graph.get("dimensions", {}).get(symbol)
    if not isinstance(bounds, dict) or not isinstance(bounds.get("max"), int):
        fail(f"{case['id']}: lifecycle symbol {symbol!r} has no maximum")

    invalid_feeds = OrderedDict((name, np.asarray(array).copy()) for name, array in next_feeds.items())
    extended = False
    for index, (source_name, array) in enumerate(invalid_feeds.items()):
        package_name = f"input{index}"
        shape = graph.get("inputs", {}).get(package_name, {}).get("shape", [])
        axes = [axis for axis, extent in enumerate(shape) if extent == symbol]
        for axis in axes:
            if array.shape[axis] != bounds["max"]:
                fail(
                    f"{case['id']}/{source_name}: lifecycle max fixture axis {axis} "
                    f"is {array.shape[axis]}, expected {bounds['max']}"
                )
            selector = [slice(None)] * array.ndim
            selector[axis] = slice(0, 1)
            invalid_feeds[source_name] = np.concatenate(
                (array, array[tuple(selector)]), axis=axis,
            )
            extended = True
    if not extended:
        fail(f"{case['id']}: no input binds lifecycle symbol {symbol!r}")

    manifests = {}
    for name, feeds in (
        ("next", next_feeds),
        ("invalid", invalid_feeds),
        ("changed", _changed_fixture(first_feeds)),
    ):
        destination = case_dir / f"webgpu-lifecycle-{name}"
        destination.mkdir()
        manifests[name] = write_candidate_inputs(
            destination, graph, feeds,
            allow_one_past_symbol=symbol if name == "invalid" else None,
        )
    return manifests


def run_wasm(bundle: Path, wasm: Path, package: Path, inputs: Path, destination: Path) -> dict[str, Any]:
    run_command([
        "node", os.fspath(WASM_RUNNER),
        "--bundle", os.fspath(bundle),
        "--wasm", os.fspath(wasm),
        "--package", os.fspath(package),
        "--inputs", os.fspath(inputs),
        "--out", os.fspath(destination),
    ], cwd=ROOT)
    return json.loads((destination / "outputs.json").read_text(encoding="utf-8"))


def run_native(
    native: Path,
    package: Path,
    input_manifest: Path,
    destination: Path,
    backend: str,
    require_physical: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    graph = json.loads((package / "graph.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True)
    report_path = destination / "runtime-report.json"
    command = [
        os.fspath(native), "run", os.fspath(package), f"--{backend}", "--threads", "1",
        "--report-json", os.fspath(report_path),
    ]
    for descriptor in manifest["inputs"]:
        shape = ",".join(str(extent) for extent in descriptor["shape"])
        command.extend(("--input", f"{descriptor['name']}[{shape}]={descriptor['file']}"))
    output_dtypes = {
        descriptor["tensor"]: descriptor["dtype"]
        for node in graph["nodes"]
        for descriptor in node["outputs"].values()
    }
    for output in graph["outputs"]:
        dtype = output_dtypes.get(output)
        if dtype not in RAW_SUFFIXES:
            fail(f"native output {output!r} has no supported graph dtype")
        command.extend(("--output", f"{output}={destination / (output + RAW_SUFFIXES[dtype])}"))
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    completed = run_command(command, cwd=ROOT, environment=environment)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "volvoxai.native-cli.run-report":
        fail(
            f"native {backend} did not publish a run report: "
            f"schema={report.get('schema')!r}"
        )
    compilation = report.get("compile", {})
    execution = report.get("execute", {})
    for label, stage in (("compile", compilation), ("execute", execution)):
        if stage.get("status") != "NATIVE_STATUS_OK":
            fail(
                f"native {backend} {label} did not succeed: "
                f"{stage.get('status')!r} {stage.get('code')!r} {stage.get('message')!r}"
            )
        if stage.get("backend") != backend:
            fail(f"native --{backend} {label} selected unexpected backend {stage.get('backend')!r}")
    expected_device = "host" if backend == "cpu" else f"builtin:{backend}"
    if compilation.get("device") != expected_device or execution.get("device") != expected_device:
        fail(f"native {backend} report did not preserve its requested device identity")
    # The report no longer carries a separate policy block. Strict no-fallback is
    # attested by the route counters instead: every active node was selected on
    # the requested built-in provider, with nothing fell back or missing.
    for label, stage in (("compile", compilation), ("execute", execution)):
        route = stage.get("route")
        if not isinstance(route, dict):
            fail(f"native {backend} {label} omitted its execution route")
        if route.get("provider") != backend or route.get("builtin") is not True:
            fail(f"native {backend} {label} route named provider {route.get('provider')!r}")
        if route.get("attested") is not True:
            fail(f"native {backend} {label} route is not attested")
        if route.get("fallbackNodes") or route.get("missingNodes"):
            fail(
                f"native {backend} {label} did not attest a strict no-fallback route: "
                f"fallbackNodes={route.get('fallbackNodes')} missingNodes={route.get('missingNodes')}"
            )
        if route.get("selectedNodes") != route.get("activeNodes"):
            fail(
                f"native {backend} {label} selected {route.get('selectedNodes')} of "
                f"{route.get('activeNodes')} active nodes"
            )
    physical_device = (
        validate_physical_native(backend, completed.stdout, completed.stderr)
        if require_physical else None
    )
    outputs = report.get("outputs")
    if not isinstance(outputs, list):
        fail("native CPU report omitted output descriptors")
    for descriptor in outputs:
        dtype = DATA_TYPE_NAMES.get(descriptor.get("dtype"))
        if dtype not in RAW_SUFFIXES:
            fail(f"native output {descriptor.get('name')!r} reports unsupported dtype {descriptor.get('dtype')!r}")
        descriptor["dtype"] = dtype
        filename = f"{descriptor.get('name')}{RAW_SUFFIXES[dtype]}"
        path = destination / filename
        if not path.is_file() or path.stat().st_size != descriptor.get("byteLength"):
            fail(f"native output {descriptor.get('name')!r} raw bytes differ from report")
        descriptor["file"] = filename
    metadata = {
        "schema": "volvoxai.onnx-oracle-output",
        "version": 1,
        "backend": f"native-{backend}",
        "outputs": outputs,
        "runtimeEvidence": {
            "report": os.fspath(report_path.resolve()),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        **({"physicalDevice": physical_device} if physical_device is not None else {}),
    }
    (destination / "outputs.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def validate_physical_native(
    backend: str, stdout: str, stderr: str,
) -> dict[str, str]:
    if backend == "cpu":
        fail("physical native evidence cannot be requested for CPU")
    if backend == "vulkan":
        matches = re.findall(
            r"^\[VolvoxAI GPU\] Vulkan Compute initialized successfully! "
            r"Device: ([^;\r\n]+); packed INT8 dot: (enabled|unavailable)$",
            stdout, re.MULTILINE,
        )
        if len(matches) != 1:
            fail(f"native Vulkan published {len(matches)} physical adapter identities")
        if matches[0][0].strip().casefold() in {"unknown", "gpu"}:
            fail("native Vulkan did not publish a concrete physical adapter identity")
        identity = {"backend": backend, "device": matches[0][0], "packedInt8Dot": matches[0][1]}
    elif backend == "opengl":
        matches = re.findall(
            r"^\[VolvoxAI GPU\] OpenGL Compute initialized: "
            r"([^\r\n]+?) / ([^\r\n]+?) / ([^\r\n]+)$",
            stdout, re.MULTILINE,
        )
        if len(matches) != 1:
            fail(f"native OpenGL published {len(matches)} physical adapter identities")
        if any(value.strip().casefold() == "unknown" for value in matches[0][:2]):
            fail("native OpenGL did not publish concrete vendor and renderer identities")
        identity = {
            "backend": backend, "vendor": matches[0][0],
            "device": matches[0][1], "version": matches[0][2],
        }
    elif backend == "metal":
        matches = re.findall(
            r"^\[VolvoxAI GPU\] Metal initialized successfully on: ([^\r\n]+)$",
            stdout, re.MULTILINE,
        )
        if len(matches) != 1:
            fail(f"native Metal published {len(matches)} physical adapter identities")
        if matches[0].strip().casefold() in {"unknown", "gpu"}:
            fail("native Metal did not publish a concrete physical adapter identity")
        identity = {"backend": backend, "device": matches[0]}
    elif backend == "cuda":
        matches = re.findall(
            r"^\[CUDA\] device ([0-9]+): ([^\r\n]+?) "
            r"\(compute ([0-9]+\.[0-9]+)\)$",
            stderr, re.MULTILINE,
        )
        if not matches:
            fail("native CUDA did not publish a selected-device identity")
        if len(set(matches)) != 1:
            fail("native CUDA published conflicting selected-device identities")
        device_index, runtime_name, compute_capability = matches[0]
        if runtime_name.strip().casefold() in {"unknown", "gpu", "nvidia gpu"}:
            fail("native CUDA did not publish a concrete selected-device identity")
        executable = shutil.which("nvidia-smi")
        if executable is None:
            fail("physical CUDA evidence requires nvidia-smi")
        completed = run_command([
            executable, f"--id={device_index}", "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ], cwd=ROOT)
        devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if len(devices) != 1 or "," not in devices[0]:
            fail("nvidia-smi did not publish exactly the selected CUDA adapter")
        smi_name, driver = (part.strip() for part in devices[0].rsplit(",", 1))
        if smi_name.casefold() != runtime_name.strip().casefold():
            fail(
                f"CUDA runtime selected {runtime_name!r}, but nvidia-smi index "
                f"{device_index} reports {smi_name!r}"
            )
        identity = {
            "backend": backend, "deviceIndex": device_index,
            "device": runtime_name.strip(), "compute": compute_capability,
            "driver": driver, "initializations": str(len(matches)),
        }
    else:
        fail(f"unsupported physical native backend {backend!r}")
    description = " ".join(identity.values())
    if SOFTWARE_ADAPTER.search(description):
        fail(f"native {backend} software adapter rejected ({description})")
    required = os.environ.get("VOLVOXAI_PARITY_GPU_ADAPTER", "").strip()
    if required and required.casefold() not in description.casefold():
        fail(f"native {backend} adapter {description!r} does not match {required!r}")
    return identity


def run_webgpu(
    deno: str,
    bundle: Path,
    wasm: Path,
    package: Path,
    inputs: Path,
    destination: Path,
    require_physical: bool,
    *,
    lifecycle_inputs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    command = [
        deno, "run", "--unstable-webgpu", "--allow-read", "--allow-write",
        "--allow-env", "--allow-ffi", os.fspath(WEBGPU_RUNNER),
        "--bundle", os.fspath(bundle),
        "--wasm", os.fspath(wasm),
        "--package", os.fspath(package),
        "--inputs", os.fspath(inputs),
        "--out", os.fspath(destination),
    ]
    if require_physical:
        command.append("--require-physical")
    if lifecycle_inputs is not None:
        command.extend((
            "--next-inputs", os.fspath(lifecycle_inputs["next"]),
            "--invalid-inputs", os.fspath(lifecycle_inputs["invalid"]),
            "--changed-inputs", os.fspath(lifecycle_inputs["changed"]),
        ))
    run_command(command, cwd=ROOT)
    return json.loads((destination / "outputs.json").read_text(encoding="utf-8"))


def load_candidate_arrays(
    case: dict[str, Any],
    destination: Path,
    metadata: dict[str, Any],
    oracle_meta: list[dict[str, Any]],
):
    if metadata.get("schema") != "volvoxai.onnx-oracle-output" or metadata.get("version") != 1:
        fail(f"{case['id']}/{metadata.get('backend')}: invalid output artifact schema")
    actual = metadata.get("outputs")
    if not isinstance(actual, list):
        fail(f"{case['id']}/{metadata.get('backend')}: output metadata is missing")
    if [item.get("name") for item in actual] != [item["name"] for item in oracle_meta]:
        fail(f"{case['id']}/{metadata.get('backend')}: output names/order differ from ORT")
    arrays = {}
    for candidate, oracle in zip(actual, oracle_meta):
        for field in ("name", "dtype", "shape", "byteLength"):
            if candidate.get(field) != oracle[field]:
                fail(
                    f"{case['id']}/{metadata.get('backend')}/{oracle['name']}: "
                    f"{field}={candidate.get(field)!r} differs from ORT {oracle[field]!r}"
                )
        dtype = RUNTIME_DTYPES[oracle["dtype"]]
        payload = (destination / candidate["file"]).read_bytes()
        if len(payload) != oracle["byteLength"]:
            fail(f"{case['id']}/{metadata.get('backend')}/{oracle['name']}: raw byte length mismatch")
        arrays[oracle["name"]] = np.frombuffer(payload, dtype=dtype).reshape(oracle["shape"]).copy()
    return arrays


def compare(case: dict[str, Any], backend: str, oracle: dict[str, np.ndarray], candidate: dict[str, np.ndarray]):
    summaries = []
    for policy in case["outputs"]:
        name = policy["name"]
        expected = oracle[name]
        actual = candidate[name]
        comparison = policy["comparison"]
        if comparison["mode"] == "exact":
            if expected.tobytes(order="C") != actual.tobytes(order="C"):
                unequal = np.flatnonzero(expected.reshape(-1) != actual.reshape(-1))
                index = int(unequal[0]) if unequal.size else -1
                detail = ""
                if index >= 0:
                    detail = (
                        f" at flat index {index}: {expected.reshape(-1)[index]} "
                        f"!= {actual.reshape(-1)[index]}"
                    )
                fail(
                    f"{case['id']}/{backend}/{name}: exact bytes differ{detail}"
                )
            summaries.append({"name": name, "mode": "exact", "elements": int(expected.size)})
            continue
        if not np.isfinite(actual).all():
            fail(f"{case['id']}/{backend}/{name}: candidate returned non-finite values")
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        allowed = comparison["atol"] + comparison["rtol"] * np.abs(expected.astype(np.float64))
        bad = np.flatnonzero(difference.reshape(-1) > allowed.reshape(-1))
        scale = np.maximum(np.abs(expected.astype(np.float64)), 1e-30)
        relative = difference / scale
        max_abs = float(np.max(difference))
        max_rel = float(np.max(relative))
        if bad.size:
            index = int(bad[0])
            fail(
                f"{case['id']}/{backend}/{name}: tolerance exceeded at flat index {index}; "
                f"ORT={expected.reshape(-1)[index]!r} candidate={actual.reshape(-1)[index]!r} "
                f"abs={difference.reshape(-1)[index]:.8g} allowed={allowed.reshape(-1)[index]:.8g}"
            )
        summaries.append({
            "name": name,
            "mode": "tolerance",
            "elements": int(expected.size),
            "rtol": comparison["rtol"],
            "atol": comparison["atol"],
            "max_abs": max_abs,
            "max_rel": max_rel,
        })
    return summaries


def execute_case(
    case: dict[str, Any],
    opset: int,
    root: Path,
    bundle: Path,
    wasm: Path,
    full_bundle: Path,
    full_wasm: Path,
    native: Path,
    native_backends: list[str],
    webgpu: bool,
    deno: str,
    require_physical_webgpu: bool,
    require_physical_native: bool,
):
    case_dir = root / case["id"]
    case_dir.mkdir()
    specification = BUILDERS[case["builder"]]()
    model_path = case_dir / "source.onnx"
    write_model(model_path, case["id"], opset, specification)
    oracle, oracle_meta = run_ort(model_path, specification["feeds"], case["outputs"], case_dir / "ort")
    package = case_dir / "package"
    graph = export_package(case, model_path, package)
    input_manifest = write_candidate_inputs(case_dir, graph, specification["feeds"])
    lifecycle_inputs = (
        write_webgpu_lifecycle_inputs(case, case_dir, graph, specification["feeds"])
        if webgpu else None
    )
    wasm_dir = case_dir / "wasm"
    wasm_metadata = run_wasm(bundle, wasm, package, input_manifest, wasm_dir)
    wasm_arrays = load_candidate_arrays(case, wasm_dir, wasm_metadata, oracle_meta)
    comparisons = {
        "wasm": compare(case, "wasm", oracle, wasm_arrays),
    }
    for backend in native_backends:
        label = f"native-{backend}"
        destination = case_dir / label
        metadata = run_native(
            native, package, input_manifest, destination, backend,
            require_physical_native and backend != "cpu",
        )
        arrays = load_candidate_arrays(case, destination, metadata, oracle_meta)
        comparisons[label] = compare(case, label, oracle, arrays)
    if webgpu:
        destination = case_dir / "webgpu"
        metadata = run_webgpu(
            deno, full_bundle, full_wasm, package, input_manifest, destination,
            require_physical_webgpu,
            lifecycle_inputs=lifecycle_inputs,
        )
        arrays = load_candidate_arrays(case, destination, metadata, oracle_meta)
        comparisons["webgpu"] = compare(case, "webgpu", oracle, arrays)
    result = {
        "id": case["id"],
        "oracle_class": case["oracle_class"],
        "lowered_ops": case["expected_lowered_ops"],
        "backends": comparisons,
    }
    if lifecycle_inputs is not None:
        lifecycle = metadata.get("stableResult", {}).get("dynamicLifecycle")
        if lifecycle != {
            "minToMax": True,
            "invalidRejected": True,
            "invalidCode": "INVALID_ARGUMENT",
            "changedMinObserved": True,
            "returnedToMinShape": True,
        }:
            fail(f"{case['id']}: WebGPU dynamic lifecycle evidence is incomplete")
        lifecycle_outputs = metadata.get("lifecycleOutputs")
        if not isinstance(lifecycle_outputs, dict):
            fail(f"{case['id']}: WebGPU dynamic lifecycle outputs are missing")
        lifecycle_comparisons = {}
        lifecycle_feeds = {
            "next": BUILDERS[case["webgpu_lifecycle"]["next_builder"]]()["feeds"],
            "changed": _changed_fixture(specification["feeds"]),
        }
        for phase, feeds in lifecycle_feeds.items():
            phase_oracle, phase_meta = run_ort(
                model_path, feeds, case["outputs"],
                case_dir / f"webgpu-lifecycle-ort-{phase}",
            )
            candidate_metadata = {
                "schema": "volvoxai.onnx-oracle-output",
                "version": 1,
                "backend": f"webgpu-lifecycle-{phase}",
                "outputs": lifecycle_outputs.get(phase),
            }
            arrays = load_candidate_arrays(
                case, destination, candidate_metadata, phase_meta,
            )
            lifecycle_comparisons[phase] = compare(
                case, f"webgpu-lifecycle-{phase}", phase_oracle, arrays,
            )
        result["webgpu_dynamic_lifecycle"] = {
            "contract": lifecycle,
            "onnx_comparisons": lifecycle_comparisons,
        }
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], help="Run one case id; repeatable")
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
        "--native-backend", action="append", choices=("cpu", "vulkan", "opengl", "metal", "cuda"),
        default=[], help="Add a strict native backend to the always-run CPU candidate; repeatable",
    )
    parser.add_argument(
        "--webgpu", action="store_true",
        help="Also run the strict Deno WebGPU candidate",
    )
    parser.add_argument("--deno", default="deno", help="Deno executable used by --webgpu")
    parser.add_argument(
        "--require-physical-webgpu", action="store_true",
        help="Reject missing or software WebGPU adapter identity (requires --webgpu)",
    )
    parser.add_argument(
        "--require-physical-native", action="store_true",
        help="Reject software or missing native GPU adapter identity",
    )
    parser.add_argument("--work-dir", type=Path, help="Preserve artifacts in a new directory")
    parser.add_argument("--report", type=Path, help="Write the final JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.require_physical_webgpu and not args.webgpu:
        fail("--require-physical-webgpu requires --webgpu")
    native_backends = list(dict.fromkeys(["cpu", *args.native_backend]))
    if args.report:
        args.report = args.report.resolve()
        if args.report.exists():
            args.report.unlink()
    check_dependencies()
    opset, cases = load_config(set(args.case))
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
            fail(f"{label} is unavailable at {path}; build it before running the oracle")
    if not os.access(native, os.X_OK):
        fail(f"native runner is not executable: {native}")
    if shutil.which("node") is None:
        fail("node is unavailable")
    if args.webgpu and shutil.which(args.deno) is None:
        fail(f"Deno executable is unavailable: {args.deno}")

    temporary = None
    if args.work_dir:
        work = args.work_dir.resolve()
        if work.exists():
            fail(f"--work-dir must not already exist: {work}")
        work.mkdir(parents=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="volvoxai-onnx-oracle-")
        work = Path(temporary.name)
    try:
        results = []
        for case in cases:
            print(f"onnx-oracle {case['id']}: running", flush=True)
            result = execute_case(
                case, opset, work, bundle, wasm, full_bundle, full_wasm,
                native, native_backends,
                args.webgpu, args.deno, args.require_physical_webgpu,
                args.require_physical_native,
            )
            results.append(result)
            print(f"onnx-oracle {case['id']}: PASS", flush=True)
        report = {
            "schema": "volvoxai.onnx-oracle-report",
            "version": 1,
            "status": "pass",
            "providers": [
                "onnxruntime:CPUExecutionProvider", "wasm",
                *(f"native-{backend}" for backend in native_backends),
                *(["webgpu"] if args.webgpu else []),
            ],
            "dependencies": PINNED,
            "opset": opset,
            "cases": results,
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
        print(f"onnx-oracle: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
