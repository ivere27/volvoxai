#!/usr/bin/env python3
"""Assemble independently exported TinyReceiptVQA ONNX graph packages.

This tool is deliberately non-transforming.  It accepts one already-exported
router directory and exactly one already-exported directory for each explicit
TinyReceipt family.  Every input directory must contain the canonical
``graph.json`` and ``model.safetensors`` files.  The files are copied byte for
byte into a self-contained suite and described by a precision-neutral manifest.

The assembler never reads a PyTorch/Hugging Face checkpoint, never inspects
SafeTensors entries, and never infers family names, ordering, routing semantics,
precision, or backend support from graph topology.  Those facts are explicit
arguments and are validated against the public graph interface where possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.numpy import safe_open


# Keep the application-specific assembler under examples/, while reusing the
# generic exporter's conservative descriptor matrix instead of maintaining a
# second target registry here.  This also makes direct invocation work from a
# directory other than the repository root.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.exporter.capabilities import TARGETS, expand_targets, validate_graph
from tools.exporter.quantization_storage import reject_legacy_safetensors_metadata


PACKAGE_FORMAT = "volvoxai-tiny-receipt-vqa-onnx-suite-package-v1"
GRAPH_FORMAT = "volvox-graph/v1"
GRAPH_FILENAME = "graph.json"
WEIGHTS_FILENAME = "model.safetensors"
ONNX_FRONTEND = "target-aware-onnx/v1"
PACKAGE_CLASSES = frozenset(("fp32", "w8a32", "w8a8-v1", "hybrid"))
FAMILY_ORDER = (
    "phone",
    "address",
    "store",
    "item_row",
    "item_math",
    "item_lookup",
    "math",
    "other",
)
ROUTER_MASK_MODES = (
    "keep-mask-i32",
    "normalized-weights-f32",
)
EXECUTION_DTYPES = {"float32", "int32", "int8", "uint8"}

_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9._:+-]*$")
_HASH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AssemblyError(ValueError):
    """An input package or explicit suite contract is invalid."""


class _DuplicateJsonKeyError(ValueError):
    """Fail-closed graph documents may not contain duplicate object keys."""


@dataclass(frozen=True)
class ExportPackage:
    directory: Path
    graph_path: Path
    weights_path: Path
    graph_sha256: str
    weights_sha256: str
    graph: dict[str, Any]
    weights: Mapping[str, Any]
    package_class: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class AssemblyResult:
    output_dir: Path
    manifest_path: Path
    router_graph_path: Path
    family_graph_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _TensorMetadata:
    """SafeTensors header information sufficient for descriptor validation."""

    shape: tuple[int, ...]
    dtype: str
    centered_abs_sums: tuple[int, ...] | None = None
    flat_values: tuple[int, ...] | None = None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as error:
        raise AssemblyError(f"could not parse {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AssemblyError(f"{label} {path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AssemblyError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _safetensors_dtype(value: str, label: str) -> str:
    dtype = {
        "F16": "float16",
        "F32": "float32",
        "I32": "int32",
        "I8": "int8",
        "U8": "uint8",
    }.get(value)
    if dtype is None:
        raise AssemblyError(f"{label} uses unsupported SafeTensors dtype {value!r}")
    return dtype


def _weight_scale_names(graph: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return result
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("opType") not in {
            "MatMul",
            "Linear",
            "Gemm",
        }:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for port in ("weight_scale", "scale"):
            name = inputs.get(port)
            if isinstance(name, str) and name:
                result.add(name)
    return result


def _central_quantization_table(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    root = graph.get("quantization")
    table = root.get("tensors") if isinstance(root, Mapping) else None
    return table if isinstance(table, Mapping) else {}


def _quantization_parameter_names(graph: Mapping[str, Any]) -> set[str]:
    return {
        name
        for descriptor in _central_quantization_table(graph).values()
        if isinstance(descriptor, Mapping)
        for name in (
            descriptor.get("scale_tensor"),
            descriptor.get("zero_point_tensor"),
        )
        if isinstance(name, str)
    }


def _read_safetensors_metadata(
    path: Path,
    graph: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    """Read headers plus the small referenced quantization parameters."""

    value_names = _weight_scale_names(graph) | _quantization_parameter_names(graph)
    result: dict[str, Any] = {}
    try:
        with safe_open(str(path), framework="numpy", device="cpu") as source:
            reject_legacy_safetensors_metadata(source.metadata() or {})
            names = list(source.keys())
            for name in names:
                tensor = source.get_slice(name)
                result[name] = _TensorMetadata(
                    shape=tuple(int(value) for value in tensor.get_shape()),
                    dtype=_safetensors_dtype(
                        str(tensor.get_dtype()), f"{label} tensor {name!r}"
                    ),
                )
            for name in sorted(value_names.intersection(names)):
                # Affine parameters and W8A32 scale operands are descriptor
                # data, not opaque model content. Loading only these vectors
                # avoids materializing a large model merely to assemble it.
                result[name] = np.asarray(source.get_tensor(name))

            # QLinear needs an exact I32 overflow proof, but retaining every
            # model weight would multiply peak memory across nine packages.
            # Materialize one tensor at a time and retain only per-output
            # centered absolute sums plus the small I32 bias vector.
            weight_quantization = _central_quantization_table(graph)
            for node in graph.get("nodes", []):
                if not isinstance(node, Mapping) or node.get("opType") != "QLinear":
                    continue
                inputs = node.get("inputs")
                if not isinstance(inputs, Mapping):
                    continue
                weight_name = inputs.get("weight")
                bias_name = inputs.get("bias")
                descriptor = (
                    weight_quantization.get(weight_name)
                    if isinstance(weight_quantization, Mapping) else None
                )
                zero_name = (
                    descriptor.get("zero_point_tensor")
                    if isinstance(descriptor, Mapping) else None
                )
                zero_points = result.get(zero_name) if isinstance(zero_name, str) else None
                if (
                    not isinstance(weight_name, str) or weight_name not in names
                    or not isinstance(bias_name, str) or bias_name not in names
                    or not isinstance(zero_points, np.ndarray)
                ):
                    continue
                weight = np.asarray(source.get_tensor(weight_name), dtype=np.int64)
                bias = np.asarray(source.get_tensor(bias_name), dtype=np.int64).reshape(-1)
                zero_points = np.asarray(zero_points, dtype=np.int64).reshape(-1)
                if weight.ndim != 2 or zero_points.size != weight.shape[0]:
                    continue
                centered = weight - zero_points.reshape(-1, 1)
                sums = tuple(int(value) for value in np.sum(np.abs(centered), axis=1, dtype=np.int64))
                weight_meta = result[weight_name]
                bias_meta = result[bias_name]
                result[weight_name] = _TensorMetadata(
                    shape=weight_meta.shape,
                    dtype=weight_meta.dtype,
                    centered_abs_sums=sums,
                )
                result[bias_name] = _TensorMetadata(
                    shape=bias_meta.shape,
                    dtype=bias_meta.dtype,
                    flat_values=tuple(int(value) for value in bias),
                )
    except AssemblyError:
        raise
    except Exception as error:
        raise AssemblyError(f"could not parse {label} {path}: {error}") from error
    return result


def _source_package_class(graph: Mapping[str, Any], label: str) -> str:
    source = graph.get("source")
    if not isinstance(source, Mapping):
        raise AssemblyError(f"{label}.source must identify an ONNX exporter result")
    if source.get("frontend") != ONNX_FRONTEND:
        raise AssemblyError(f"{label}.source.frontend must equal {ONNX_FRONTEND!r}")
    if not isinstance(source.get("onnx"), str) or not source.get("onnx"):
        raise AssemblyError(f"{label}.source.onnx must be a nonempty source filename")
    package_class = source.get("package_class")
    if package_class not in PACKAGE_CLASSES:
        raise AssemblyError(
            f"{label}.source.package_class must be one of "
            f"{', '.join(sorted(PACKAGE_CLASSES))}"
        )
    return str(package_class)


def _shape(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
        for dimension in value
    ):
        raise AssemblyError(f"{label} must be a concrete shape of positive integers")
    return list(value)


def _dtype(value: Any, label: str) -> str:
    if value not in EXECUTION_DTYPES:
        raise AssemblyError(
            f"{label} must be one of {', '.join(sorted(EXECUTION_DTYPES))}"
        )
    return str(value)


def _input_schema(graph: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    inputs = graph.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise AssemblyError(f"{label}.inputs must be a nonempty object")
    normalized: dict[str, dict[str, Any]] = {}
    for name in sorted(inputs):
        descriptor = inputs[name]
        if not isinstance(name, str) or not name:
            raise AssemblyError(f"{label}.inputs contains an invalid tensor name")
        if not isinstance(descriptor, dict):
            raise AssemblyError(f"{label}.inputs.{name} must be an object")
        _shape(descriptor.get("shape"), f"{label}.inputs.{name}.shape")
        _dtype(descriptor.get("dtype"), f"{label}.inputs.{name}.dtype")
        if "quantization" in descriptor:
            raise AssemblyError(f"{label}.inputs.{name} uses forbidden inline quantization")
        # Preserve the complete public descriptor.  Metadata such as image
        # normalization is part of an application-visible family schema.
        normalized[name] = dict(descriptor)
        quantization = _central_quantization_table(graph).get(name)
        if isinstance(quantization, Mapping):
            normalized[name]["quantization"] = dict(quantization)
    return normalized


def _output_schema(graph: dict[str, Any], label: str) -> list[dict[str, Any]]:
    output_names = graph.get("outputs")
    if (
        not isinstance(output_names, list)
        or not output_names
        or any(not isinstance(name, str) or not name for name in output_names)
        or len(set(output_names)) != len(output_names)
    ):
        raise AssemblyError(f"{label}.outputs must be a nonempty unique string array")

    producers: dict[str, dict[str, Any]] = {}
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise AssemblyError(f"{label}.nodes must be an array")
    for node_index, node in enumerate(nodes):
        node_label = f"{label}.nodes[{node_index}]"
        if not isinstance(node, dict):
            raise AssemblyError(f"{node_label} must be an object")
        outputs = node.get("outputs")
        output_shapes = node.get("outputs_shape")
        output_dtypes = node.get("outputs_dtype")
        if not isinstance(outputs, dict) or not outputs:
            raise AssemblyError(f"{node_label}.outputs must be a nonempty object")
        if not isinstance(output_shapes, dict):
            raise AssemblyError(f"{node_label}.outputs_shape must be an object")
        if not isinstance(output_dtypes, dict):
            raise AssemblyError(f"{node_label}.outputs_dtype must be an object")
        if set(output_shapes) != set(outputs):
            raise AssemblyError(
                f"{node_label}.outputs_shape must exactly describe every output port"
            )
        if set(output_dtypes) != set(outputs):
            raise AssemblyError(
                f"{node_label}.outputs_dtype must exactly describe every output port"
            )
        if "outputs_quantization" in node:
            raise AssemblyError(f"{node_label} uses forbidden inline quantization")
        for port, tensor_name in outputs.items():
            if not isinstance(port, str) or not port or not isinstance(tensor_name, str) or not tensor_name:
                raise AssemblyError(f"{node_label}.outputs contains an invalid port or tensor name")
            if tensor_name in producers:
                raise AssemblyError(f"{label} tensor {tensor_name!r} has multiple producers")
            descriptor: dict[str, Any] = {
                "name": tensor_name,
                "shape": _shape(
                    output_shapes.get(port), f"{node_label}.outputs_shape.{port}"
                ),
                "dtype": _dtype(
                    output_dtypes[port],
                    f"{node_label}.outputs_dtype.{port}",
                ),
            }
            quantization = _central_quantization_table(graph).get(tensor_name)
            if isinstance(quantization, Mapping):
                descriptor["quantization"] = dict(quantization)
            producers[tensor_name] = descriptor

    result: list[dict[str, Any]] = []
    for name in output_names:
        descriptor = producers.get(name)
        if descriptor is None:
            raise AssemblyError(f"{label} declared output {name!r} has no node producer")
        result.append(descriptor)
    return result


def _graph_schema(graph: dict[str, Any], label: str) -> dict[str, Any]:
    if graph.get("format") != GRAPH_FORMAT:
        raise AssemblyError(f"{label}.format must equal {GRAPH_FORMAT!r}")
    return {
        "inputs": _input_schema(graph, label),
        "outputs": _output_schema(graph, label),
    }


def _load_export_package(directory: str | Path, label: str) -> ExportPackage:
    root = Path(directory)
    if not root.is_dir():
        raise AssemblyError(f"{label} export directory does not exist: {root}")
    graph_path = root / GRAPH_FILENAME
    weights_path = root / WEIGHTS_FILENAME
    if not graph_path.is_file():
        raise AssemblyError(f"{label} export is missing {GRAPH_FILENAME}: {graph_path}")
    if not weights_path.is_file():
        raise AssemblyError(f"{label} export is missing {WEIGHTS_FILENAME}: {weights_path}")
    try:
        if weights_path.stat().st_size <= 0:
            raise AssemblyError(f"{label} {WEIGHTS_FILENAME} must not be empty")
    except OSError as error:
        raise AssemblyError(f"could not inspect {weights_path}: {error}") from error
    graph_sha256 = _sha256(graph_path)
    weights_sha256 = _sha256(weights_path)
    graph = _read_json(graph_path, f"{label} graph")
    package_class = _source_package_class(graph, f"{label} graph")
    weights = _read_safetensors_metadata(weights_path, graph, f"{label} weights")
    if _sha256(graph_path) != graph_sha256 or _sha256(weights_path) != weights_sha256:
        raise AssemblyError(f"{label} export changed while it was being validated")
    return ExportPackage(
        directory=root,
        graph_path=graph_path,
        weights_path=weights_path,
        graph_sha256=graph_sha256,
        weights_sha256=weights_sha256,
        graph=graph,
        weights=weights,
        package_class=package_class,
        schema=_graph_schema(graph, f"{label} graph"),
    )


def _label(value: str, description: str) -> str:
    if not isinstance(value, str) or not _LABEL_RE.fullmatch(value):
        raise AssemblyError(
            f"{description} must match {_LABEL_RE.pattern!r}; got {value!r}"
        )
    return value


def _profiles(values: Sequence[str], description: str) -> tuple[str, ...]:
    if not values:
        raise AssemblyError(f"{description} requires at least one explicit profile")
    result: list[str] = []
    for value in values:
        normalized = _label(value, description)
        if normalized not in TARGETS:
            raise AssemblyError(
                f"{description} must be one of {', '.join(TARGETS)}; got {normalized!r}"
            )
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _precision(value: str, description: str) -> str:
    normalized = _label(value, description)
    if normalized not in PACKAGE_CLASSES:
        raise AssemblyError(
            f"{description} must be one of {', '.join(sorted(PACKAGE_CLASSES))}; "
            f"got {normalized!r}"
        )
    return normalized


def _producer_hashes(values: Mapping[str, str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, digest in (values or {}).items():
        if not isinstance(name, str) or not _HASH_NAME_RE.fullmatch(name):
            raise AssemblyError(f"producer hash name is invalid: {name!r}")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise AssemblyError(f"producer hash {name!r} must be a lowercase SHA-256 digest")
        result[name] = digest
    return dict(sorted(result.items()))


def _schema_input(schema: dict[str, Any], name: str, label: str) -> dict[str, Any]:
    descriptor = schema["inputs"].get(name)
    if descriptor is None:
        raise AssemblyError(f"{label} {name!r} is not a graph input")
    return descriptor


def _schema_output(schema: dict[str, Any], name: str, label: str) -> dict[str, Any]:
    for descriptor in schema["outputs"]:
        if descriptor["name"] == name:
            return descriptor
    raise AssemblyError(f"{label} {name!r} is not a declared graph output")


def _tensor_descriptor(
    package: ExportPackage,
    name: str,
    label: str,
) -> dict[str, Any]:
    descriptor = package.schema["inputs"].get(name)
    if descriptor is not None:
        return descriptor
    for node_index, node in enumerate(package.graph.get("nodes", [])):
        if not isinstance(node, Mapping):
            continue
        outputs = node.get("outputs")
        if not isinstance(outputs, Mapping):
            continue
        for port, tensor_name in outputs.items():
            if tensor_name != name:
                continue
            shapes = node.get("outputs_shape")
            dtypes = node.get("outputs_dtype")
            if not isinstance(shapes, Mapping):
                raise AssemblyError(f"{label} producer has no outputs_shape object")
            result: dict[str, Any] = {
                "name": name,
                "shape": _shape(
                    shapes.get(port),
                    f"{label} producer nodes[{node_index}].outputs_shape.{port}",
                ),
                "dtype": _dtype(
                    dtypes.get(port, "float32") if isinstance(dtypes, Mapping) else "float32",
                    f"{label} producer nodes[{node_index}].outputs_dtype.{port}",
                ),
            }
            quantization = _central_quantization_table(package.graph).get(name)
            if isinstance(quantization, Mapping):
                result["quantization"] = dict(quantization)
            return result
    raise AssemblyError(f"{label} tensor {name!r} has no graph input or node producer")


def _producer(
    graph: Mapping[str, Any],
    tensor_name: str,
    label: str,
) -> tuple[int, Mapping[str, Any], str]:
    matches: list[tuple[int, Mapping[str, Any], str]] = []
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise AssemblyError(f"{label} graph nodes must be an array")
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            continue
        outputs = node.get("outputs")
        if not isinstance(outputs, Mapping):
            continue
        for port, name in outputs.items():
            if name == tensor_name:
                matches.append((index, node, str(port)))
    if len(matches) != 1:
        raise AssemblyError(f"{label} must have exactly one node producer")
    return matches[0]


def _depends_on_graph_input(
    graph: Mapping[str, Any],
    tensor_name: str,
    input_name: str,
) -> bool:
    if tensor_name == input_name:
        return True
    producers: dict[str, Mapping[str, Any]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, Mapping) or not isinstance(node.get("outputs"), Mapping):
            continue
        for output_name in node["outputs"].values():
            if isinstance(output_name, str):
                producers[output_name] = node
    pending = [tensor_name]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == input_name:
            return True
        if current in visited:
            continue
        visited.add(current)
        node = producers.get(current)
        inputs = node.get("inputs") if node is not None else None
        if isinstance(inputs, Mapping):
            pending.extend(
                value for value in inputs.values() if isinstance(value, str) and value
            )
    return False


def _validate_offline_profiles(
    package: ExportPackage,
    profiles: Sequence[str],
    label: str,
) -> None:
    result = validate_graph(package.graph, profiles, weights=package.weights)
    if result.supported:
        return
    details = []
    for diagnostic in result.diagnostics[:4]:
        target = f" ({diagnostic.target})" if diagnostic.target else ""
        details.append(f"{diagnostic.code}{target}: {diagnostic.message}")
    if len(result.diagnostics) > len(details):
        details.append(f"and {len(result.diagnostics) - len(details)} more")
    raise AssemblyError(
        f"{label} fails offline profile validation: {'; '.join(details)}"
    )


def _validate_router_interface(
    package: ExportPackage,
    *,
    q_ids_input: str,
    mask_mode: str,
    mask_input: str,
    family_id_output: str,
) -> None:
    if mask_mode not in ROUTER_MASK_MODES:
        raise AssemblyError(
            f"router mask mode must be one of {', '.join(ROUTER_MASK_MODES)}"
        )
    if q_ids_input == mask_input:
        raise AssemblyError("router q_ids and semantic mask inputs must be distinct")
    q_ids = _schema_input(package.schema, q_ids_input, "router q_ids input")
    mask = _schema_input(package.schema, mask_input, "router semantic mask input")
    route = _schema_output(package.schema, family_id_output, "router family ID output")
    q_shape = _shape(q_ids.get("shape"), "router q_ids input shape")
    mask_shape = _shape(mask.get("shape"), "router semantic mask input shape")
    if q_ids.get("dtype") != "int32" or q_ids.get("quantization") is not None:
        raise AssemblyError("router q_ids input must be unquantized int32")
    if len(q_shape) != 2 or q_shape[0] != 1:
        raise AssemblyError("router q_ids input must have fixed B=1 shape [1,S]")
    expected_mask_dtype = "int32" if mask_mode == "keep-mask-i32" else "float32"
    if mask.get("dtype") != expected_mask_dtype or mask.get("quantization") is not None:
        raise AssemblyError(
            f"router {mask_mode} input must be unquantized {expected_mask_dtype}"
        )
    if mask_shape != q_shape:
        raise AssemblyError("router semantic mask/weight input shape must equal q_ids shape")
    if (
        route.get("dtype") != "int32"
        or route.get("shape") != [1]
        or route.get("quantization") is not None
    ):
        raise AssemblyError("router family ID output must be unquantized int32 [1]")

    producer_index, producer, output_port = _producer(
        package.graph, family_id_output, "router family ID output"
    )
    op_type = producer.get("opType")
    if op_type not in {"ArgMax", "QArgMax"} or output_port != "out":
        raise AssemblyError(
            "router family ID output must be produced directly by ArgMax or QArgMax"
        )
    inputs = producer.get("inputs")
    params = producer.get("params")
    if not isinstance(inputs, Mapping) or set(inputs) != {"input"}:
        raise AssemblyError("router terminal ArgMax must have exactly one input port")
    if not isinstance(params, Mapping) or not isinstance(params.get("axis"), int):
        raise AssemblyError("router terminal ArgMax must declare an integer axis")
    logits_name = inputs.get("input")
    if not isinstance(logits_name, str) or not logits_name:
        raise AssemblyError("router terminal ArgMax input must name the logits tensor")
    logits = _tensor_descriptor(package, logits_name, "router logits")
    logits_shape = _shape(logits.get("shape"), "router logits shape")
    axis = int(params["axis"])
    if axis < 0:
        axis += len(logits_shape)
    if logits_shape != [1, len(FAMILY_ORDER)] or axis != len(logits_shape) - 1:
        raise AssemblyError(
            f"router ArgMax input must be [1,{len(FAMILY_ORDER)}] logits reduced "
            "over the final family axis"
        )
    if op_type == "ArgMax" and (
        params.get("select_last_index", 0) != 0
        or params.get("keepdims", True) not in {0, False}
    ):
        raise AssemblyError(
            "router ArgMax must use first-index ties and remove the family axis"
        )

    nodes = package.graph.get("nodes", [])
    for node_index, node in enumerate(nodes):
        if node_index == producer_index or not isinstance(node, Mapping):
            continue
        node_inputs = node.get("inputs")
        if isinstance(node_inputs, Mapping) and family_id_output in node_inputs.values():
            raise AssemblyError("router family ID must terminate the graph for host dispatch")
    if not _depends_on_graph_input(package.graph, logits_name, q_ids_input):
        raise AssemblyError("router logits must depend on the declared q_ids input")
    if not _depends_on_graph_input(package.graph, logits_name, mask_input):
        raise AssemblyError("router logits must depend on the declared semantic mask input")

    source = package.graph.get("source")
    features = source.get("features") if isinstance(source, Mapping) else None
    routers = features.get("router") if isinstance(features, Mapping) else None
    all_masked_contracts = (
        {"zero-vector-via-normalized-zero-weights"}
        if mask_mode == "normalized-weights-f32"
        else {"zero-vector-via-clamp-min-one", "zero-vector-via-qmaskedmean"}
    )
    attested = any(
        isinstance(feature, Mapping)
        and feature.get("output") == family_id_output
        and feature.get("tie_policy") == "first-index"
        and feature.get("all_masked") in all_masked_contracts
        and mask_input in feature.get("semantic_mask_inputs", [])
        for feature in (routers if isinstance(routers, list) else [])
    )
    if not attested:
        raise AssemblyError(
            "router export lacks an exact masked-mean/first-tie/all-masked feature attestation"
        )

    declared_outputs = package.graph.get("outputs", [])
    additional_outputs = [name for name in declared_outputs if name != family_id_output]
    if len(additional_outputs) > 1 or (
        additional_outputs and additional_outputs[0] != logits_name
    ):
        raise AssemblyError(
            "router may declare only family_id and its optional [1,F] logits output"
        )


def _copy_artifact(source: Path, destination: Path, expected_sha256: str) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as error:
        raise AssemblyError(f"could not copy {source} to {destination}: {error}") from error
    if _sha256(destination) != expected_sha256:
        raise AssemblyError(f"copied artifact hash mismatch for {destination}")


def _artifact_record(prefix: str, package: ExportPackage) -> dict[str, Any]:
    return {
        "graph": {
            "file": f"{prefix}/{GRAPH_FILENAME}",
            "sha256": package.graph_sha256,
        },
        "weights": {
            "file": f"{prefix}/{WEIGHTS_FILENAME}",
            "sha256": package.weights_sha256,
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise AssemblyError(f"could not write {path}: {error}") from error


def assemble_onnx_suite(
    router_dir: str | Path,
    family_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    router_mask_mode: str,
    router_mask_input: str,
    router_precision: str,
    family_precision: str,
    router_profiles: Sequence[str],
    family_profiles: Sequence[str],
    router_q_ids_input: str = "q_ids",
    router_family_id_output: str = "family_id",
    producer_hashes: Mapping[str, str] | None = None,
) -> AssemblyResult:
    """Validate and copy nine pre-exported graph packages into one suite."""

    actual_families = set(family_dirs)
    expected_families = set(FAMILY_ORDER)
    if actual_families != expected_families or len(family_dirs) != len(FAMILY_ORDER):
        missing = sorted(expected_families - actual_families)
        unexpected = sorted(actual_families - expected_families)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise AssemblyError(
            "family mappings must contain exactly the eight explicit families"
            + (f": {', '.join(details)}" if details else "")
        )

    router_precision = _precision(router_precision, "router precision")
    family_precision = _precision(family_precision, "family precision")
    router_profiles = _profiles(router_profiles, "router profile")
    family_profiles = _profiles(family_profiles, "family profile")
    router_targets = expand_targets(router_profiles)
    family_targets = expand_targets(family_profiles)
    common_targets = tuple(
        target for target in router_targets if target in set(family_targets)
    )
    if not common_targets:
        raise AssemblyError(
            "router and explicit-family profiles must resolve to a nonempty target intersection"
        )
    producer_hash_record = _producer_hashes(producer_hashes)

    router = _load_export_package(router_dir, "router")
    families = {
        family: _load_export_package(family_dirs[family], f"family {family}")
        for family in FAMILY_ORDER
    }
    if router.package_class != router_precision:
        raise AssemblyError(
            f"router precision {router_precision!r} does not match exported package "
            f"class {router.package_class!r}"
        )
    for family in FAMILY_ORDER:
        if families[family].package_class != family_precision:
            raise AssemblyError(
                f"family {family} precision {family_precision!r} does not match "
                f"exported package class {families[family].package_class!r}"
            )
    _validate_router_interface(
        router,
        q_ids_input=router_q_ids_input,
        mask_mode=router_mask_mode,
        mask_input=router_mask_input,
        family_id_output=router_family_id_output,
    )

    reference_family = FAMILY_ORDER[0]
    reference_schema = families[reference_family].schema
    for family in FAMILY_ORDER[1:]:
        if families[family].schema != reference_schema:
            raise AssemblyError(
                f"family {family} public input/output schema differs from {reference_family}"
            )
    _validate_offline_profiles(router, router_profiles, "router")
    for family in FAMILY_ORDER:
        _validate_offline_profiles(
            families[family], family_profiles, f"family {family}"
        )

    destination = Path(output_dir)
    if destination.exists():
        raise AssemblyError(f"output directory must not already exist: {destination}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-", dir=str(destination.parent)
            )
        )
    except OSError as error:
        raise AssemblyError(f"could not create output staging directory: {error}") from error

    try:
        _copy_artifact(
            router.graph_path, staging / "router" / GRAPH_FILENAME, router.graph_sha256
        )
        _copy_artifact(
            router.weights_path,
            staging / "router" / WEIGHTS_FILENAME,
            router.weights_sha256,
        )
        for family in FAMILY_ORDER:
            package = families[family]
            prefix = Path("families") / family
            _copy_artifact(
                package.graph_path, staging / prefix / GRAPH_FILENAME, package.graph_sha256
            )
            _copy_artifact(
                package.weights_path,
                staging / prefix / WEIGHTS_FILENAME,
                package.weights_sha256,
            )

        manifest: dict[str, Any] = {
            "format": PACKAGE_FORMAT,
            "source_kind": "independent-onnx-exported-static-graphs",
            "family_order": list(FAMILY_ORDER),
            "target_qualification": {
                "offline_common_targets": list(common_targets),
                "qualified_suite_profiles": [],
                "strict_no_fallback_execution": "not-run",
                "reason": (
                    "the assembler performs descriptor validation but receives no "
                    "strict runtime execution attestation"
                ),
            },
            "routing": {
                "batch_size": 1,
                "selection_scope": "whole-generation",
                "id_to_family": [
                    {"id": index, "family": family}
                    for index, family in enumerate(FAMILY_ORDER)
                ],
            },
            "router": {
                **_artifact_record("router", router),
                "precision": router_precision,
                "requested_profiles": list(router_profiles),
                "offline_validated_targets": list(router_targets),
                "semantic_inputs": {
                    "q_ids": router_q_ids_input,
                    "mask_or_weights": router_mask_input,
                },
                "semantic_outputs": {"family_id": router_family_id_output},
                "semantic_mask": {
                    "mode": router_mask_mode,
                    "input": router_mask_input,
                    "semantics": (
                        "I32 nonzero exactly for kept question tokens"
                        if router_mask_mode == "keep-mask-i32"
                        else "F32 normalized token weights with padded tokens equal to zero"
                    ),
                },
                "interface": router.schema,
            },
            "explicit_family_contract": {
                "precision": family_precision,
                "requested_profiles": list(family_profiles),
                "offline_validated_targets": list(family_targets),
                "interface": reference_schema,
            },
            "explicit_families": {
                family: {
                    **_artifact_record(f"families/{family}", families[family]),
                    "family_id": index,
                }
                for index, family in enumerate(FAMILY_ORDER)
            },
        }
        if producer_hash_record:
            manifest["producer_hashes"] = producer_hash_record
        manifest_path = staging / "package_manifest.json"
        _write_json(manifest_path, manifest)
        try:
            os.replace(staging, destination)
        except OSError as error:
            raise AssemblyError(f"could not publish assembled suite {destination}: {error}") from error
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return AssemblyResult(
        output_dir=destination,
        manifest_path=destination / "package_manifest.json",
        router_graph_path=destination / "router" / GRAPH_FILENAME,
        family_graph_paths=tuple(
            destination / "families" / family / GRAPH_FILENAME
            for family in FAMILY_ORDER
        ),
    )


def _assignment(value: str, label: str) -> tuple[str, str]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise AssemblyError(f"{label} must use NAME=VALUE syntax")
    return name, raw_path


def _family_assignments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        family, raw_path = _assignment(value, "--family")
        if family in result:
            raise AssemblyError(f"duplicate family mapping for {family!r}")
        result[family] = Path(raw_path)
    return result


def _hash_assignments(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, digest = _assignment(value, "--producer-hash")
        if name in result:
            raise AssemblyError(f"duplicate producer hash {name!r}")
        result[name] = digest
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", required=True, type=Path, help="router export directory")
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        metavar="FAMILY=DIR",
        help="explicit family export directory; repeat exactly once for each family",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--router-mask-mode", required=True, choices=ROUTER_MASK_MODES)
    parser.add_argument("--router-mask-input", required=True)
    parser.add_argument("--router-q-ids-input", default="q_ids")
    parser.add_argument("--router-family-id-output", default="family_id")
    parser.add_argument("--router-precision", required=True)
    parser.add_argument("--family-precision", required=True)
    parser.add_argument("--router-profile", action="append", required=True)
    parser.add_argument("--family-profile", action="append", required=True)
    parser.add_argument(
        "--producer-hash",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="optional producer attestation hash; repeat with unique names",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = assemble_onnx_suite(
            args.router,
            _family_assignments(args.family),
            args.out_dir,
            router_mask_mode=args.router_mask_mode,
            router_mask_input=args.router_mask_input,
            router_q_ids_input=args.router_q_ids_input,
            router_family_id_output=args.router_family_id_output,
            router_precision=args.router_precision,
            family_precision=args.family_precision,
            router_profiles=args.router_profile,
            family_profiles=args.family_profile,
            producer_hashes=_hash_assignments(args.producer_hash),
        )
    except AssemblyError as error:
        parser.error(str(error))
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
