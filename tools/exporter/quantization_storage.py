"""Canonical safetensors-backed affine quantization metadata.

The graph document owns topology, never numeric quantization data.  A
quantized graph tensor is associated with immutable scale/zero-point tensors
through a small reference-only table::

    "quantization": {
      "format": "volvox-affine-safetensors/v1",
      "tensors": {
        "hidden": {
          "scheme": "per_tensor",
          "scale_tensor": "__quant__.abc.scale",
          "zero_point_tensor": "__quant__.abc.zero_point"
        }
      }
    }

All referenced tensors are stored in the package's safetensors shard.  This
keeps graph JSON deterministic and compact, makes imported ONNX/TFLite Q/DQ
parameters reusable, and prevents optimizers from accidentally treating a
rounded JSON number as a different quantization contract.

The graph format remains ``volvox-graph/v1``.  Numeric affine descriptors are
an exporter-internal authoring input only; package readers accept the central
reference table above and reject every former inline or companion layout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, MutableMapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .ir import find_retired_affine_param_path


GRAPH_FORMAT = "volvox-graph/v1"
QUANTIZATION_FORMAT = "volvox-affine-safetensors/v1"
RUNTIME_BYTE_DTYPES = frozenset({"int8", "uint8"})
LEGACY_SAFETENSORS_METADATA = frozenset({
    "weights_quantization",
    "weights_quantization_storage",
})


@dataclass(frozen=True)
class ExternalizationReport:
    tensors: int
    scales_created: int
    zero_points_created: int
    parameters_reused: int


@dataclass(frozen=True)
class ResolvedQuantization:
    scheme: str
    axis: int | None
    scales: np.ndarray
    zero_points: np.ndarray


def _fail(code: str, message: str, *, tensor: str | None = None) -> None:
    raise ExporterError(Diagnostic(
        code=code,
        message=message,
        stage="quantization-storage",
        source_node=tensor,
        constraint="numeric affine parameters live only in safetensors",
    ))


def reject_legacy_safetensors_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject the retired companion-manifest representation at I/O boundaries."""

    forbidden = sorted(LEGACY_SAFETENSORS_METADATA.intersection(metadata))
    if forbidden:
        _fail(
            "VXPKGWEIGHT002",
            "legacy safetensors quantization metadata is forbidden: "
            + ", ".join(forbidden),
        )


def _dtype_name(value: Any) -> str | None:
    try:
        dtype = np.asarray(value).dtype
    except Exception:
        return None
    if dtype == np.dtype(np.int8):
        return "int8"
    if dtype == np.dtype(np.uint8):
        return "uint8"
    if dtype == np.dtype(np.int32):
        return "int32"
    if dtype == np.dtype(np.float32):
        return "float32"
    if dtype == np.dtype(np.float16):
        return "float16"
    return str(dtype)


def _declared_dtype_name(value: Any) -> str | None:
    """Normalize metadata-only and concrete tensor dtype spellings."""

    declared = getattr(value, "dtype", None)
    if declared is not None:
        name = str(declared)
        if name.startswith("torch."):
            name = name.removeprefix("torch.")
        if name in {"int8", "uint8", "int32", "float16", "float32"}:
            return name
    return _dtype_name(value)


def _declared_tensors(
    graph: Mapping[str, Any], tensors: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    dtypes: dict[str, str] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for name, value in tensors.items():
        declared_shape = getattr(value, "shape", None)
        declared_dtype = _declared_dtype_name(value)
        if declared_dtype is not None and declared_shape is not None:
            dtypes[str(name)] = declared_dtype
            shapes[str(name)] = tuple(int(v) for v in declared_shape)
        else:
            array = np.asarray(value)
            dtype = _dtype_name(array)
            if dtype is not None:
                dtypes[str(name)] = dtype
            shapes[str(name)] = tuple(int(v) for v in array.shape)
    inputs = graph.get("inputs")
    if isinstance(inputs, Mapping):
        for name, descriptor in inputs.items():
            if not isinstance(name, str) or not isinstance(descriptor, Mapping):
                continue
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            if isinstance(dtype, str):
                dtypes[name] = dtype
            if isinstance(shape, list) and all(isinstance(v, int) for v in shape):
                shapes[name] = tuple(shape)
    nodes = graph.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            outputs = node.get("outputs")
            output_dtypes = node.get("outputs_dtype")
            output_shapes = node.get("outputs_shape")
            if not isinstance(outputs, Mapping):
                continue
            for port, name in outputs.items():
                if not isinstance(name, str):
                    continue
                if isinstance(output_dtypes, Mapping) and isinstance(
                    output_dtypes.get(port), str
                ):
                    dtypes[name] = str(output_dtypes[port])
                else:
                    dtypes.setdefault(name, "float32")
                if isinstance(output_shapes, Mapping):
                    shape = output_shapes.get(port)
                    if isinstance(shape, list) and all(
                        isinstance(v, int) for v in shape
                    ):
                        shapes[name] = tuple(shape)
    return dtypes, shapes


def _candidate_parameters(graph: Mapping[str, Any]) -> dict[str, tuple[str | None, str | None]]:
    """Find already-materialized Q/DQ or weight parameter operands.

    Existing ONNX/TFLite quantized models already store these arrays in
    safetensors.  Reusing them preserves exact source bits and avoids duplicate
    constants.  A candidate is accepted only after numeric validation.
    """

    candidates: dict[str, tuple[str | None, str | None]] = {}
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return candidates
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        outputs = node.get("outputs")
        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            continue
        op = node.get("opType")
        if op in {"QuantizeLinear", "RequantizeLinear"}:
            scale = inputs.get("scale") or inputs.get("output_scale")
            zero = inputs.get("zero_point") or inputs.get("output_zero_point")
            for target in outputs.values():
                if isinstance(target, str):
                    candidates[target] = (
                        scale if isinstance(scale, str) else None,
                        zero if isinstance(zero, str) else None,
                    )
        if op == "DequantizeLinear":
            target = inputs.get("input")
            if isinstance(target, str):
                scale = inputs.get("scale")
                zero = inputs.get("zero_point")
                candidates.setdefault(target, (
                    scale if isinstance(scale, str) else None,
                    zero if isinstance(zero, str) else None,
                ))
        weight = inputs.get("weight")
        if isinstance(weight, str):
            scale = inputs.get("weight_scale")
            zero = inputs.get("weight_zero_point")
            if isinstance(scale, str):
                candidates.setdefault(weight, (
                    scale,
                    zero if isinstance(zero, str) else None,
                ))
    return candidates


def _descriptor_arrays(
    name: str,
    descriptor: Mapping[str, Any],
    *,
    dtype: str,
    shape: tuple[int, ...] | None,
) -> tuple[str, int | None, np.ndarray, np.ndarray]:
    scheme = descriptor.get("scheme")
    if scheme is None:
        scheme = "per_axis" if "scales" in descriptor else "per_tensor"
    if scheme == "per_tensor":
        if "scales" in descriptor or "zero_points" in descriptor or "axis" in descriptor:
            _fail("VXQSTORE004", f"tensor {name!r} mixes per-tensor and per-axis fields", tensor=name)
        scale = descriptor.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            _fail("VXQSTORE005", f"tensor {name!r} requires one numeric scale", tensor=name)
        zero = descriptor.get("zero_point", 0)
        if isinstance(zero, bool) or not isinstance(zero, int):
            _fail("VXQSTORE006", f"tensor {name!r} requires one integer zero point", tensor=name)
        scales = np.asarray([scale], dtype=np.float32)
        zero_points = np.asarray([zero], dtype=np.int64)
        axis = None
    elif scheme == "per_axis":
        axis = descriptor.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            _fail("VXQSTORE007", f"tensor {name!r} requires an integer quantization axis", tensor=name)
        if shape is None or not shape:
            _fail("VXQSTORE008", f"tensor {name!r} needs a known rank for per-axis quantization", tensor=name)
        normalized = axis + len(shape) if axis < 0 else axis
        if normalized < 0 or normalized >= len(shape):
            _fail("VXQSTORE009", f"tensor {name!r} quantization axis is outside its rank", tensor=name)
        axis = normalized
        raw_scales = descriptor.get("scales")
        if not isinstance(raw_scales, (list, tuple)):
            _fail("VXQSTORE010", f"tensor {name!r} requires a scale vector", tensor=name)
        raw_zeros = descriptor.get("zero_points", [0] * len(raw_scales))
        if not isinstance(raw_zeros, (list, tuple)):
            _fail("VXQSTORE011", f"tensor {name!r} requires a zero-point vector", tensor=name)
        if len(raw_scales) != shape[axis] or len(raw_zeros) != shape[axis]:
            _fail(
                "VXQSTORE012",
                f"tensor {name!r} quantization vector length must equal axis {axis} extent {shape[axis]}",
                tensor=name,
            )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_scales):
            _fail("VXQSTORE013", f"tensor {name!r} scale vector must be numeric", tensor=name)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_zeros):
            _fail("VXQSTORE014", f"tensor {name!r} zero-point vector must contain integers", tensor=name)
        scales = np.asarray(raw_scales, dtype=np.float32)
        zero_points = np.asarray(raw_zeros, dtype=np.int64)
    else:
        _fail("VXQSTORE015", f"tensor {name!r} has unsupported quantization scheme {scheme!r}", tensor=name)

    if scales.size == 0 or any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in scales):
        _fail("VXQSTORE016", f"tensor {name!r} scales must be finite positive F32 values", tensor=name)
    low, high = (-128, 127) if dtype == "int8" else (0, 255)
    if any(int(value) < low or int(value) > high for value in zero_points):
        _fail(
            "VXQSTORE017",
            f"tensor {name!r} zero points are outside {dtype} range [{low}, {high}]",
            tensor=name,
        )
    zero_dtype = np.int8 if dtype == "int8" else np.uint8
    return str(scheme), axis, scales, zero_points.astype(zero_dtype)


def _parameter_name(
    target: str,
    kind: str,
    tensors: Mapping[str, Any],
    claimed: set[str],
) -> str:
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
    for width in (16, 24, 32, 64):
        name = f"__quant__.{digest[:width]}.{kind}"
        if name not in tensors and name not in claimed:
            claimed.add(name)
            return name
    _fail("VXQSTORE018", f"could not allocate an unambiguous parameter name for {target!r}", tensor=target)
    raise AssertionError("unreachable")


def _same_values(value: Any, expected: np.ndarray) -> bool:
    try:
        actual = np.asarray(value)
    except Exception:
        return False
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and np.array_equal(actual, expected)
    )


def externalize_quantization(
    graph: MutableMapping[str, Any],
    tensors: MutableMapping[str, Any],
    descriptors: Mapping[str, Mapping[str, Any]],
) -> ExternalizationReport:
    """Materialize exporter-owned affine descriptors as safetensors tensors.

    ``descriptors`` is transient compiler state, not a graph representation.
    ``graph`` must already obey the public v1 shape and contain none of the old
    inline or companion fields.  The operation is planned and validated before
    either mutable object is changed, so failure cannot leave a half-authored
    package.
    """

    if graph.get("format") != GRAPH_FORMAT:
        _fail("VXQSTORE020", f"graph format must be {GRAPH_FORMAT!r}")
    _reject_legacy_quantization(graph)
    if "quantization" in graph:
        _fail(
            "VXQSTORE024",
            "quantization is already materialized; authoring cannot replace it",
        )
    if not isinstance(descriptors, Mapping):
        _fail("VXQSTORE001", "quantization descriptors must be an object")
    dtypes, shapes = _declared_tensors(graph, tensors)
    candidates = _candidate_parameters(graph)
    claimed: set[str] = set()
    additions: dict[str, np.ndarray] = {}
    table: dict[str, dict[str, Any]] = {}
    scales_created = 0
    zeros_created = 0
    reused = 0

    for name in sorted(descriptors):
        descriptor = descriptors[name]
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            _fail(
                "VXQSTORE001",
                "quantization descriptors must map tensor names to objects",
            )
        dtype = dtypes.get(name)
        if dtype not in RUNTIME_BYTE_DTYPES:
            _fail(
                "VXQSTORE019",
                f"quantized tensor {name!r} must have int8 or uint8 storage, got {dtype!r}",
                tensor=name,
            )
        scheme, axis, scales, zero_points = _descriptor_arrays(
            name, descriptor, dtype=dtype, shape=shapes.get(name)
        )
        candidate_scale, candidate_zero = candidates.get(name, (None, None))
        if candidate_scale is not None and candidate_scale in tensors:
            if not _same_values(tensors[candidate_scale], scales):
                _fail(
                    "VXQSTORE043",
                    f"existing scale tensor {candidate_scale!r} disagrees with {name!r} metadata",
                    tensor=name,
                )
            scale_name = candidate_scale
            reused += 1
        else:
            scale_name = _parameter_name(name, "scale", {**tensors, **additions}, claimed)
            additions[scale_name] = scales
            scales_created += 1
        if candidate_zero is not None and candidate_zero in tensors:
            if not _same_values(tensors[candidate_zero], zero_points):
                _fail(
                    "VXQSTORE044",
                    f"existing zero-point tensor {candidate_zero!r} disagrees with {name!r} metadata",
                    tensor=name,
                )
            zero_name = candidate_zero
            reused += 1
        else:
            zero_name = _parameter_name(name, "zero_point", {**tensors, **additions}, claimed)
            additions[zero_name] = zero_points
            zeros_created += 1
        entry: dict[str, Any] = {
            "scheme": scheme,
            "scale_tensor": scale_name,
            "zero_point_tensor": zero_name,
        }
        if axis is not None:
            entry["axis"] = axis
        table[name] = entry

    # Build the resulting document separately to keep authoring transactional.
    new_nodes: list[Any] | None = None
    if isinstance(graph.get("nodes"), list):
        new_nodes = []
        for node in graph["nodes"]:
            value = copy.deepcopy(dict(node)) if isinstance(node, Mapping) else node
            if isinstance(value, dict):
                inputs = value.get("inputs")
                outputs = value.get("outputs")
                if isinstance(inputs, dict) and isinstance(outputs, Mapping):
                    if value.get("opType") == "QuantizeLinear":
                        for target in outputs.values():
                            descriptor = table.get(target) if isinstance(target, str) else None
                            if descriptor is not None:
                                inputs["scale"] = descriptor["scale_tensor"]
                                inputs["zero_point"] = descriptor["zero_point_tensor"]
                    elif value.get("opType") == "DequantizeLinear":
                        target = inputs.get("input")
                        descriptor = table.get(target) if isinstance(target, str) else None
                        if descriptor is not None:
                            inputs["scale"] = descriptor["scale_tensor"]
                            inputs["zero_point"] = descriptor["zero_point_tensor"]
            new_nodes.append(value)

    planned_graph = copy.deepcopy(dict(graph))
    planned_tensors = dict(tensors)
    planned_tensors.update(additions)
    if new_nodes is not None:
        planned_graph["nodes"] = new_nodes
    if table:
        planned_graph["quantization"] = {
            "format": QUANTIZATION_FORMAT,
            "tensors": table,
        }
    else:
        planned_graph.pop("quantization", None)
    validate_external_quantization(planned_graph, planned_tensors)
    tensors.update(additions)
    graph.clear()
    graph.update(planned_graph)
    return ExternalizationReport(len(table), scales_created, zeros_created, reused)


def _reject_legacy_quantization(graph: Mapping[str, Any]) -> None:
    if "weights_quantization" in graph or "weights_quantization_storage" in graph:
        _fail("VXQSTORE021", "legacy weight quantization fields are forbidden")
    inputs = graph.get("inputs")
    if isinstance(inputs, Mapping):
        for name, descriptor in inputs.items():
            if isinstance(descriptor, Mapping) and "quantization" in descriptor:
                _fail(
                    "VXQSTORE022",
                    f"input {name!r} contains inline quantization",
                    tensor=str(name),
                )
    nodes = graph.get("nodes")
    if isinstance(nodes, list):
        for index, node in enumerate(nodes):
            if isinstance(node, Mapping) and "outputs_quantization" in node:
                _fail("VXQSTORE023", "node contains inline output quantization")
            if isinstance(node, Mapping):
                retired_path = find_retired_affine_param_path(node.get("params"))
                if retired_path is not None:
                    label = node.get("id") or node.get("source_name") or index
                    _fail(
                        "VXQSTORE050",
                        f"node {label!r} contains retired affine payload at {retired_path}",
                        tensor=str(label),
                    )
    standalone_tensors = graph.get("standaloneTensors")
    if isinstance(standalone_tensors, Mapping):
        for name, descriptor in standalone_tensors.items():
            if isinstance(descriptor, Mapping) and "quantization" in descriptor:
                _fail(
                    "VXQSTORE024",
                    f"standalone tensor {name!r} contains inline quantization",
                    tensor=str(name),
                )


def prune_external_quantization(
    graph: MutableMapping[str, Any], tensors: Mapping[str, Any]
) -> int:
    """Drop central entries whose activation targets a trusted rewrite removed.

    Callers must validate the input graph before running optimizer passes.  The
    helper intentionally does not repair references or migrate old layouts; it
    only removes targets that no longer exist after a semantics-preserving
    rewrite, then validates the resulting strict package contract.
    """

    if graph.get("format") != GRAPH_FORMAT:
        _fail("VXQSTORE020", f"graph format must be {GRAPH_FORMAT!r}")
    _reject_legacy_quantization(graph)
    root = graph.get("quantization")
    if root is None:
        validate_external_quantization(graph, tensors)
        return 0
    if not isinstance(root, Mapping) or set(root) != {"format", "tensors"}:
        _fail("VXQSTORE025", "quantization must contain exactly format and tensors")
    if root.get("format") != QUANTIZATION_FORMAT:
        _fail("VXQSTORE026", f"unsupported quantization format {root.get('format')!r}")
    table = root.get("tensors")
    if not isinstance(table, Mapping):
        _fail("VXQSTORE027", "quantization tensors must be a non-empty object")

    declared = {str(name) for name in tensors}
    inputs = graph.get("inputs")
    if isinstance(inputs, Mapping):
        declared.update(str(name) for name in inputs)
    nodes = graph.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            outputs = node.get("outputs") if isinstance(node, Mapping) else None
            if isinstance(outputs, Mapping):
                declared.update(
                    name for name in outputs.values() if isinstance(name, str)
                )

    retained = {
        name: copy.deepcopy(descriptor)
        for name, descriptor in table.items()
        if isinstance(name, str) and name in declared
    }
    removed = len(table) - len(retained)
    if retained:
        graph["quantization"] = {
            "format": QUANTIZATION_FORMAT,
            "tensors": retained,
        }
    else:
        graph.pop("quantization", None)
    validate_external_quantization(graph, tensors)
    return removed


def validate_external_quantization(
    graph: Mapping[str, Any], tensors: Mapping[str, Any]
) -> dict[str, ResolvedQuantization]:
    """Validate and resolve the canonical reference-only quantization table."""

    if graph.get("format") != GRAPH_FORMAT:
        _fail("VXQSTORE020", f"graph format must be {GRAPH_FORMAT!r}")
    _reject_legacy_quantization(graph)
    inputs = graph.get("inputs")
    nodes = graph.get("nodes")

    dtypes, shapes = _declared_tensors(graph, tensors)
    root = graph.get("quantization")
    if root is None:
        return {}
    if not isinstance(root, Mapping) or set(root) != {"format", "tensors"}:
        _fail("VXQSTORE025", "quantization must contain exactly format and tensors")
    if root.get("format") != QUANTIZATION_FORMAT:
        _fail("VXQSTORE026", f"unsupported quantization format {root.get('format')!r}")
    table = root.get("tensors")
    if not isinstance(table, Mapping) or not table:
        _fail("VXQSTORE027", "quantization tensors must be a non-empty object")

    parameter_names: set[str] = set()
    resolved: dict[str, ResolvedQuantization] = {}
    for name, descriptor in table.items():
        if not isinstance(name, str) or name not in dtypes:
            _fail("VXQSTORE028", f"quantization declares unknown tensor {name!r}")
        if dtypes[name] not in RUNTIME_BYTE_DTYPES:
            _fail("VXQSTORE029", f"quantization target {name!r} is not int8/uint8", tensor=name)
        if not isinstance(descriptor, Mapping):
            _fail("VXQSTORE030", f"quantization descriptor for {name!r} must be an object", tensor=name)
        scheme = descriptor.get("scheme")
        allowed = {"scheme", "scale_tensor", "zero_point_tensor"}
        if scheme == "per_axis":
            allowed.add("axis")
        if set(descriptor) != allowed:
            _fail("VXQSTORE031", f"quantization descriptor for {name!r} has invalid fields", tensor=name)
        scale_name = descriptor.get("scale_tensor")
        zero_name = descriptor.get("zero_point_tensor")
        if not isinstance(scale_name, str) or not isinstance(zero_name, str) or scale_name == zero_name:
            _fail("VXQSTORE032", f"quantization parameters for {name!r} require distinct tensor names", tensor=name)
        if scale_name not in tensors or zero_name not in tensors:
            _fail("VXQSTORE033", f"quantization parameters for {name!r} are absent from safetensors", tensor=name)
        parameter_names.update((scale_name, zero_name))
        scale_values = np.asarray(tensors[scale_name])
        zero_values = np.asarray(tensors[zero_name])
        if scale_values.dtype != np.float32 or scale_values.ndim != 1:
            _fail("VXQSTORE034", f"scale tensor for {name!r} must be rank-1 F32", tensor=name)
        expected_zero_dtype = np.dtype(np.int8 if dtypes[name] == "int8" else np.uint8)
        if zero_values.dtype != expected_zero_dtype or zero_values.ndim != 1:
            _fail(
                "VXQSTORE035",
                f"zero-point tensor for {name!r} must be rank-1 {dtypes[name]}",
                tensor=name,
            )
        axis: int | None = None
        expected_count = 1
        if scheme == "per_axis":
            axis_value = descriptor.get("axis")
            shape = shapes.get(name)
            if isinstance(axis_value, bool) or not isinstance(axis_value, int) or not shape:
                _fail("VXQSTORE036", f"per-axis target {name!r} requires a concrete axis", tensor=name)
            axis = axis_value + len(shape) if axis_value < 0 else axis_value
            if axis < 0 or axis >= len(shape):
                _fail("VXQSTORE037", f"per-axis target {name!r} axis is outside its rank", tensor=name)
            expected_count = shape[axis]
        elif scheme != "per_tensor":
            _fail("VXQSTORE038", f"unsupported scheme {scheme!r} for {name!r}", tensor=name)
        if scale_values.shape != (expected_count,) or zero_values.shape != (expected_count,):
            _fail(
                "VXQSTORE039",
                f"quantization parameters for {name!r} must have shape [{expected_count}]",
                tensor=name,
            )
        if any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in scale_values):
            _fail("VXQSTORE040", f"scales for {name!r} must be finite and positive", tensor=name)
        resolved[name] = ResolvedQuantization(
            str(scheme), axis, scale_values.copy(), zero_values.copy()
        )

    if parameter_names.intersection(table):
        _fail("VXQSTORE041", "quantization parameter tensors cannot themselves be quantized")
    graph_inputs = graph.get("inputs")
    if isinstance(graph_inputs, Mapping) and parameter_names.intersection(graph_inputs):
        _fail("VXQSTORE045", "quantization parameters cannot be public graph inputs")
    graph_outputs = graph.get("outputs")
    if isinstance(graph_outputs, list) and parameter_names.intersection(
        name for name in graph_outputs if isinstance(name, str)
    ):
        _fail("VXQSTORE046", "quantization parameters cannot be public graph outputs")
    if isinstance(nodes, list):
        for index, node in enumerate(nodes):
            if not isinstance(node, Mapping):
                continue
            node_inputs = node.get("inputs")
            node_outputs = node.get("outputs")
            if not isinstance(node_inputs, Mapping) or not isinstance(node_outputs, Mapping):
                continue
            if parameter_names.intersection(
                value for value in node_outputs.values() if isinstance(value, str)
            ):
                _fail("VXQSTORE047", "quantization parameters cannot be node outputs")
            label = node.get("id") or node.get("source_name") or node.get("opType") or index
            if node.get("opType") == "QuantizeLinear":
                for target in node_outputs.values():
                    descriptor = table.get(target) if isinstance(target, str) else None
                    if not isinstance(descriptor, Mapping) or (
                        descriptor.get("scale_tensor") != node_inputs.get("scale")
                        or descriptor.get("zero_point_tensor") != node_inputs.get("zero_point")
                    ):
                        _fail(
                            "VXQSTORE048",
                            f"QuantizeLinear node {label!r} inputs do not match output quantization references",
                        )
            elif node.get("opType") == "DequantizeLinear":
                target = node_inputs.get("input")
                descriptor = table.get(target) if isinstance(target, str) else None
                if not isinstance(descriptor, Mapping) or (
                    descriptor.get("scale_tensor") != node_inputs.get("scale")
                    or descriptor.get("zero_point_tensor") != node_inputs.get("zero_point")
                ):
                    _fail(
                        "VXQSTORE049",
                        f"DequantizeLinear node {label!r} inputs do not match input quantization references",
                    )
    return resolved
