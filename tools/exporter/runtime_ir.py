"""Strict bridge between packaged ``volvox-graph/v1`` and the typed RuntimeIR."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .ir import (
    AffineQuantization,
    find_retired_affine_param_path,
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    Provenance,
    TensorDataRef,
    TensorValue,
)
from .quantization_storage import (
    GRAPH_FORMAT,
    QUANTIZATION_FORMAT,
    validate_external_quantization,
)
from .runtime_names import is_runtime_graph_name
from .runtime_tensors import (
    JS_NUMBER_MAX_SAFE_INTEGER,
    runtime_tensor_allocation,
    safetensors_storage_dtype,
)


_NODE_FIELDS = frozenset({
    "id", "opType", "inputs", "outputs", "outputs_shape", "outputs_dtype",
    "params",
})
_ROOT_FIELDS = frozenset({
    "format", "inputs", "outputs", "nodes", "quantization",
})
_RUNTIME_GRAPH_DTYPES = frozenset({"float32", "int32", "int8", "uint8"})
_MAX_JSON_SAFE_INTEGER = JS_NUMBER_MAX_SAFE_INTEGER


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("VXRTIR028", f"runtime graph contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(token: str) -> None:
    _fail("VXRTIR024", f"runtime graph contains non-finite JSON number {token!r}")


def load_runtime_document(path: str | Path) -> dict[str, Any]:
    """Load one strict persisted RuntimeIR document without a parallel wrapper.

    Semantic validation remains the responsibility of
    :func:`import_runtime_package`, which also has the scoped safetensors
    inventory.  This reader owns only lossless JSON syntax and root typing so
    every caller enters the same typed import boundary.
    """

    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except ExporterError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("VXRTIR029", f"cannot read runtime graph {source}: {error}")
    if not isinstance(value, dict):
        _fail("VXRTIR030", "runtime graph root must be an object")
    _validate_json_value(value)
    return value


def _fail(code: str, message: str, *, name: str | None = None) -> None:
    raise ExporterError(Diagnostic(
        code=code,
        message=message,
        stage="runtime-ir",
        source_node=name,
    ))


def _array_metadata(value: Any, *, name: str) -> tuple[str, str, tuple[int, ...]]:
    try:
        array = np.asarray(value)
    except Exception as error:
        _fail("VXRTIR001", f"cannot inspect safetensors value: {error}")
    source_dtype = str(array.dtype)
    dtype = safetensors_storage_dtype(source_dtype)
    if dtype is None:
        _fail("VXRTIR002", f"unsupported packaged tensor dtype {source_dtype!r}")
    shape = tuple(int(dimension) for dimension in array.shape)
    try:
        runtime_tensor_allocation(shape, dtype)
    except ValueError as error:
        _fail(
            "VXRTIR027",
            f"safetensors tensor {name!r} cannot be constructed: {error}",
            name=name,
        )
    return dtype, source_dtype, shape


def _shape(value: Any, *, name: str, dtype: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item <= 0
        or item > _MAX_JSON_SAFE_INTEGER
        for item in value
    ):
        _fail("VXRTIR003", f"tensor {name!r} has an invalid runtime shape", name=name)
    try:
        runtime_tensor_allocation(value, dtype)
    except ValueError as error:
        _fail(
            "VXRTIR027",
            f"runtime tensor {name!r} cannot be constructed: {error}",
            name=name,
        )
    return tuple(value)


def _require_runtime_name(value: Any, label: str) -> None:
    if not is_runtime_graph_name(value):
        _fail(
            "VXRTIR026",
            f"{label} {value!r} is invalid or reserved by the runtime graph",
            name=value if isinstance(value, str) else None,
        )


def _validate_json_value(value: Any, *, path: str = "graph") -> None:
    """Require the in-memory API to represent a lossless JSON document."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if abs(value) <= _MAX_JSON_SAFE_INTEGER:
            return
        _fail("VXRTIR024", f"{path} contains an integer outside JSON's safe range")
    if isinstance(value, float):
        if (
            math.isfinite(value)
            and (not value.is_integer() or abs(value) <= _MAX_JSON_SAFE_INTEGER)
        ):
            return
        _fail("VXRTIR024", f"{path} contains a non-finite or unsafe JSON number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail("VXRTIR024", f"{path} contains a non-string JSON key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    _fail("VXRTIR024", f"{path} contains a non-JSON value {type(value).__name__}")


def _validate_runtime_document_shape(document: Mapping[str, Any]) -> None:
    """Reject incomplete or aliased persisted descriptors before affine lookup."""

    inputs = document.get("inputs")
    if not isinstance(inputs, Mapping):
        _fail("VXRTIR006", "runtime graph inputs must be an object")
    for name, descriptor in inputs.items():
        if not isinstance(descriptor, Mapping):
            _fail("VXRTIR006", "runtime graph input descriptor is malformed")
        _require_runtime_name(name, "graph input name")
        dtype = descriptor.get("dtype")
        if dtype not in _RUNTIME_GRAPH_DTYPES:
            _fail(
                "VXRTIR008",
                f"graph input {name!r} requires an explicit canonical runtime dtype",
                name=name,
            )
        _shape(descriptor.get("shape"), name=name, dtype=dtype)

    graph_outputs = document.get("outputs")
    if (
        not isinstance(graph_outputs, list)
        or not graph_outputs
        or any(
            not is_runtime_graph_name(name)
            for name in graph_outputs
        )
        or len(graph_outputs) != len(set(graph_outputs))
    ):
        _fail(
            "VXRTIR015",
            "runtime graph outputs must be a non-empty ordered array of unique tensor names",
        )

    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        _fail("VXRTIR009", "runtime graph nodes must be an array")
    for node_index, descriptor in enumerate(nodes):
        if not isinstance(descriptor, Mapping):
            _fail("VXRTIR010", f"node {node_index} is not an object")
        if "op" in descriptor:
            _fail(
                "VXRTIR019",
                f"node {node_index} contains unsupported field 'op'; use 'opType'",
            )
        outputs = descriptor.get("outputs")
        output_shapes = descriptor.get("outputs_shape")
        output_dtypes = descriptor.get("outputs_dtype")
        op_type = descriptor.get("opType")
        inputs_map = descriptor.get("inputs", {})
        if (not isinstance(op_type, str) or
                not isinstance(inputs_map, Mapping) or
                not isinstance(outputs, Mapping) or
                not isinstance(output_shapes, Mapping) or
                not isinstance(output_dtypes, Mapping)):
            _fail("VXRTIR011", f"node {node_index} has malformed runtime fields")
        if set(output_shapes) != set(outputs):
            _fail(
                "VXRTIR020",
                f"node {node_index} outputs_shape must exactly describe every output port",
            )
        if set(output_dtypes) != set(outputs):
            _fail(
                "VXRTIR021",
                f"node {node_index} outputs_dtype must exactly describe every output port",
            )
        _require_runtime_name(op_type, f"node {node_index} opType")
        for port, tensor_name in inputs_map.items():
            _require_runtime_name(port, f"node {node_index} input port")
            _require_runtime_name(tensor_name, f"node {node_index} input tensor")
        for port, tensor_name in outputs.items():
            _require_runtime_name(port, f"node {node_index} output port")
            _require_runtime_name(tensor_name, f"node {node_index} output tensor")
            dtype = output_dtypes[port]
            if dtype not in _RUNTIME_GRAPH_DTYPES:
                _fail(
                    "VXRTIR014",
                    f"node {node_index} output {port!r} requires a canonical runtime dtype",
                )
            _shape(output_shapes[port], name=tensor_name, dtype=dtype)
        params = descriptor.get("params")
        if "params" in descriptor and not isinstance(params, Mapping):
            _fail("VXRTIR022", f"node {node_index} params must be an object")
        retired_path = find_retired_affine_param_path(params)
        if retired_path is not None:
            _fail(
                "VXRTIR023",
                f"node {node_index} contains retired affine payload at {retired_path}",
            )


def import_runtime_package(
    document: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    source_name: str = "graph.json",
) -> GraphIR:
    """Load a fully resolved package into verified RuntimeIR.

    The bridge is intentionally strict: it never migrates former inline or
    companion metadata and it requires the actual safetensors inventory.
    """

    if not isinstance(document, Mapping) or document.get("format") != GRAPH_FORMAT:
        _fail("VXRTIR004", f"runtime document must use {GRAPH_FORMAT}")
    _validate_json_value(document)
    graph_document = copy.deepcopy(dict(document))
    _validate_runtime_document_shape(graph_document)
    resolved = validate_external_quantization(graph_document, tensors)
    runtime_root_fields = {
        key: copy.deepcopy(value)
        for key, value in graph_document.items() if key not in _ROOT_FIELDS
    }
    source = runtime_root_fields.get("source")
    abi_changes = source.get("abi_changes") if isinstance(source, Mapping) else None
    graph = GraphIR(
        source_format="volvoxai",
        source_name=source_name,
        dialect=IRDialect.RUNTIME,
        abi_changes=(
            [copy.deepcopy(dict(item)) for item in abi_changes
             if isinstance(item, Mapping)]
            if isinstance(abi_changes, list) else []
        ),
        metadata={
            "runtime_root_fields": runtime_root_fields,
        },
    )

    for name, value in tensors.items():
        if not is_runtime_graph_name(name):
            _fail(
                "VXRTIR026",
                f"safetensors tensor name {name!r} is invalid or reserved",
                name=name if isinstance(name, str) else None,
            )
        dtype, source_dtype, shape = _array_metadata(value, name=name)
        graph.add_tensor(TensorValue(
            name=name,
            shape=shape,
            dtype=dtype,
            source_dtype=source_dtype,
            initializer=True,
            data=TensorDataRef(tensor_name=name),
        ))

    inputs = graph_document.get("inputs")
    if not isinstance(inputs, Mapping):
        _fail("VXRTIR006", "runtime graph inputs must be an object")
    for name, descriptor in inputs.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            _fail("VXRTIR006", "runtime graph input descriptor is malformed")
        if name in graph.tensors:
            _fail("VXRTIR007", f"graph input {name!r} collides with model data", name=name)
        dtype = descriptor.get("dtype")
        if not isinstance(dtype, str):
            _fail(
                "VXRTIR008",
                f"graph input {name!r} requires an explicit dtype",
                name=name,
            )
        graph.add_tensor(TensorValue(
            name=name,
            shape=_shape(descriptor.get("shape"), name=name, dtype=dtype),
            dtype=dtype,
            source_dtype=dtype,
            public_input=True,
            metadata={
                "runtime_input_fields": {
                    key: copy.deepcopy(value) for key, value in descriptor.items()
                    if key not in {"shape", "dtype", "quantization"}
                },
            },
        ))
        graph.inputs.append(name)

    nodes = graph_document.get("nodes")
    if not isinstance(nodes, list):
        _fail("VXRTIR009", "runtime graph nodes must be an array")
    for node_index, descriptor in enumerate(nodes):
        if not isinstance(descriptor, Mapping):
            _fail("VXRTIR010", f"node {node_index} is not an object")
        if "op" in descriptor:
            _fail(
                "VXRTIR019",
                f"node {node_index} contains unsupported field 'op'; use 'opType'",
            )
        op_type = descriptor.get("opType")
        inputs_map = descriptor.get("inputs", {})
        outputs_map = descriptor.get("outputs")
        output_shapes = descriptor.get("outputs_shape")
        output_dtypes = descriptor.get("outputs_dtype")
        if (not isinstance(op_type, str) or not op_type or
                not isinstance(inputs_map, Mapping) or
                not isinstance(outputs_map, Mapping) or
                not isinstance(output_shapes, Mapping) or
                not isinstance(output_dtypes, Mapping)):
            _fail("VXRTIR011", f"node {node_index} has malformed runtime fields")
        if set(output_shapes) != set(outputs_map):
            _fail(
                "VXRTIR020",
                f"node {node_index} outputs_shape must exactly describe every output port",
            )
        if set(output_dtypes) != set(outputs_map):
            _fail(
                "VXRTIR021",
                f"node {node_index} outputs_dtype must exactly describe every output port",
            )
        params = descriptor.get("params")
        if "params" in descriptor and not isinstance(params, Mapping):
            _fail("VXRTIR022", f"node {node_index} params must be an object")
        for port, name in outputs_map.items():
            if not isinstance(port, str) or not isinstance(name, str) or not name:
                _fail("VXRTIR012", f"node {node_index} has an invalid output")
            if name in graph.tensors:
                _fail("VXRTIR013", f"node output {name!r} is multiply defined", name=name)
            dtype = output_dtypes.get(port)
            if not isinstance(dtype, str):
                _fail(
                    "VXRTIR014",
                    f"node output {name!r} requires an explicit dtype",
                    name=name,
                )
            graph.add_tensor(TensorValue(
                name=name,
                shape=_shape(output_shapes.get(port), name=name, dtype=dtype),
                dtype=dtype,
                source_dtype=dtype,
            ))
        attributes = () if params is None else (
            OpAttribute("params", "volvox.params", copy.deepcopy(params)),
        )
        node_name = descriptor.get("id")
        if not isinstance(node_name, str) or not node_name:
            node_name = f"@runtime/{node_index}:{op_type}"
        graph.add_node(OpNode.from_maps(
            name=node_name,
            op_type=op_type,
            inputs=dict(inputs_map),
            outputs=dict(outputs_map),
            attributes=attributes,
            provenance=(Provenance(
                source_format="volvoxai",
                source_name=node_name,
                source_op=op_type,
                location=f"{source_name}:nodes[{node_index}]",
            ),),
            metadata={
                "runtime_node_fields": {
                    key: copy.deepcopy(value)
                    for key, value in descriptor.items() if key not in _NODE_FIELDS
                },
            },
        ))

    outputs = graph_document.get("outputs")
    if not isinstance(outputs, list) or not outputs or any(
        not is_runtime_graph_name(name) for name in outputs
    ):
        _fail("VXRTIR015", "runtime graph outputs must be a non-empty ordered string array")
    for name in outputs:
        tensor = graph.tensors.get(name)
        if tensor is None:
            _fail("VXRTIR016", f"runtime output {name!r} is unresolved", name=name)
        tensor.public_output = True
        graph.outputs.append(name)

    for name, quantization in resolved.items():
        tensor = graph.tensors.get(name)
        descriptor = graph_document["quantization"]["tensors"][name]
        if tensor is None:
            _fail("VXRTIR017", f"quantized tensor {name!r} is unresolved", name=name)
        tensor.quantization = AffineQuantization(
            scheme=quantization.scheme,
            axis=quantization.axis,
            scale=descriptor["scale_tensor"],
            zero_point=descriptor["zero_point_tensor"],
        )
    graph.verify(IRDialect.RUNTIME)
    from .portable_quantized_validation import validate_portable_quantized_graph

    validate_portable_quantized_graph(graph, resolved)
    # Run this after the typed structural/affine verifier so its stable,
    # precise diagnostics retain precedence. Descriptor validation then closes
    # contracts for every float and quantized operator, without selecting or
    # implying a target backend route.
    from .capabilities import validate_runtime_descriptors

    descriptor_validation = validate_runtime_descriptors(
        graph_document,
        weights=tensors,
    )
    if descriptor_validation.diagnostics:
        raise ExporterError(descriptor_validation.diagnostics[0])
    return graph


def export_runtime_package(
    graph: GraphIR,
    tensors: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize verified RuntimeIR and prune unreachable safetensors data."""

    graph.verify(IRDialect.RUNTIME)
    document = copy.deepcopy(graph.metadata.get("runtime_root_fields", {}))
    if graph.abi_changes:
        source = document.setdefault("source", {})
        if not isinstance(source, dict):
            _fail("VXRTIR018", "runtime source must be an object to record ABI changes")
        source["abi_changes"] = copy.deepcopy(graph.abi_changes)
    document["format"] = GRAPH_FORMAT
    document["inputs"] = {}
    for name in graph.inputs:
        tensor = graph.tensors[name]
        descriptor = copy.deepcopy(tensor.metadata.get("runtime_input_fields", {}))
        descriptor["shape"] = list(tensor.shape)
        descriptor["dtype"] = tensor.dtype
        document["inputs"][name] = descriptor
    document["outputs"] = list(graph.outputs)
    document["nodes"] = []
    for node in graph.nodes:
        runtime = copy.deepcopy(node.metadata.get("runtime_node_fields", {}))
        runtime["id"] = node.name
        runtime["opType"] = node.op_type
        runtime["inputs"] = node.input_map()
        runtime["outputs"] = node.output_map()
        runtime["outputs_shape"] = {
            port.name: list(graph.tensors[port.value].shape)
            for port in node.outputs if port.value is not None
        }
        output_dtypes = {
            port.name: graph.tensors[port.value].dtype
            for port in node.outputs if port.value is not None
        }
        # The sole current v1 contract is fully typed: every output port has an
        # explicit dtype, including F32.  Omitting the default here would make
        # a typed round trip less precise and break package capability checks.
        runtime["outputs_dtype"] = output_dtypes
        for attribute in node.attributes:
            if attribute.name == "params" and attribute.kind == "volvox.params":
                runtime["params"] = copy.deepcopy(attribute.value)
        document["nodes"].append(runtime)
    quantized = {
        name: tensor.quantization for name, tensor in graph.tensors.items()
        if tensor.quantization is not None
    }
    if quantized:
        document["quantization"] = {
            "format": QUANTIZATION_FORMAT,
            "tensors": {
                name: {
                    "scheme": quantized[name].scheme,
                    **({"axis": quantized[name].axis}
                       if quantized[name].scheme == "per_axis" else {}),
                    "scale_tensor": quantized[name].scale,
                    "zero_point_tensor": quantized[name].zero_point,
                }
                for name in sorted(quantized)
            },
        }
    live_initializers = {
        name for name, tensor in graph.tensors.items() if tensor.initializer
    }
    output_tensors = {
        name: value for name, value in tensors.items() if name in live_initializers
    }
    validate_external_quantization(document, output_tensors)
    return document, output_tensors
