"""Strict bridge between packaged ``volvox-graph/v1`` and the typed RuntimeIR."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import Diagnostic, ExporterError
from .generated.kernel_registry import PROFILE_MEMBERS
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
from .shape_system import (
    ShapeContractError,
    ShapeEnvironment,
    TensorShapeSpec,
    create_tensor_shape_spec,
)


_NODE_FIELDS = frozenset({
    "id", "opType", "inputs", "outputs", "params",
})
_ROOT_FIELDS = frozenset({
    "format", "dimensions", "inputs", "outputs", "nodes",
    "banks", "quantization",
})
_INPUT_FIELDS = frozenset({"dtype", "shape"})
_OUTPUT_FIELDS = frozenset({"tensor", "dtype", "shape"})
_RUNTIME_GRAPH_DTYPES = frozenset({"float32", "int32", "int8", "uint8"})
_MAX_JSON_SAFE_INTEGER = JS_NUMBER_MAX_SAFE_INTEGER
_DYNAMIC_QUANTIZED_IMPORT_MEMBERS = tuple(PROFILE_MEMBERS["portable"])


def _requires_dynamic_quantized_domain_proof(
    document: Mapping[str, Any],
) -> bool:
    dimensions = document.get("dimensions")
    quantization = document.get("quantization")
    quantized_tensors = (
        quantization.get("tensors")
        if isinstance(quantization, Mapping)
        else None
    )
    return (
        isinstance(dimensions, Mapping)
        and bool(dimensions)
        and isinstance(quantized_tensors, Mapping)
        and bool(quantized_tensors)
    )


def prove_dynamic_quantized_runtime_domain(
    document: Mapping[str, Any],
    tensors: Mapping[str, Any],
):
    """Return the immutable exact-domain proof required by dynamic W8A8 import."""

    if not _requires_dynamic_quantized_domain_proof(document):
        return None
    from .portable_domain import (
        PortableDomainProofError,
        prove_portable_graph_domain,
    )

    try:
        return prove_portable_graph_domain(
            document,
            tensors,
            _DYNAMIC_QUANTIZED_IMPORT_MEMBERS,
        )
    except PortableDomainProofError as error:
        _fail(
            "VXRTIR037",
            "dynamic quantized RuntimeIR failed canonical bounded-domain "
            f"affine validation ({error.code}): {error}",
        )


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


def _shape_spec(
    value: Any,
    *,
    name: str,
    dtype: str,
    environment: ShapeEnvironment,
) -> TensorShapeSpec:
    try:
        shape = create_tensor_shape_spec(value, environment, f"tensor {name!r} shape")
        maximum = tuple(
            environment.get(axis).max if isinstance(axis, str) else axis
            for axis in shape
        )
        runtime_tensor_allocation(maximum, dtype)
    except (ShapeContractError, ValueError) as error:
        _fail(
            "VXRTIR003",
            f"tensor {name!r} has an invalid bounded shape: {error}",
            name=name,
        )
    return shape


def _shape_environment(document: Mapping[str, Any]) -> ShapeEnvironment:
    dimensions = document.get("dimensions")
    if not isinstance(dimensions, Mapping):
        _fail("VXRTIR033", "runtime graph dimensions must be an object")
    constraints: list[dict[str, Any]] = []
    for name, descriptor in dimensions.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            _fail("VXRTIR033", "runtime graph dimension descriptor is malformed")
        if "name" in descriptor:
            _fail(
                "VXRTIR033",
                f"dimension {name!r} must use its object key as the symbol name",
            )
        constraints.append({"name": name, **dict(descriptor)})
    try:
        return ShapeEnvironment(tuple(constraints))
    except ShapeContractError as error:
        _fail("VXRTIR033", f"invalid runtime dimension constraints: {error}")


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


def _validate_runtime_document_shape(
    document: Mapping[str, Any],
) -> ShapeEnvironment:
    """Reject incomplete dynamic-v1 descriptors before affine lookup."""

    environment = _shape_environment(document)
    unsupported_root_fields = sorted(set(document) - _ROOT_FIELDS)
    if unsupported_root_fields:
        _fail(
            "VXRTIR038",
            "runtime graph root contains unsupported field "
            f"{unsupported_root_fields[0]!r}",
        )

    banks = document.get("banks")
    if banks is not None:
        if not isinstance(banks, Mapping):
            _fail("VXRTIR041", "runtime graph banks must be an object")
        for name, dimension in banks.items():
            _require_runtime_name(name, "graph bank name")
            if not isinstance(dimension, str) or environment.get(dimension) is None:
                _fail(
                    "VXRTIR041",
                    f"graph bank {name!r} must name a declared dimension",
                    name=name,
                )

    inputs = document.get("inputs")
    if not isinstance(inputs, Mapping):
        _fail("VXRTIR006", "runtime graph inputs must be an object")
    for name, descriptor in inputs.items():
        if not isinstance(descriptor, Mapping):
            _fail("VXRTIR006", "runtime graph input descriptor is malformed")
        _require_runtime_name(name, "graph input name")
        unsupported_input_fields = sorted(set(descriptor) - _INPUT_FIELDS)
        if unsupported_input_fields:
            _fail(
                "VXRTIR039",
                f"graph input {name!r} contains unsupported field "
                f"{unsupported_input_fields[0]!r}",
                name=name,
            )
        dtype = descriptor.get("dtype")
        if dtype not in _RUNTIME_GRAPH_DTYPES:
            _fail(
                "VXRTIR008",
                f"graph input {name!r} requires an explicit canonical runtime dtype",
                name=name,
            )
        _shape_spec(
            descriptor.get("shape"),
            name=name,
            dtype=dtype,
            environment=environment,
        )

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
        if "outputs_shape" in descriptor or "outputs_dtype" in descriptor:
            _fail(
                "VXRTIR034",
                f"node {node_index} uses legacy split output descriptors; "
                "re-export required",
            )
        unsupported_node_fields = sorted(set(descriptor) - _NODE_FIELDS)
        if unsupported_node_fields:
            _fail(
                "VXRTIR040",
                f"node {node_index} contains unsupported field "
                f"{unsupported_node_fields[0]!r}",
            )
        outputs = descriptor.get("outputs")
        op_type = descriptor.get("opType")
        inputs_map = descriptor.get("inputs", {})
        if (not isinstance(op_type, str) or
                not isinstance(inputs_map, Mapping) or
                not isinstance(outputs, Mapping) or not outputs):
            _fail("VXRTIR011", f"node {node_index} has malformed runtime fields")
        _require_runtime_name(op_type, f"node {node_index} opType")
        for port, tensor_name in inputs_map.items():
            _require_runtime_name(port, f"node {node_index} input port")
            _require_runtime_name(tensor_name, f"node {node_index} input tensor")
        for port, output_descriptor in outputs.items():
            _require_runtime_name(port, f"node {node_index} output port")
            if (
                not isinstance(output_descriptor, Mapping)
                or set(output_descriptor) != _OUTPUT_FIELDS
            ):
                _fail(
                    "VXRTIR020",
                    f"node {node_index} output {port!r} must contain exactly "
                    "tensor, dtype, and shape assertions",
                )
            tensor_name = output_descriptor.get("tensor")
            _require_runtime_name(tensor_name, f"node {node_index} output tensor")
            dtype = output_descriptor.get("dtype")
            if dtype not in _RUNTIME_GRAPH_DTYPES:
                _fail(
                    "VXRTIR014",
                    f"node {node_index} output {port!r} requires a canonical runtime dtype",
                )
            _shape_spec(
                output_descriptor.get("shape"),
                name=tensor_name,
                dtype=dtype,
                environment=environment,
            )
        params = descriptor.get("params")
        if not isinstance(params, Mapping):
            _fail("VXRTIR022", f"node {node_index} params must be an object")
        retired_path = find_retired_affine_param_path(params)
        if retired_path is not None:
            _fail(
                "VXRTIR023",
                f"node {node_index} contains retired affine payload at {retired_path}",
            )
    return environment


def _runtime_params(node: OpNode) -> Mapping[str, Any]:
    values = [
        attribute.value
        for attribute in node.attributes
        if attribute.name == "params" and attribute.kind == "volvox.params"
    ]
    return values[0] if len(values) == 1 and isinstance(values[0], Mapping) else {}


def _operator_shape_request(
    graph: GraphIR,
    node: OpNode,
    *,
    logical: bool,
) -> dict[str, Any]:
    """Adapt one RuntimeIR node to the shared operator-contract request.

    Declared outputs are always supplied. Author-targeted contracts such as
    Resize and symbolic target shapes cannot be inferred from inputs alone.
    Quantized nodes are kept on their affine-aware validation path because a
    GraphIR stores parameter references, not the resolved numeric descriptors
    consumed by this shape-contract API.
    """

    request: dict[str, Any] = {
        "inputs": {
            port: {
                "shape": list(graph.tensors[name].shape),
                "dtype": graph.tensors[name].dtype,
            }
            for port, name in node.input_map().items()
        },
        "params": dict(_runtime_params(node)),
        "declaredOutputs": {
            port: {
                "shape": list(graph.tensors[name].shape),
                "dtype": graph.tensors[name].dtype,
            }
            for port, name in node.output_map().items()
        },
    }
    if logical:
        request["environment"] = graph.shape_environment
    return request


def _verify_node_output_assertions(graph: GraphIR) -> None:
    """Check logical output descriptors against canonical shape functions.

    Concrete descriptors continue through the exhaustive descriptor validator
    below.  A symbolic descriptor is admitted only when this bridge has an
    explicit symbolic-preserving rule; it is never trusted merely because the
    JSON document supplied a plausible output shape.
    """

    for node in graph.nodes:
        input_names = node.input_map()
        output_names = node.output_map()
        descriptors = {
            port: graph.tensors[name]
            for port, name in output_names.items()
        }
        symbolic = any(
            isinstance(axis, str)
            for name in (*input_names.values(), *output_names.values())
            for axis in graph.tensors[name].shape
        )
        if symbolic and any(
            graph.tensors[name].quantization is not None
            for name in (*input_names.values(), *output_names.values())
        ):
            # GraphIR carries affine tensor references, not their numeric
            # safetensors values. The closed document validator below hydrates
            # those values and performs the canonical whole-domain proof; a
            # reference-free request here would necessarily be incomplete.
            continue
        if not symbolic:
            # Concrete direct operators use the shared generated-ID-backed
            # canonical implementation directly.  Quantized descriptors need
            # resolved numeric affine values and continue through the exact
            # package descriptor validator below.
            if any(
                graph.tensors[name].quantization is not None
                for name in (*input_names.values(), *output_names.values())
            ) or any(
                graph.tensors[name].dtype == "float16"
                for name in input_names.values()
            ):
                continue
            from .operator_shape_contracts import (
                OperatorShapeContractError,
                infer_concrete_operator_shapes,
            )

            request = _operator_shape_request(graph, node, logical=False)
            try:
                inferred = infer_concrete_operator_shapes(node.op_type, request)
            except OperatorShapeContractError as error:
                if error.code == "UNKNOWN_OPERATOR":
                    continue
                _fail(
                    "VXRTIR035",
                    f"node {node.name!r} violates canonical concrete shape "
                    f"inference: {error}",
                    name=node.name,
                )
            if set(inferred) != set(descriptors):
                _fail(
                    "VXRTIR036",
                    f"node {node.name!r} output ports disagree with canonical "
                    "shape inference",
                    name=node.name,
                )
            for port, expected_descriptor in inferred.items():
                declared = descriptors[port]
                if (
                    declared.shape != expected_descriptor.shape
                    or declared.dtype != expected_descriptor.dtype
                ):
                    _fail(
                        "VXRTIR036",
                        f"node {node.name!r} output {port!r} asserts shape/dtype "
                        f"{declared.shape!r}/{declared.dtype!r}, but canonical "
                        "inference requires "
                        f"{expected_descriptor.shape!r}/"
                        f"{expected_descriptor.dtype!r}",
                        name=node.name,
                    )
            continue

        # Symbolic RuntimeIR is admitted only through the shared canonical
        # bounded-domain proof layer. Deferred or unsupported operators have no
        # proof route and therefore fail closed before descriptor validation.
        from .operator_shape_contracts import (
            OperatorShapeContractError,
            get_operator_shape_contract,
        )

        try:
            contract = get_operator_shape_contract(node.op_type)
        except OperatorShapeContractError as error:
            _fail(
                "VXRTIR035",
                f"node {node.name!r} has no canonical bounded-domain shape "
                f"proof for {node.op_type!r}: {error}",
                name=node.name,
            )
        proof = contract.prove_domain(
            _operator_shape_request(graph, node, logical=True)
        )
        if not proof.supported:
            _fail(
                "VXRTIR035",
                f"node {node.name!r} fails canonical bounded-domain "
                f"inference ({proof.code}): {proof.reason}",
                name=node.name,
            )
        if set(proof.outputs) != set(descriptors):
            _fail(
                "VXRTIR036",
                f"node {node.name!r} output ports disagree with canonical "
                "bounded-domain inference",
                name=node.name,
            )
        for port, inferred_descriptor in proof.outputs.items():
            declared = descriptors[port]
            if (
                declared.shape != inferred_descriptor.shape
                or declared.dtype != inferred_descriptor.dtype
            ):
                _fail(
                    "VXRTIR036",
                    f"node {node.name!r} output {port!r} asserts shape/dtype "
                    f"{declared.shape!r}/{declared.dtype!r}, but canonical "
                    f"bounded-domain inference requires "
                    f"{inferred_descriptor.shape!r}/"
                    f"{inferred_descriptor.dtype!r}",
                    name=node.name,
                )


def import_runtime_package(
    document: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    source_name: str = "graph.json",
    bounded_domain_proof: object | None = None,
) -> GraphIR:
    """Load a fully resolved package into verified RuntimeIR.

    The bridge is intentionally strict: it never migrates former inline or
    companion metadata and it requires the actual safetensors inventory.
    """

    if not isinstance(document, Mapping) or document.get("format") != GRAPH_FORMAT:
        _fail("VXRTIR004", f"runtime document must use {GRAPH_FORMAT}")
    _validate_json_value(document)
    dynamic_quantized = _requires_dynamic_quantized_domain_proof(document)
    validated_dynamic_proof = False
    if dynamic_quantized:
        from .portable_domain import PortableGraphDomainProof

        if not isinstance(bounded_domain_proof, PortableGraphDomainProof):
            _fail(
                "VXRTIR037",
                "dynamic quantized RuntimeIR requires one immutable canonical "
                "bounded-domain affine proof before typed import",
            )
        if (
            bounded_domain_proof.backend_members
            != _DYNAMIC_QUANTIZED_IMPORT_MEMBERS
        ):
            _fail(
                "VXRTIR037",
                "dynamic quantized RuntimeIR proof must target exactly the "
                "canonical CPU/WASM/WebGPU import contract",
            )
        expected_proof = prove_dynamic_quantized_runtime_domain(
            document,
            tensors,
        )
        if bounded_domain_proof != expected_proof:
            _fail(
                "VXRTIR037",
                "dynamic quantized RuntimeIR proof is stale or does not match "
                "the exact graph, bounds, quantization references, and tensor payloads",
            )
        validated_dynamic_proof = True
    graph_document = copy.deepcopy(dict(document))
    environment = _validate_runtime_document_shape(graph_document)
    resolved = validate_external_quantization(graph_document, tensors)
    graph = GraphIR(
        source_format="volvoxai",
        source_name=source_name,
        dialect=IRDialect.RUNTIME,
        shape_environment=environment,
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
            shape=_shape_spec(
                descriptor.get("shape"),
                name=name,
                dtype=dtype,
                environment=environment,
            ),
            dtype=dtype,
            source_dtype=dtype,
            public_input=True,
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
        if (not isinstance(op_type, str) or not op_type or
                not isinstance(inputs_map, Mapping) or
                not isinstance(outputs_map, Mapping) or not outputs_map):
            _fail("VXRTIR011", f"node {node_index} has malformed runtime fields")
        params = descriptor.get("params")
        if not isinstance(params, Mapping):
            _fail("VXRTIR022", f"node {node_index} params must be an object")
        normalized_outputs: dict[str, str] = {}
        for port, output_descriptor in outputs_map.items():
            if not isinstance(port, str) or not isinstance(output_descriptor, Mapping):
                _fail("VXRTIR012", f"node {node_index} has an invalid output")
            name = output_descriptor.get("tensor")
            if not isinstance(name, str) or not name:
                _fail("VXRTIR012", f"node {node_index} has an invalid output tensor")
            if name in graph.tensors:
                _fail("VXRTIR013", f"node output {name!r} is multiply defined", name=name)
            dtype = output_descriptor.get("dtype")
            if not isinstance(dtype, str):
                _fail(
                    "VXRTIR014",
                    f"node output {name!r} requires an explicit dtype",
                    name=name,
                )
            graph.add_tensor(TensorValue(
                name=name,
                shape=_shape_spec(
                    output_descriptor.get("shape"),
                    name=name,
                    dtype=dtype,
                    environment=environment,
                ),
                dtype=dtype,
                source_dtype=dtype,
            ))
            normalized_outputs[port] = name
        attributes = (
            OpAttribute("params", "volvox.params", copy.deepcopy(params)),
        )
        node_name = descriptor.get("id")
        if not isinstance(node_name, str) or not node_name:
            _fail("VXRTIR010", f"node {node_index} requires a non-empty id")
        graph.add_node(OpNode.from_maps(
            name=node_name,
            op_type=op_type,
            inputs=dict(inputs_map),
            outputs=normalized_outputs,
            attributes=attributes,
            provenance=(Provenance(
                source_format="volvoxai",
                source_name=node_name,
                source_op=op_type,
                location=f"{source_name}:nodes[{node_index}]",
            ),),
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
    _verify_node_output_assertions(graph)
    from .portable_quantized_validation import validate_portable_quantized_graph

    if resolved:
        if graph.shape_environment.dimensions:
            if not validated_dynamic_proof:
                _fail(
                    "VXRTIR037",
                    "dynamic quantized RuntimeIR requires bounded-domain affine "
                    "validation before publication",
                )
        else:
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


def _detect_weight_banks(published) -> dict[str, tuple[str, int]]:
    """Find fixed weights whose axis 0 is selected at run time.

    Two shapes qualify. `MoELinear`'s ``expert_weight`` is an expert bank by
    definition. A `Gather` with ``axis`` 0 over an initializer whose indices are
    themselves runtime values is the LoRA-family form that constant folding
    leaves behind (``Unsqueeze`` per family, then ``Concat``, then ``Gather``).

    `MoERouter`'s weight is deliberately excluded: it is ``[d_model, experts]``,
    so its expert axis is 1 and it is never slot-indexed.

    Returns tensor name -> (synthesized dimension name, slot count).
    """
    banks: dict[str, tuple[str, int]] = {}

    def consider(name: str) -> None:
        tensor = published.tensors.get(name)
        if tensor is None or not tensor.initializer or name in banks:
            return
        if len(tensor.shape) < 2:
            return
        slots = tensor.shape[0]
        if not isinstance(slots, int) or slots < 1:
            return
        # A per-bank dimension keeps declarations independent; the exported slot
        # count becomes the upper bound, and 1 the lower.
        banks[name] = (f"bank_{name}", slots)

    for node in published.nodes:
        inputs = node.input_map()
        if node.op_type == "MoELinear":
            weight = inputs.get("expert_weight") or inputs.get("weight")
            if weight:
                consider(weight)
            continue
        if node.op_type != "Gather":
            continue
        source = inputs.get("input")
        indices = inputs.get("indices")
        if not source or not indices:
            continue
        index_tensor = published.tensors.get(indices)
        if index_tensor is None or index_tensor.initializer:
            # A constant selection folds away; it is not a runtime bank.
            continue
        axis = 0
        for attribute in node.attributes:
            if attribute.name == "params" and attribute.kind == "volvox.params":
                axis = attribute.value.get("axis", 0)
        if axis in (0, -len(published.tensors[source].shape)):
            consider(source)
    return banks


def export_runtime_package(
    graph: GraphIR,
    tensors: Mapping[str, Any],
    *,
    shape_profile: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize bounded RuntimeIR and prune unreachable safetensors data.

    Passing ``shape_profile`` is the sole constant-binding publication mode.
    It emits the same dynamic-first v1 schema with an empty ``dimensions``
    table; it never falls back to the retired split-output representation.
    """

    published = graph if shape_profile is None else graph.bind_shape_profile(shape_profile)
    published.verify_logical_polymorphic()
    _verify_node_output_assertions(published)
    # Provenance, optimizer reports, and ABI-change reports are deliberately
    # out-of-band. The v1 graph JSON is the closed executable schema consumed
    # by the TypeScript logical loader.
    document: dict[str, Any] = {
        "format": GRAPH_FORMAT,
        "dimensions": {
            constraint.name: {
                "min": constraint.min,
                "max": constraint.max,
                **(
                    {"multiple_of": constraint.multiple_of}
                    if constraint.multiple_of is not None
                    else {}
                ),
            }
            for constraint in published.shape_environment.dimensions
        },
        "inputs": {},
        "outputs": list(published.outputs),
        "nodes": [],
    }
    banks = _detect_weight_banks(published)
    if banks:
        # Declaring a bank is additive: it only tells a runtime that axis 0 is
        # sliceable and how far it may grow. Nothing is required to use it.
        for tensor_name, (dimension, slots) in sorted(banks.items()):
            document["dimensions"][dimension] = {"min": 1, "max": slots}
        document["banks"] = {
            tensor_name: dimension
            for tensor_name, (dimension, _) in sorted(banks.items())
        }
    for name in published.inputs:
        tensor = published.tensors[name]
        document["inputs"][name] = {
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
        }
    for node in published.nodes:
        params: dict[str, Any] = {}
        for attribute in node.attributes:
            if attribute.name == "params" and attribute.kind == "volvox.params":
                params = copy.deepcopy(attribute.value)
        document["nodes"].append({
            "id": node.name,
            "opType": node.op_type,
            "inputs": node.input_map(),
            "outputs": {
                port.name: {
                    "tensor": port.value,
                    "dtype": published.tensors[port.value].dtype,
                    "shape": list(published.tensors[port.value].shape),
                }
                for port in node.outputs if port.value is not None
            },
            "params": params,
        })
    quantized = {
        name: tensor.quantization for name, tensor in published.tensors.items()
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
        name for name, tensor in published.tensors.items() if tensor.initializer
    }
    output_tensors = {
        name: value for name, value in tensors.items() if name in live_initializers
    }
    validate_external_quantization(document, output_tensors)
    from .capabilities import validate_runtime_descriptors

    validation = validate_runtime_descriptors(document, weights=output_tensors)
    if validation.diagnostics:
        raise ExporterError(validation.diagnostics[0])
    return document, output_tensors
