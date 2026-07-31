"""ONNX ModelProto -> source-faithful VolvoxAI typed IR.

No folding, specialization, Q/DQ collapse, layout conversion, or runtime
operator selection is permitted here.  Those transformations belong to named
verified passes after this importer boundary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from ..errors import Diagnostic, ExporterError
from ..ir import (
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    Provenance,
    TensorDataRef,
    TensorValue,
    ValuePort,
)


_ONNX_DTYPES = {
    0: "onnx.undefined",
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
    17: "float8e4m3fn",
    18: "float8e4m3fnuz",
    19: "float8e5m2",
    20: "float8e5m2fnuz",
    21: "uint4",
    22: "int4",
    23: "float4e2m1",
}


def _fail(code: str, message: str, *, node: Optional[str] = None) -> None:
    raise ExporterError(Diagnostic(
        code=code,
        message=message,
        stage="onnx-import",
        source_node=node,
    ))


def _shape_from_type(tensor_type: Any) -> tuple[int | str | None, ...]:
    if not tensor_type.HasField("shape"):
        return ()
    result: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        choice = dimension.WhichOneof("value")
        if choice == "dim_value":
            result.append(int(dimension.dim_value) if dimension.dim_value > 0 else None)
        elif choice == "dim_param":
            result.append(dimension.dim_param or None)
        else:
            result.append(None)
    return tuple(result)


def _value_type(value_info: Any) -> tuple[str, tuple[int | str | None, ...]]:
    type_proto = value_info.type
    choice = type_proto.WhichOneof("value")
    if choice == "tensor_type":
        tensor_type = type_proto.tensor_type
        return (_ONNX_DTYPES.get(int(tensor_type.elem_type),
                                 f"onnx.dtype.{int(tensor_type.elem_type)}"),
                _shape_from_type(tensor_type))
    if choice == "sparse_tensor_type":
        tensor_type = type_proto.sparse_tensor_type
        element = _ONNX_DTYPES.get(int(tensor_type.elem_type),
                                   f"onnx.dtype.{int(tensor_type.elem_type)}")
        return f"onnx.sparse<{element}>", _shape_from_type(tensor_type)
    return f"onnx.{choice or 'unknown'}", ()


def _tensor_ref(tensor: Any) -> Optional[TensorDataRef]:
    external = {item.key: item.value for item in tensor.external_data}
    if not external:
        return None
    def integer(name: str) -> Optional[int]:
        value = external.get(name)
        if value is None or value == "":
            return None
        try:
            parsed = int(value)
        except ValueError:
            _fail("VXONNX004", f"initializer {tensor.name!r} has invalid external {name}")
        if parsed < 0:
            _fail("VXONNX004", f"initializer {tensor.name!r} has negative external {name}")
        return parsed
    return TensorDataRef(
        tensor_name=tensor.name,
        source_uri=external.get("location"),
        byte_offset=integer("offset"),
        byte_length=integer("length"),
        checksum=external.get("checksum"),
    )


def _attribute_value(attribute: Any, attribute_proto: Any) -> Any:
    kind = int(attribute.type)
    if kind == attribute_proto.FLOAT:
        return float(attribute.f)
    if kind == attribute_proto.INT:
        return int(attribute.i)
    if kind == attribute_proto.STRING:
        return bytes(attribute.s)
    if kind == attribute_proto.FLOATS:
        return tuple(float(value) for value in attribute.floats)
    if kind == attribute_proto.INTS:
        return tuple(int(value) for value in attribute.ints)
    if kind == attribute_proto.STRINGS:
        return tuple(bytes(value) for value in attribute.strings)
    # Tensor, graph, sparse-tensor, and type attributes remain exact in raw.
    return None


def _schema_port_names(onnx: Any, op_type: str, domain: str, version: Optional[int],
                       count: int, *, outputs: bool) -> list[str]:
    if version is None:
        return [("output" if outputs else "input") + f"_{index}"
                for index in range(count)]
    try:
        schema = onnx.defs.get_schema(op_type, version, domain)
        formals = schema.outputs if outputs else schema.inputs
    except Exception:
        return [("output" if outputs else "input") + f"_{index}"
                for index in range(count)]
    names: list[str] = []
    for index in range(count):
        if index < len(formals):
            base = formals[index].name or ("output" if outputs else "input")
        elif formals:
            base = formals[-1].name or ("output" if outputs else "input")
        else:
            base = "output" if outputs else "input"
        if base in names:
            base = f"{base}_{index}"
        names.append(base)
    return names


def _add_value_info(graph: GraphIR, value_info: Any, *, public_input: bool = False,
                    public_output: bool = False) -> None:
    if not value_info.name:
        _fail("VXONNX005", "ONNX value_info has an empty name")
    dtype, shape = _value_type(value_info)
    existing = graph.tensors.get(value_info.name)
    if existing is None:
        graph.add_tensor(TensorValue(
            name=value_info.name,
            shape=shape,
            dtype=dtype,
            source_dtype=dtype,
            public_input=public_input,
            public_output=public_output,
            metadata={"onnx_type_proto": value_info.type.SerializeToString()},
        ))
        return
    if existing.dtype != dtype or existing.shape != shape:
        _fail("VXONNX006",
              f"ONNX value {value_info.name!r} has conflicting type declarations")
    existing.public_input |= public_input
    existing.public_output |= public_output
    existing.metadata.setdefault("onnx_type_proto", value_info.type.SerializeToString())


def _import_graph(onnx: Any, graph_proto: Any, *, source_name: str,
                  opsets: dict[str, int], path: str) -> GraphIR:
    graph = GraphIR(
        source_format="onnx",
        source_name=source_name,
        dialect=IRDialect.SOURCE,
        opsets=dict(opsets),
        metadata={
            "onnx_graph_name": graph_proto.name,
            "onnx_graph_doc": graph_proto.doc_string,
            "onnx_graph_proto_sha256": hashlib.sha256(
                graph_proto.SerializeToString()).hexdigest(),
        },
    )
    for value_info in graph_proto.value_info:
        _add_value_info(graph, value_info)
    for value_info in graph_proto.input:
        _add_value_info(graph, value_info, public_input=True)
        graph.inputs.append(value_info.name)
    for value_info in graph_proto.output:
        _add_value_info(graph, value_info, public_output=True)
        graph.outputs.append(value_info.name)

    for tensor in graph_proto.initializer:
        dtype = _ONNX_DTYPES.get(int(tensor.data_type),
                                 f"onnx.dtype.{int(tensor.data_type)}")
        shape = tuple(int(dimension) if dimension > 0 else None
                      for dimension in tensor.dims)
        existing = graph.tensors.get(tensor.name)
        if existing is None:
            existing = TensorValue(
                name=tensor.name,
                shape=shape,
                dtype=dtype,
                source_dtype=dtype,
            )
            graph.add_tensor(existing)
        elif existing.dtype != dtype or existing.shape != shape:
            _fail("VXONNX007",
                  f"initializer {tensor.name!r} conflicts with its value_info")
        existing.initializer = True
        existing.raw_data = bytes(tensor.raw_data) if tensor.raw_data else None
        existing.data = _tensor_ref(tensor)
        existing.metadata["onnx_tensor_proto"] = tensor.SerializeToString()

    for sparse in graph_proto.sparse_initializer:
        name = sparse.values.name
        if not name or name in graph.tensors:
            _fail("VXONNX008", f"invalid or duplicate sparse initializer {name!r}")
        dtype = _ONNX_DTYPES.get(int(sparse.values.data_type),
                                 f"onnx.dtype.{int(sparse.values.data_type)}")
        graph.add_tensor(TensorValue(
            name=name,
            shape=tuple(int(value) if value > 0 else None for value in sparse.dims),
            dtype=f"onnx.sparse<{dtype}>",
            source_dtype=f"onnx.sparse<{dtype}>",
            initializer=True,
            metadata={"onnx_sparse_tensor_proto": sparse.SerializeToString()},
        ))

    for node_index, node_proto in enumerate(graph_proto.node):
        domain = node_proto.domain or ""
        version = opsets.get(domain)
        node_name = node_proto.name or f"@node/{node_index}:{domain}:{node_proto.op_type}"
        input_names = _schema_port_names(
            onnx, node_proto.op_type, domain, version, len(node_proto.input),
            outputs=False)
        output_names = _schema_port_names(
            onnx, node_proto.op_type, domain, version, len(node_proto.output),
            outputs=True)
        attributes: list[OpAttribute] = []
        regions: list[GraphIR] = []
        for attribute_index, attribute in enumerate(node_proto.attribute):
            kind = onnx.AttributeProto.AttributeType.Name(attribute.type)
            attributes.append(OpAttribute(
                name=attribute.name,
                kind=kind,
                value=_attribute_value(attribute, onnx.AttributeProto),
                raw=attribute.SerializeToString(),
            ))
            if attribute.type == onnx.AttributeProto.GRAPH:
                regions.append(_import_graph(
                    onnx, attribute.g,
                    source_name=f"{source_name}#{node_name}.{attribute.name}",
                    opsets=opsets, path=path))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for region_index, region in enumerate(attribute.graphs):
                    regions.append(_import_graph(
                        onnx, region,
                        source_name=(f"{source_name}#{node_name}.{attribute.name}"
                                     f"[{region_index}]"),
                        opsets=opsets, path=path))
        inputs = tuple(ValuePort(
            input_names[index], value if value else None, index)
            for index, value in enumerate(node_proto.input))
        outputs = tuple(ValuePort(
            output_names[index], value if value else None, index)
            for index, value in enumerate(node_proto.output))
        for port in outputs:
            if port.value is None or port.value in graph.tensors:
                continue
            graph.add_tensor(TensorValue(
                name=port.value,
                shape=(),
                dtype="onnx.undefined",
                source_dtype="onnx.undefined",
                metadata={"onnx_missing_value_info": True},
            ))
        graph.add_node(OpNode(
            name=node_name,
            op_type=node_proto.op_type,
            inputs=inputs,
            outputs=outputs,
            domain=domain,
            version=version,
            attributes=tuple(attributes),
            regions=tuple(regions),
            provenance=(Provenance(
                source_format="onnx",
                source_name=node_proto.name,
                source_op=node_proto.op_type,
                location=f"{path}:graph.node[{node_index}]",
                domain=domain,
                version=version,
            ),),
            metadata={
                "onnx_node_proto": node_proto.SerializeToString(),
                "onnx_original_name": node_proto.name,
            },
        ))

    locally_defined = set(graph.inputs)
    locally_defined.update(name for name, tensor in graph.tensors.items()
                           if tensor.initializer)
    locally_defined.update(
        port.value for node in graph.nodes for port in node.outputs
        if port.value is not None)
    captures: list[str] = []
    for node in graph.nodes:
        for port in node.inputs:
            if port.value is None or port.value in locally_defined or port.value in captures:
                continue
            captures.append(port.value)
            if port.value not in graph.tensors:
                graph.add_tensor(TensorValue(
                    name=port.value,
                    shape=(),
                    dtype="onnx.undefined",
                    source_dtype="onnx.undefined",
                    metadata={"onnx_implicit_capture": True},
                ))
    graph.captures.extend(captures)
    graph.verify(IRDialect.SOURCE)
    return graph


def import_onnx_source(source: str | Path | Any, *, check: bool = True) -> GraphIR:
    """Import a checked ONNX model without performing any graph rewrite."""

    try:
        import onnx
    except ImportError as error:  # pragma: no cover - environment-specific.
        _fail("VXONNX001", f"ONNX importer dependency is unavailable: {error}")
    path = str(source) if isinstance(source, (str, Path)) else "<ModelProto>"
    try:
        if isinstance(source, (str, Path)):
            if check:
                onnx.checker.check_model(source, full_check=False,
                                         check_custom_domain=False)
            model = onnx.load_model(source, load_external_data=False)
        else:
            model = source
            if check:
                onnx.checker.check_model(model, full_check=False,
                                         check_custom_domain=False)
    except Exception as error:
        _fail("VXONNX002", f"ONNX checker/load failed for {path}: {error}")
    if not isinstance(model, onnx.ModelProto):
        _fail("VXONNX003", "source is not an ONNX ModelProto")
    opsets: dict[str, int] = {}
    for item in model.opset_import:
        domain = item.domain or ""
        if domain in opsets:
            _fail("VXONNX009", f"model repeats opset domain {domain!r}")
        opsets[domain] = int(item.version)
    graph = _import_graph(
        onnx, model.graph, source_name=path, opsets=opsets, path=path)
    graph.functions = tuple(function.SerializeToString() for function in model.functions)
    graph.metadata.update({
        "onnx_ir_version": int(model.ir_version),
        "onnx_producer_name": model.producer_name,
        "onnx_producer_version": model.producer_version,
        "onnx_domain": model.domain,
        "onnx_model_version": int(model.model_version),
        "onnx_doc_string": model.doc_string,
        "onnx_metadata_props": tuple((item.key, item.value)
                                     for item in model.metadata_props),
        "onnx_model_proto_sha256": hashlib.sha256(
            model.SerializeToString()).hexdigest(),
    })
    graph.verify(IRDialect.SOURCE)
    return graph
