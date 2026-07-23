"""TensorFlow Lite/LiteRT FlatBuffer -> source-faithful typed IR.

This importer intentionally performs no operator lowering, constant folding,
layout conversion, or quantization rewrite.  It uses the schema bindings from
Google's official ``ai-edge-litert`` package; the handwritten vtable parser in
the historical one-shot exporter is not used here.

Source quantization vectors remain source metadata at this boundary.  A later
verified quantization pass must materialize them as tensor payloads before a
``volvox-graph/v1`` package is published; they are never runtime graph JSON.

FlatBuffer tables are not self-contained byte ranges: offsets in a builtin
options table may point elsewhere in the model.  The IR therefore retains the
exact source model once and records every table's source offset.  It also stores
a canonical standalone encoding for generated option/tensor table objects when
the schema exposes one.  Custom option byte vectors remain byte-for-byte exact.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Optional

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


_TENSOR_DTYPES = {
    0: "float32",
    1: "float16",
    2: "int32",
    3: "uint8",
    4: "int64",
    5: "string",
    6: "bool",
    7: "int16",
    8: "complex64",
    9: "int8",
    10: "float64",
    11: "complex128",
    12: "uint64",
    13: "resource",
    14: "variant",
    15: "uint32",
    16: "uint16",
    17: "int4",
    18: "bfloat16",
    19: "int2",
    20: "uint4",
    21: "float8e4m3fn",
    22: "float8e5m2",
}


def _fail(code: str, message: str, *, node: Optional[str] = None) -> None:
    raise ExporterError(Diagnostic(
        code=code,
        message=message,
        stage="tflite-import",
        source_node=node,
    ))


def _schema_module() -> Any:
    try:
        from ai_edge_litert import schema_py_generated as schema
    except ImportError as error:  # pragma: no cover - environment-specific.
        _fail(
            "VXTFLITE001",
            "the TFLite importer requires Google's official ai-edge-litert "
            f"schema bindings: {error}",
        )
    required = (
        "Model", "ModelT", "OperatorT", "TensorT", "BuiltinOperator",
        "BuiltinOptions", "BuiltinOptions2", "TensorType",
    )
    missing = [name for name in required if not hasattr(schema, name)]
    if missing:
        _fail(
            "VXTFLITE001",
            "the installed ai-edge-litert schema is missing required symbols: "
            + ", ".join(missing),
        )
    return schema


def _read_source(source: str | Path | bytes | bytearray | memoryview) -> tuple[bytes, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return path.read_bytes(), str(path)
        except OSError as error:
            _fail("VXTFLITE002", f"cannot read TFLite model {path}: {error}")
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), "<memory>"
    _fail(
        "VXTFLITE002",
        "TFLite source must be a filesystem path or a bytes-like object",
    )


def _table_position(table: Any) -> Optional[int]:
    raw_table = getattr(table, "_tab", table)
    position = getattr(raw_table, "Pos", None)
    return int(position) if isinstance(position, int) else None


def _enum_names(enum_type: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    for name, value in vars(enum_type).items():
        if name.startswith("_") or not isinstance(value, int):
            continue
        result.setdefault(int(value), name)
    return result


def _enum_name(enum_type: Any, value: int) -> str:
    return _enum_names(enum_type).get(int(value), f"UNKNOWN_{int(value)}")


def _display_text(value: Optional[bytes]) -> str:
    if value is None:
        return ""
    return bytes(value).decode("utf-8", errors="replace")


def _stable_object(value: Any) -> Any:
    """Convert a generated object-API tree into fingerprint-stable values."""

    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, dict):
        return {str(key): _stable_object(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return tuple(_stable_object(item) for item in value)
    if hasattr(value, "tolist"):
        return _stable_object(value.tolist())
    if hasattr(value, "item"):
        try:
            return _stable_object(value.item())
        except (TypeError, ValueError):
            pass
    fields = getattr(value, "__dict__", None)
    if fields is not None:
        result = {"_type": type(value).__name__}
        result.update({name: _stable_object(item)
                       for name, item in sorted(fields.items())})
        return result
    return repr(value)


def _canonical_table(value: Any) -> bytes:
    """Serialize one generated object-API table as a standalone FlatBuffer."""

    if value is None:
        return b""
    import flatbuffers

    builder = flatbuffers.Builder(256)
    root = value.Pack(builder)
    builder.Finish(root)
    return bytes(builder.Output())


def _int_vector(table: Any, field: str) -> tuple[int, ...]:
    length = int(getattr(table, field + "Length")())
    getter = getattr(table, field)
    return tuple(int(getter(index)) for index in range(length))


def _bool_vector(table: Any, field: str) -> tuple[bool, ...]:
    length = int(getattr(table, field + "Length")())
    getter = getattr(table, field)
    return tuple(bool(getter(index)) for index in range(length))


def _bytes_vector(table: Any, field: str) -> bytes:
    length = int(getattr(table, field + "Length")())
    getter = getattr(table, field)
    return bytes(int(getter(index)) for index in range(length))


def _source_tensor_name(subgraph_index: int, tensor_index: int) -> str:
    return f"@tflite/subgraph/{subgraph_index}/tensor/{tensor_index}"


def _validate_tensor_index(
    index: int,
    tensor_count: int,
    *,
    where: str,
    optional: bool = False,
) -> None:
    if optional and index == -1:
        return
    if index < 0 or index >= tensor_count:
        _fail(
            "VXTFLITE006",
            f"{where} references tensor index {index}, but the subgraph has "
            f"{tensor_count} tensors",
            node=where,
        )


def _buffer_descriptor(buffer: Any, buffer_index: int) -> dict[str, Any]:
    data_present = not buffer.DataIsNone()
    data_length = int(buffer.DataLength())
    if data_present:
        data = buffer.DataAsNumpy()
        if not hasattr(data, "shape"):
            data = _bytes_vector(buffer, "Data")
        data_sha256 = hashlib.sha256(data).hexdigest()
    else:
        data_sha256 = None
    return {
        "index": buffer_index,
        "table_offset": _table_position(buffer),
        "data_present": data_present,
        "data_length": data_length,
        "data_sha256": data_sha256,
        "external_offset": int(buffer.Offset()),
        "external_size": int(buffer.Size()),
    }


def _tensor_data_ref(
    tensor_name: str,
    descriptor: dict[str, Any],
    source_name: str,
) -> Optional[TensorDataRef]:
    if descriptor["data_present"]:
        return TensorDataRef(
            tensor_name=tensor_name,
            shard=f"tflite.buffer/{descriptor['index']}",
            source_uri=None if source_name == "<memory>" else source_name,
            byte_length=descriptor["data_length"],
            checksum=descriptor["data_sha256"],
        )
    offset = int(descriptor["external_offset"])
    size = int(descriptor["external_size"])
    if offset or size:
        return TensorDataRef(
            tensor_name=tensor_name,
            source_uri=None if source_name == "<memory>" else source_name,
            byte_offset=offset,
            byte_length=size,
        )
    return None


def _normalized_shape(shape: Iterable[int]) -> tuple[int | None, ...]:
    return tuple(int(dimension) if int(dimension) > 0 else None
                 for dimension in shape)


def _operator_code_descriptor(schema: Any, code: Any, index: int) -> dict[str, Any]:
    deprecated_code = int(code.DeprecatedBuiltinCode())
    builtin_code = int(code.BuiltinCode())
    resolved_code = max(deprecated_code, builtin_code)
    custom_code = code.CustomCode()
    code_object = schema.OperatorCodeT.InitFromObj(code)
    return {
        "index": index,
        "deprecated_builtin_code": deprecated_code,
        "builtin_code": builtin_code,
        "resolved_builtin_code": resolved_code,
        "builtin_name": _enum_name(schema.BuiltinOperator, resolved_code),
        "custom_code": bytes(custom_code) if custom_code is not None else None,
        "version": int(code.Version()),
        "table_offset": _table_position(code),
        "object": _stable_object(code_object),
        "canonical_table": _canonical_table(code_object),
    }


def _option_attribute(
    schema: Any,
    operator: Any,
    operator_object: Any,
    *,
    second: bool,
) -> tuple[Optional[OpAttribute], dict[str, Any]]:
    if second:
        option_type = int(operator.BuiltinOptions2Type())
        option_table = operator.BuiltinOptions2()
        option_object = operator_object.builtinOptions2
        enum_type = schema.BuiltinOptions2
        attribute_name = "builtin_options2"
    else:
        option_type = int(operator.BuiltinOptionsType())
        option_table = operator.BuiltinOptions()
        option_object = operator_object.builtinOptions
        enum_type = schema.BuiltinOptions
        attribute_name = "builtin_options"
    option_name = _enum_name(enum_type, option_type)
    descriptor = {
        "type": option_type,
        "type_name": option_name,
        "table_offset": _table_position(option_table),
    }
    if option_type == 0:
        return None, descriptor
    if option_table is None or option_object is None:
        descriptor["object"] = None
        descriptor["canonical_table"] = b""
        return OpAttribute(
            name=attribute_name,
            kind=option_name,
            value=None,
            raw=b"",
        ), descriptor
    canonical = _canonical_table(option_object)
    stable = _stable_object(option_object)
    descriptor["object"] = stable
    descriptor["canonical_table"] = canonical
    return OpAttribute(
        name=attribute_name,
        kind=option_name,
        value=stable,
        raw=canonical,
    ), descriptor


def _signature_descriptor(
    schema: Any,
    signature: Any,
    index: int,
    tensor_counts: tuple[int, ...],
) -> dict[str, Any]:
    subgraph_index = int(signature.SubgraphIndex())
    if subgraph_index < 0 or subgraph_index >= len(tensor_counts):
        _fail(
            "VXTFLITE009",
            f"signature {index} references missing subgraph {subgraph_index}",
        )

    def tensor_maps(field: str) -> tuple[dict[str, Any], ...]:
        length = int(getattr(signature, field + "Length")())
        getter = getattr(signature, field)
        result: list[dict[str, Any]] = []
        for map_index in range(length):
            tensor_map = getter(map_index)
            if tensor_map is None:
                _fail("VXTFLITE009", f"signature {index} has a null {field} map")
            tensor_index = int(tensor_map.TensorIndex())
            _validate_tensor_index(
                tensor_index,
                tensor_counts[subgraph_index],
                where=f"signature[{index}].{field}[{map_index}]",
            )
            name = tensor_map.Name()
            result.append({
                "name": bytes(name) if name is not None else None,
                "display_name": _display_text(name),
                "tensor_index": tensor_index,
                "tensor": _source_tensor_name(subgraph_index, tensor_index),
                "table_offset": _table_position(tensor_map),
            })
        return tuple(result)

    key = signature.SignatureKey()
    signature_object = schema.SignatureDefT.InitFromObj(signature)
    return {
        "index": index,
        "signature_key": bytes(key) if key is not None else None,
        "display_signature_key": _display_text(key),
        "subgraph_index": subgraph_index,
        "inputs": tensor_maps("Inputs"),
        "outputs": tensor_maps("Outputs"),
        "table_offset": _table_position(signature),
        "object": _stable_object(signature_object),
        "canonical_table": _canonical_table(signature_object),
    }


def _import_subgraph(
    schema: Any,
    subgraph: Any,
    *,
    subgraph_index: int,
    source_name: str,
    operator_codes: tuple[dict[str, Any], ...],
    buffers: tuple[dict[str, Any], ...],
    raw: bytes,
) -> GraphIR:
    tensor_count = int(subgraph.TensorsLength())
    input_indices = _int_vector(subgraph, "Inputs")
    output_indices = _int_vector(subgraph, "Outputs")
    for interface, indices in (("input", input_indices), ("output", output_indices)):
        for position, tensor_index in enumerate(indices):
            _validate_tensor_index(
                tensor_index,
                tensor_count,
                where=f"subgraph[{subgraph_index}].{interface}[{position}]",
            )

    operators: list[Any] = []
    produced: set[int] = set()
    for operator_index in range(int(subgraph.OperatorsLength())):
        operator = subgraph.Operators(operator_index)
        if operator is None:
            _fail(
                "VXTFLITE005",
                f"subgraph {subgraph_index} has a null operator at {operator_index}",
            )
        operators.append(operator)
        for position, tensor_index in enumerate(_int_vector(operator, "Outputs")):
            _validate_tensor_index(
                tensor_index,
                tensor_count,
                where=(f"subgraph[{subgraph_index}].operator[{operator_index}]"
                       f".output[{position}]"),
                optional=True,
            )
            if tensor_index >= 0:
                produced.add(tensor_index)

    name = subgraph.Name()
    graph = GraphIR(
        source_format="tflite",
        source_name=f"{source_name}#subgraph[{subgraph_index}]",
        dialect=IRDialect.SOURCE,
        metadata={
            "tflite_subgraph_index": subgraph_index,
            "tflite_subgraph_name": bytes(name) if name is not None else None,
            "tflite_display_subgraph_name": _display_text(name),
            "tflite_subgraph_table_offset": _table_position(subgraph),
            "tflite_input_tensor_indices": input_indices,
            "tflite_output_tensor_indices": output_indices,
            "tflite_debug_metadata_index": int(subgraph.DebugMetadataIndex()),
        },
    )
    public_inputs = set(input_indices)
    public_outputs = set(output_indices)

    for tensor_index in range(tensor_count):
        tensor = subgraph.Tensors(tensor_index)
        if tensor is None:
            _fail(
                "VXTFLITE005",
                f"subgraph {subgraph_index} has a null tensor at {tensor_index}",
            )
        tensor_name = _source_tensor_name(subgraph_index, tensor_index)
        dtype_code = int(tensor.Type())
        dtype_name = _enum_name(schema.TensorType, dtype_code)
        shape = _int_vector(tensor, "Shape")
        shape_signature = _int_vector(tensor, "ShapeSignature")
        buffer_index = int(tensor.Buffer())
        if buffer_index < 0 or buffer_index >= len(buffers):
            _fail(
                "VXTFLITE007",
                f"tensor {tensor_name} references missing buffer {buffer_index}",
                node=tensor_name,
            )
        tensor_object = schema.TensorT.InitFromObj(tensor)
        original_name = tensor.Name()
        quantization = tensor.Quantization()
        sparsity = tensor.Sparsity()
        metadata: dict[str, Any] = {
            "tflite_tensor_index": tensor_index,
            "tflite_name": (bytes(original_name)
                            if original_name is not None else None),
            "tflite_display_name": _display_text(original_name),
            "tflite_dtype_code": dtype_code,
            "tflite_dtype_name": dtype_name,
            "tflite_shape": shape,
            "tflite_shape_present": not tensor.ShapeIsNone(),
            "tflite_shape_signature": shape_signature,
            "tflite_shape_signature_present": not tensor.ShapeSignatureIsNone(),
            "tflite_has_rank": bool(tensor.HasRank()),
            "tflite_buffer_index": buffer_index,
            "tflite_external_buffer": int(tensor.ExternalBuffer()),
            "tflite_is_variable": bool(tensor.IsVariable()),
            "tflite_tensor_table_offset": _table_position(tensor),
            "tflite_tensor_object": _stable_object(tensor_object),
            "tflite_tensor_canonical_table": _canonical_table(tensor_object),
        }
        if quantization is not None:
            quantization_object = schema.QuantizationParametersT.InitFromObj(
                quantization)
            details_type = int(quantization.DetailsType())
            metadata["tflite_quantization"] = {
                "min": tuple(float(quantization.Min(i))
                             for i in range(int(quantization.MinLength()))),
                "max": tuple(float(quantization.Max(i))
                             for i in range(int(quantization.MaxLength()))),
                "scale": tuple(float(quantization.Scale(i))
                               for i in range(int(quantization.ScaleLength()))),
                "zero_point": tuple(int(quantization.ZeroPoint(i))
                                    for i in range(int(quantization.ZeroPointLength()))),
                "quantized_dimension": int(quantization.QuantizedDimension()),
                "details_type": details_type,
                "details_type_name": _enum_name(
                    schema.QuantizationDetails, details_type),
                "table_offset": _table_position(quantization),
                "object": _stable_object(quantization_object),
                "canonical_table": _canonical_table(quantization_object),
            }
        if sparsity is not None:
            sparsity_object = schema.SparsityParametersT.InitFromObj(sparsity)
            metadata["tflite_sparsity"] = {
                "table_offset": _table_position(sparsity),
                "object": _stable_object(sparsity_object),
                "canonical_table": _canonical_table(sparsity_object),
            }

        buffer = buffers[buffer_index]
        initializer = (
            bool(buffer["data_present"] or buffer["external_offset"] or
                 buffer["external_size"])
            or bool(tensor.IsVariable())
            or (tensor_index not in public_inputs and tensor_index not in produced)
        )
        graph.add_tensor(TensorValue(
            name=tensor_name,
            shape=_normalized_shape(shape),
            dtype=_TENSOR_DTYPES.get(dtype_code, f"tflite.dtype.{dtype_code}"),
            source_dtype=f"tflite.TensorType.{dtype_name}",
            layout="unknown",
            initializer=initializer,
            public_input=tensor_index in public_inputs,
            public_output=tensor_index in public_outputs,
            data=_tensor_data_ref(tensor_name, buffer, source_name),
            metadata=metadata,
        ))

    for tensor_index in input_indices:
        tensor_name = _source_tensor_name(subgraph_index, tensor_index)
        if tensor_name not in graph.inputs:
            graph.inputs.append(tensor_name)
    for tensor_index in output_indices:
        tensor_name = _source_tensor_name(subgraph_index, tensor_index)
        if tensor_name not in graph.outputs:
            graph.outputs.append(tensor_name)

    for operator_index, operator in enumerate(operators):
        opcode_index = int(operator.OpcodeIndex())
        if opcode_index < 0 or opcode_index >= len(operator_codes):
            _fail(
                "VXTFLITE008",
                f"subgraph {subgraph_index} operator {operator_index} references "
                f"missing opcode {opcode_index}",
            )
        opcode = operator_codes[opcode_index]
        input_indices_for_op = _int_vector(operator, "Inputs")
        output_indices_for_op = _int_vector(operator, "Outputs")
        for kind, indices in (("input", input_indices_for_op),
                              ("output", output_indices_for_op)):
            for position, tensor_index in enumerate(indices):
                _validate_tensor_index(
                    tensor_index,
                    tensor_count,
                    where=(f"subgraph[{subgraph_index}].operator[{operator_index}]"
                           f".{kind}[{position}]"),
                    optional=True,
                )

        operator_object = schema.OperatorT.InitFromObj(operator)
        attributes: list[OpAttribute] = []
        option1, option1_descriptor = _option_attribute(
            schema, operator, operator_object, second=False)
        option2, option2_descriptor = _option_attribute(
            schema, operator, operator_object, second=True)
        if option1 is not None:
            attributes.append(option1)
        if option2 is not None:
            attributes.append(option2)
        custom_options = _bytes_vector(operator, "CustomOptions")
        if not operator.CustomOptionsIsNone():
            attributes.append(OpAttribute(
                name="custom_options",
                kind=_enum_name(
                    schema.CustomOptionsFormat, int(operator.CustomOptionsFormat())),
                value=custom_options,
                raw=custom_options,
            ))

        resolved_code = int(opcode["resolved_builtin_code"])
        custom_code = opcode["custom_code"]
        if resolved_code == int(schema.BuiltinOperator.CUSTOM):
            op_type = _display_text(custom_code) or "CUSTOM"
            domain = "tflite.custom"
        else:
            op_type = str(opcode["builtin_name"])
            domain = "tflite"
        node_name = (
            f"@tflite/subgraph/{subgraph_index}/operator/{operator_index}:"
            f"{op_type}"
        )
        large_offset = int(operator.LargeCustomOptionsOffset())
        large_size = int(operator.LargeCustomOptionsSize())
        large_payload: Optional[bytes] = None
        if large_size and 0 <= large_offset <= len(raw) - large_size:
            large_payload = raw[large_offset:large_offset + large_size]
        graph.add_node(OpNode(
            name=node_name,
            op_type=op_type,
            inputs=tuple(ValuePort(
                f"input_{position}",
                (_source_tensor_name(subgraph_index, tensor_index)
                 if tensor_index >= 0 else None),
                position,
            ) for position, tensor_index in enumerate(input_indices_for_op)),
            outputs=tuple(ValuePort(
                f"output_{position}",
                (_source_tensor_name(subgraph_index, tensor_index)
                 if tensor_index >= 0 else None),
                position,
            ) for position, tensor_index in enumerate(output_indices_for_op)),
            domain=domain,
            version=int(opcode["version"]),
            attributes=tuple(attributes),
            provenance=(Provenance(
                source_format="tflite",
                source_name=(f"subgraph[{subgraph_index}].operator[{operator_index}]"),
                source_op=op_type,
                location=(f"{source_name}:subgraphs[{subgraph_index}]"
                          f".operators[{operator_index}]"),
                domain=domain,
                version=int(opcode["version"]),
            ),),
            metadata={
                "tflite_operator_index": operator_index,
                "tflite_operator_table_offset": _table_position(operator),
                "tflite_opcode_index": opcode_index,
                "tflite_operator_code": opcode,
                "tflite_input_tensor_indices": input_indices_for_op,
                "tflite_output_tensor_indices": output_indices_for_op,
                "tflite_builtin_options": option1_descriptor,
                "tflite_builtin_options2": option2_descriptor,
                "tflite_custom_options_present": not operator.CustomOptionsIsNone(),
                "tflite_custom_options_format": int(operator.CustomOptionsFormat()),
                "tflite_mutating_variable_inputs": _bool_vector(
                    operator, "MutatingVariableInputs"),
                "tflite_intermediate_tensor_indices": _int_vector(
                    operator, "Intermediates"),
                "tflite_large_custom_options_offset": large_offset,
                "tflite_large_custom_options_size": large_size,
                "tflite_large_custom_options": large_payload,
                "tflite_debug_metadata_index": int(operator.DebugMetadataIndex()),
                "tflite_operator_object": _stable_object(operator_object),
                "tflite_operator_canonical_table": _canonical_table(operator_object),
            },
        ))

    graph.verify(IRDialect.SOURCE)
    return graph


def import_tflite_source(
    source: str | Path | bytes | bytearray | memoryview,
) -> GraphIR:
    """Import a TFLite/LiteRT model without performing graph rewrites.

    Subgraph zero is returned as the model entry graph.  Every other subgraph
    is available, in source index order, through :func:`tflite_subgraphs` and
    ``graph.metadata["tflite_additional_subgraphs"]``.
    """

    schema = _schema_module()
    raw, source_name = _read_source(source)
    if len(raw) < 8 or raw[4:8] != b"TFL3":
        _fail(
            "VXTFLITE003",
            f"{source_name} is not a TFLite FlatBuffer with a TFL3 identifier",
        )
    try:
        if not schema.Model.ModelBufferHasIdentifier(raw, 0):
            _fail(
                "VXTFLITE003",
                f"{source_name} is not a TFLite FlatBuffer with a TFL3 identifier",
            )
        model = schema.Model.GetRootAs(raw, 0)
        # Force a complete schema traversal up front.  Later code may otherwise
        # discover a corrupt nested table only after partially constructing IR.
        model_object = schema.ModelT.InitFromObj(model)
        model_object_type = type(model_object).__name__
        del model_object
    except ExporterError:
        raise
    except Exception as error:
        _fail("VXTFLITE004", f"cannot decode TFLite model {source_name}: {error}")

    if int(model.SubgraphsLength()) == 0:
        _fail("VXTFLITE005", "TFLite model contains no subgraphs")
    if int(model.BuffersLength()) == 0:
        _fail("VXTFLITE005", "TFLite model contains no buffer table")

    buffers = tuple(
        _buffer_descriptor(model.Buffers(index), index)
        for index in range(int(model.BuffersLength()))
    )
    operator_codes = tuple(
        _operator_code_descriptor(schema, model.OperatorCodes(index), index)
        for index in range(int(model.OperatorCodesLength()))
    )
    tensor_counts = tuple(
        int(model.Subgraphs(index).TensorsLength())
        for index in range(int(model.SubgraphsLength()))
    )
    subgraphs = tuple(
        _import_subgraph(
            schema,
            model.Subgraphs(index),
            subgraph_index=index,
            source_name=source_name,
            operator_codes=operator_codes,
            buffers=buffers,
            raw=raw,
        )
        for index in range(int(model.SubgraphsLength()))
    )
    signatures = tuple(
        _signature_descriptor(
            schema, model.SignatureDefs(index), index, tensor_counts)
        for index in range(int(model.SignatureDefsLength()))
    )

    metadata_entries: list[dict[str, Any]] = []
    for index in range(int(model.MetadataLength())):
        item = model.Metadata(index)
        buffer_index = int(item.Buffer())
        if buffer_index < 0 or buffer_index >= len(buffers):
            _fail(
                "VXTFLITE007",
                f"model metadata {index} references missing buffer {buffer_index}",
            )
        item_object = schema.MetadataT.InitFromObj(item)
        item_name = item.Name()
        metadata_entries.append({
            "index": index,
            "name": bytes(item_name) if item_name is not None else None,
            "display_name": _display_text(item_name),
            "buffer": buffer_index,
            "table_offset": _table_position(item),
            "object": _stable_object(item_object),
            "canonical_table": _canonical_table(item_object),
        })

    external_buffer_groups = tuple(
        _stable_object(schema.ExternalBufferGroupT.InitFromObj(
            model.ExternalBufferGroups(index)))
        for index in range(int(model.ExternalBufferGroupsLength()))
    )
    external_buffers = tuple(
        _stable_object(schema.ExternalBufferT.InitFromObj(
            model.ExternalBuffers(index)))
        for index in range(int(model.ExternalBuffersLength()))
    )
    description = model.Description()
    try:
        schema_version = importlib_metadata.version("ai-edge-litert")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover.
        schema_version = "unknown"

    root = subgraphs[0]
    root.source_name = source_name
    root.metadata.update({
        "tflite_model_version": int(model.Version()),
        "tflite_description": (bytes(description)
                               if description is not None else None),
        "tflite_display_description": _display_text(description),
        "tflite_model_sha256": hashlib.sha256(raw).hexdigest(),
        "tflite_model_bytes": raw,
        "tflite_schema_provider": "ai-edge-litert",
        "tflite_schema_provider_version": schema_version,
        "tflite_model_object": {
            "_type": model_object_type,
            "version": int(model.Version()),
            "description": (bytes(description)
                            if description is not None else None),
        },
        "tflite_operator_codes": operator_codes,
        "tflite_buffers": buffers,
        "tflite_metadata_buffer": _int_vector(model, "MetadataBuffer"),
        "tflite_metadata": tuple(metadata_entries),
        "tflite_signatures": signatures,
        "tflite_external_buffer_groups": external_buffer_groups,
        "tflite_external_buffers": external_buffers,
        "tflite_additional_subgraphs": subgraphs[1:],
    })
    root.verify(IRDialect.SOURCE)
    return root


def tflite_subgraphs(graph: GraphIR) -> tuple[GraphIR, ...]:
    """Return all imported subgraphs in their original source-index order."""

    if graph.source_format != "tflite":
        raise ValueError("graph was not produced by the TFLite source importer")
    additional = graph.metadata.get("tflite_additional_subgraphs", ())
    if not isinstance(additional, tuple) or not all(
            isinstance(item, GraphIR) for item in additional):
        raise ValueError("TFLite graph has malformed subgraph metadata")
    return (graph, *additional)


__all__ = ["import_tflite_source", "tflite_subgraphs"]
