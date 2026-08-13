"""Independent NumPy oracle for verified ``volvox-graph/v1`` RuntimeIR.

This module is deliberately small and strict.  It is not a fallback runtime and
must never silently approximate an operator it does not understand.  Its job is
to provide a readable semantic oracle for exporter and optimizer differential
tests, including stable captures for every node output.

Quantized values use their central :class:`AffineQuantization` tensor
references.  Quantize/requantize arithmetic mirrors VolvoxAI's portable
contract: F32 intermediate stages, round-to-nearest ties-to-even, NaN mapping to
the destination zero point, and saturation before the integer conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence

import numpy as np

from .errors import Diagnostic, ExporterError
from .ir import AffineQuantization, GraphIR, IRDialect, OpNode, TensorValue
from .quantized_embedding import embedding_ids_preflight_proof
from .shape_system import (
    PublicInputShapeContract,
    ShapeBinding,
    ShapeContractError,
    ShapedRuntimeTensorView,
    bind_public_input_shapes,
)


_NUMPY_DTYPES = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int32": np.dtype(np.int32),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
}
_RUNTIME_DTYPES_BY_NUMPY = {
    value: key for key, value in _NUMPY_DTYPES.items()
    if key != "float16"
}
_BYTE_DTYPES = frozenset({"int8", "uint8"})
_SUPPORTED_OPS = frozenset({
    "Identity",
    "Add", "Sub", "Mul", "Div",
    "MatMul", "Linear", "BatchMatMul", "QLinear", "QBatchMatMul",
    "Conv2D", "QConv2D", "Embedding", "QEmbedding",
    "CrossSDPA", "QSDPA",
    "Relu", "ReLU", "Sigmoid", "GELU", "SiLU",
    "LayerNorm", "GroupNorm",
    "QAdd", "QGELU", "QSiLU", "QLayerNorm", "QGroupNorm",
    "Reshape", "Expand", "Transpose", "Concat",
    "QuantizeLinear", "DequantizeLinear", "RequantizeLinear",
})


def _fail(code: str, message: str, node: Optional[OpNode] = None) -> NoReturn:
    raise ExporterError(Diagnostic(
        code=code,
        message=message,
        stage="reference-execute",
        source_node=node.name if node is not None else None,
        source_op=node.op_type if node is not None else None,
    ))


def _readonly_copy(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReferenceExecution:
    """One execution with stable, immutable captures.

    ``intermediates`` follows graph/node/port order and contains every node
    output, including public outputs.  ``tensors`` additionally contains model
    initializers and public inputs and is useful when diagnosing an operand.
    """

    outputs: Mapping[str, np.ndarray]
    intermediates: Mapping[str, np.ndarray]
    tensors: Mapping[str, np.ndarray]
    binding: ShapeBinding

    def tensor(self, name: str) -> np.ndarray:
        return self.tensors[name]


class ReferenceExecutor:
    """Execute the supported correctness core of a typed RuntimeIR graph."""

    def __init__(self, graph: GraphIR, initializers: Mapping[str, Any]):
        if not isinstance(graph, GraphIR):
            raise TypeError("graph must be a GraphIR")
        graph.verify(IRDialect.RUNTIME)
        self.graph = graph
        self._input_contract = PublicInputShapeContract(
            graph.shape_environment,
            tuple({
                "name": name,
                "dtype": graph.tensors[name].dtype,
                "shape": graph.tensors[name].shape,
            } for name in graph.inputs),
        )
        self._initializers = self._load_initializers(initializers)

    def bind(self, inputs: Mapping[str, Any]) -> ShapeBinding:
        """Validate one complete concrete public-input set atomically."""

        if not isinstance(inputs, Mapping):
            _fail("VXREF001", "reference inputs must be a tensor mapping")
        views: dict[str, ShapedRuntimeTensorView] = {}
        for name, value in inputs.items():
            if not isinstance(name, str):
                _fail("VXREF002", "public input names must be strings")
            array = np.asarray(value)
            dtype = _RUNTIME_DTYPES_BY_NUMPY.get(array.dtype)
            if dtype is None:
                _fail(
                    "VXREF007",
                    f"input {name!r} has unsupported dtype {array.dtype!r}",
                )
            views[name] = ShapedRuntimeTensorView(
                data=array,
                shape=tuple(int(value) for value in array.shape),
                dtype=dtype,
                byte_length=int(array.nbytes),
            )
        try:
            return bind_public_input_shapes(self._input_contract, views)
        except ShapeContractError as error:
            _fail(
                "VXREF008",
                "concrete public-input binding failed "
                f"({error.code} at {error.path}): {error.detail}",
            )

    def run(self, binding: ShapeBinding) -> ReferenceExecution:
        """Execute one already validated concrete binding."""

        if not isinstance(binding, ShapeBinding):
            _fail(
                "VXREF001",
                "reference execution requires a ShapeBinding; call bind(inputs) first",
            )
        binding = self._revalidate_binding(binding)
        bound_graph = self.graph.bind_shape_profile(dict(binding.symbols))
        bound_executor = ReferenceExecutor(bound_graph, self._initializers)
        return bound_executor._run_bound(binding)

    def _revalidate_binding(self, binding: ShapeBinding) -> ShapeBinding:
        """Reject fabricated, stale, or mutated binding data before execution."""

        views: dict[str, ShapedRuntimeTensorView] = {}
        for value in binding.inputs:
            array = np.asarray(value.data)
            dtype = _RUNTIME_DTYPES_BY_NUMPY.get(array.dtype)
            if dtype is None:
                _fail(
                    "VXREF007",
                    f"bound input {value.name!r} has unsupported dtype {array.dtype!r}",
                )
            views[value.name] = ShapedRuntimeTensorView(
                data=value.data,
                shape=tuple(int(axis) for axis in array.shape),
                dtype=dtype,
                byte_length=int(array.nbytes),
            )
        try:
            checked = bind_public_input_shapes(self._input_contract, views)
        except ShapeContractError as error:
            _fail(
                "VXREF008",
                "concrete public-input rebinding failed "
                f"({error.code} at {error.path}): {error.detail}",
            )
        supplied_metadata = tuple(
            (value.name, value.dtype, value.shape, value.element_count, value.size_bytes)
            for value in binding.inputs
        )
        checked_metadata = tuple(
            (value.name, value.dtype, value.shape, value.element_count, value.size_bytes)
            for value in checked.inputs
        )
        if (
            binding.signature != checked.signature
            or binding.symbols != checked.symbols
            or supplied_metadata != checked_metadata
            or any(
                supplied.data is not validated.data
                for supplied, validated in zip(binding.inputs, checked.inputs)
            )
        ):
            _fail("VXREF008", "concrete public-input binding is stale or fabricated")
        return checked

    def _run_bound(self, binding: ShapeBinding) -> ReferenceExecution:
        """Execute against this executor's fully concrete graph."""

        values: dict[str, np.ndarray] = {
            name: _readonly_copy(value) for name, value in self._initializers.items()
        }
        for name in self.graph.inputs:
            bound = binding.input(name)
            if bound is None:
                _fail("VXREF002", f"binding omits public input {name!r}")
            values[name] = self._validate_array(name, bound.data, role="input")

        captures: dict[str, np.ndarray] = {}
        for node in self.graph.nodes:
            if node.op_type not in _SUPPORTED_OPS:
                _fail(
                    "VXREF003",
                    f"unsupported reference operator {node.op_type!r}; refusing to approximate it",
                    node,
                )
            try:
                result = self._execute_node(node, values)
            except ExporterError:
                raise
            except (ArithmeticError, TypeError, ValueError) as error:
                _fail("VXREF004", f"{node.op_type} contract failed: {error}", node)
            output_name = self._single_output_name(node)
            stored = self._validate_result(output_name, result, node)
            values[output_name] = stored
            captures[output_name] = stored

        outputs = {
            name: _readonly_copy(values[name]) for name in self.graph.outputs
        }
        return ReferenceExecution(
            outputs=MappingProxyType(outputs),
            intermediates=MappingProxyType(dict(captures)),
            tensors=MappingProxyType(dict(values)),
            binding=binding,
        )

    def _load_initializers(self, initializers: Mapping[str, Any]) -> dict[str, np.ndarray]:
        if not isinstance(initializers, Mapping):
            _fail("VXREF005", "initializers must be the resolved safetensors mapping")
        expected = {
            name for name, tensor in self.graph.tensors.items() if tensor.initializer
        }
        supplied = set(initializers)
        if supplied != expected:
            missing = sorted(expected - supplied, key=repr)
            unexpected = sorted(supplied - expected, key=repr)
            _fail(
                "VXREF006",
                "resolved safetensors inventory differs from RuntimeIR "
                f"(missing={missing}, unexpected={unexpected})",
            )
        return {
            name: self._validate_array(name, initializers[name], role="initializer")
            for name in self.graph.tensors if name in expected
        }

    def _validate_array(self, name: str, value: Any, *, role: str) -> np.ndarray:
        tensor = self.graph.tensors[name]
        expected_dtype = _NUMPY_DTYPES.get(tensor.dtype)
        if expected_dtype is None:
            _fail("VXREF007", f"tensor {name!r} has unsupported dtype {tensor.dtype!r}")
        array = np.asarray(value)
        if any(not isinstance(dimension, int) or isinstance(dimension, bool)
               for dimension in tensor.shape):
            _fail("VXREF008", f"{role} {name!r} does not have a concrete shape")
        expected_shape = tuple(int(dimension) for dimension in tensor.shape)
        if array.dtype != expected_dtype or array.shape != expected_shape:
            _fail(
                "VXREF008",
                f"{role} {name!r} is shape={array.shape}, dtype={array.dtype}; "
                f"expected shape={expected_shape}, dtype={expected_dtype}",
            )
        return _readonly_copy(array)

    def _validate_result(self, name: str, value: Any, node: OpNode) -> np.ndarray:
        descriptor = self.graph.tensors[name]
        array = np.asarray(value)
        expected_shape = tuple(int(dimension) for dimension in descriptor.shape)
        expected_dtype = _NUMPY_DTYPES[descriptor.dtype]
        if array.shape != expected_shape or array.dtype != expected_dtype:
            _fail(
                "VXREF009",
                f"{node.op_type} produced {name!r} as shape={array.shape}, "
                f"dtype={array.dtype}; descriptor requires shape={expected_shape}, "
                f"dtype={expected_dtype}",
                node,
            )
        return _readonly_copy(array)

    @staticmethod
    def _single_output_name(node: OpNode) -> str:
        outputs = [port.value for port in node.outputs if port.value is not None]
        if len(outputs) != 1:
            _fail(
                "VXREF010",
                f"{node.op_type} reference requires exactly one output, got {len(outputs)}",
                node,
            )
        return outputs[0]

    @staticmethod
    def _params(node: OpNode, allowed: Iterable[str]) -> dict[str, Any]:
        other_attributes = [
            attribute.name for attribute in node.attributes
            if not (attribute.name == "params" and
                    attribute.kind == "volvox.params")
        ]
        if other_attributes:
            _fail(
                "VXREF011",
                f"{node.op_type} has unsupported runtime attributes {other_attributes}",
                node,
            )
        attributes = [
            attribute for attribute in node.attributes
            if attribute.name == "params" and attribute.kind == "volvox.params"
        ]
        if len(attributes) > 1 or (attributes and not isinstance(attributes[0].value, Mapping)):
            _fail("VXREF011", f"{node.op_type} has malformed runtime params", node)
        params = dict(attributes[0].value) if attributes else {}
        unknown = sorted(set(params) - set(allowed), key=repr)
        if unknown:
            _fail(
                "VXREF012",
                f"{node.op_type} has unsupported semantic params {unknown}",
                node,
            )
        return params

    @staticmethod
    def _require_ports(
        node: OpNode,
        inputs: Mapping[str, np.ndarray],
        required: Iterable[str],
        optional: Iterable[str] = (),
    ) -> None:
        required_set = set(required)
        allowed = required_set | set(optional)
        present = set(inputs)
        if not required_set <= present or not present <= allowed:
            _fail(
                "VXREF013",
                f"{node.op_type} ports are {sorted(present)}; required "
                f"{sorted(required_set)}, allowed {sorted(allowed)}",
                node,
            )

    def _node_inputs(self, node: OpNode, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            port.name: values[port.value]
            for port in node.inputs if port.value is not None
        }

    def _execute_node(
        self, node: OpNode, values: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        inputs = self._node_inputs(node, values)
        op = node.op_type
        if op == "Identity":
            return self._identity(node, inputs)
        if op in {"Add", "Sub", "Mul", "Div"}:
            return self._binary(node, inputs)
        if op == "QAdd":
            return self._qadd(node, inputs)
        if op in {"MatMul", "Linear"}:
            return self._matmul(node, inputs)
        if op == "BatchMatMul":
            return self._batch_matmul(node, inputs)
        if op == "QLinear":
            return self._qlinear(node, inputs)
        if op == "QBatchMatMul":
            return self._q_batch_matmul(node, inputs)
        if op == "Conv2D":
            return self._conv2d(node, inputs)
        if op == "QConv2D":
            return self._qconv2d(node, inputs)
        if op == "Embedding":
            return self._embedding(node, inputs)
        if op == "QEmbedding":
            return self._q_embedding(node, inputs)
        if op == "CrossSDPA":
            return self._attention(node, inputs)
        if op == "QSDPA":
            return self._qsdpa(node, inputs)
        if op in {"Relu", "ReLU", "Sigmoid"}:
            return self._activation(node, inputs)
        if op in {"GELU", "SiLU"}:
            return self._smooth_activation(node, inputs)
        if op in {"QGELU", "QSiLU"}:
            return self._q_smooth_activation(node, inputs)
        if op in {"LayerNorm", "GroupNorm"}:
            return self._normalization(node, inputs)
        if op in {"QLayerNorm", "QGroupNorm"}:
            return self._q_normalization(node, inputs)
        if op == "Reshape":
            return self._reshape(node, inputs)
        if op == "Expand":
            return self._expand(node, inputs)
        if op == "Transpose":
            return self._transpose(node, inputs)
        if op == "Concat":
            return self._concat(node, inputs)
        if op == "QuantizeLinear":
            return self._quantize(node, inputs)
        if op == "DequantizeLinear":
            return self._dequantize(node, inputs)
        if op == "RequantizeLinear":
            return self._requantize(node, inputs)
        _fail("VXREF003", f"unsupported reference operator {op!r}", node)

    def _identity(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        self._params(node, ())
        self._require_ports(node, inputs, ("input",))
        output = self.graph.tensors[self._single_output_name(node)]
        source = self.graph.tensors[node.input_map()["input"]]
        if source.dtype != output.dtype or source.shape != output.shape:
            _fail("VXREF014", "Identity requires identical input/output descriptors", node)
        if source.dtype not in {"float32", "int32", "int8", "uint8"}:
            _fail("VXREF014", f"Identity does not support {source.dtype} storage", node)
        self._require_byte_mapping_preserved(node, (source, output))
        return np.array(inputs["input"], copy=True)

    def _binary(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        params = self._params(node, ("relu",) if node.op_type == "Add" else ())
        self._require_ports(node, inputs, ("a", "b"))
        output = self.graph.tensors[self._single_output_name(node)]
        operand_names = node.input_map()
        a_desc = self.graph.tensors[operand_names["a"]]
        b_desc = self.graph.tensors[operand_names["b"]]
        if {a_desc.dtype, b_desc.dtype, output.dtype} != {"float32"}:
            _fail("VXREF015", f"{node.op_type} reference requires F32 edges", node)
        operators = {
            "Add": np.add,
            "Sub": np.subtract,
            "Mul": np.multiply,
            "Div": np.divide,
        }
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = operators[node.op_type](
                inputs["a"], inputs["b"], dtype=np.float32,
            )
        if node.op_type == "Add":
            relu = params.get("relu", 0)
            if isinstance(relu, bool) or relu not in {0, 1, 2}:
                _fail("VXREF016", "Add relu must be 0, 1, or 2", node)
            if relu:
                maximum = np.float32(6.0) if relu == 2 else np.float32(np.inf)
                result = np.clip(result, np.float32(0.0), maximum)
        return np.asarray(result, dtype=np.float32)

    def _qadd(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        params = self._params(node, ("relu",))
        self._require_ports(node, inputs, ("a", "b"))
        names = node.input_map()
        output_name = self._single_output_name(node)
        descriptors = (
            self.graph.tensors[names["a"]],
            self.graph.tensors[names["b"]],
            self.graph.tensors[output_name],
        )
        if (
            any(descriptor.dtype not in _BYTE_DTYPES for descriptor in descriptors)
            or not (descriptors[0].shape == descriptors[1].shape == descriptors[2].shape)
        ):
            _fail(
                "VXREF048",
                "QAdd requires exact-shape I8/U8 activation edges",
                node,
            )
        relu = params.get("relu", 0)
        if relu is None:
            relu = 0
        if (
            isinstance(relu, bool)
            or not isinstance(relu, int)
            or relu not in {0, 1, 2}
        ):
            _fail("VXREF049", "QAdd relu must be 0, 1, or 2", node)

        left_q, left_scales, left_zeros = self._affine(names["a"])
        right_q, right_scales, right_zeros = self._affine(names["b"])
        output_q, output_scales, output_zeros = self._affine(output_name)
        for descriptor in (left_q, right_q, output_q):
            self._require_per_tensor(descriptor, node)
        left_scale = np.float32(left_scales.reshape(-1)[0])
        right_scale = np.float32(right_scales.reshape(-1)[0])
        output_scale = np.float32(output_scales.reshape(-1)[0])
        output_zero = int(output_zeros.reshape(-1)[0])
        left = np.multiply(
            np.subtract(
                inputs["a"].astype(np.float32),
                np.float32(int(left_zeros.reshape(-1)[0])),
                dtype=np.float32,
            ),
            left_scale,
            dtype=np.float32,
        )
        right = np.multiply(
            np.subtract(
                inputs["b"].astype(np.float32),
                np.float32(int(right_zeros.reshape(-1)[0])),
                dtype=np.float32,
            ),
            right_scale,
            dtype=np.float32,
        )
        result = self._quantize_values(
            np.add(left, right, dtype=np.float32),
            output_scale,
            output_zero,
            descriptors[2].dtype,
        )
        if relu:
            result = np.maximum(result, output_zero)
            if relu == 2:
                upper = int(self._quantize_values(
                    np.asarray([6.0], dtype=np.float32),
                    output_scale,
                    output_zero,
                    descriptors[2].dtype,
                )[0])
                result = np.minimum(result, upper)
        return np.asarray(result, dtype=_NUMPY_DTYPES[descriptors[2].dtype])

    def _matmul(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        if node.op_type == "MatMul" and set(inputs) == {"a", "b"}:
            self._params(node, ())
            a_name, b_name = node.input_map()["a"], node.input_map()["b"]
            output = self.graph.tensors[self._single_output_name(node)]
            if {self.graph.tensors[a_name].dtype,
                    self.graph.tensors[b_name].dtype, output.dtype} != {"float32"}:
                _fail("VXREF017", "generic MatMul requires F32 edges", node)
            return np.asarray(np.matmul(inputs["a"], inputs["b"]), dtype=np.float32)

        self._require_ports(
            node, inputs, ("input", "weight"),
            ("bias", "weight_scale", "weight_zero_point"),
        )
        params = self._params(node, ("weight_layout", "transB"))
        layout = params.get("weight_layout")
        trans_b = params.get("transB", False)
        if isinstance(trans_b, bool):
            pass
        elif isinstance(trans_b, int) and trans_b in {0, 1}:
            trans_b = bool(trans_b)
        else:
            _fail("VXREF018", "Linear transB must be boolean or 0/1", node)
        if trans_b:
            if layout not in {None, "dout_din"}:
                _fail("VXREF018", "Linear transB conflicts with weight_layout", node)
            layout = "dout_din"
        if layout not in {"din_dout", "dout_din"}:
            _fail(
                "VXREF018",
                "Linear-style MatMul requires explicit weight_layout din_dout or dout_din",
                node,
            )
        input_name = node.input_map()["input"]
        weight_name = node.input_map()["weight"]
        output = self.graph.tensors[self._single_output_name(node)]
        input_desc = self.graph.tensors[input_name]
        weight_desc = self.graph.tensors[weight_name]
        if input_desc.dtype != "float32" or output.dtype != "float32":
            _fail("VXREF019", "Linear requires F32 activation/output edges", node)
        if inputs["weight"].ndim != 2:
            _fail("VXREF020", "Linear weight must be rank two", node)

        if weight_desc.dtype in {"float32", "float16"}:
            if (weight_desc.quantization is not None or
                    "weight_scale" in inputs or "weight_zero_point" in inputs):
                _fail("VXREF021", "floating Linear weight cannot carry quantization inputs", node)
            weight = np.asarray(inputs["weight"], dtype=np.float32)
        elif weight_desc.dtype in _BYTE_DTYPES:
            descriptor, scales, zeros = self._affine(weight_name)
            explicit = node.input_map()
            if (explicit.get("weight_scale") != descriptor.scale or
                    explicit.get("weight_zero_point") != descriptor.zero_point):
                _fail(
                    "VXREF022",
                    "Linear weight operands must match its central descriptor",
                    node,
                )
            axis = 0 if layout == "dout_din" else 1
            if descriptor.scheme == "per_axis" and self._normalized_axis(
                descriptor.axis, inputs["weight"].ndim, node,
            ) != axis:
                _fail("VXREF023", "Linear per-axis weight mapping is not per output", node)
            weight = self._dequantized_array(inputs["weight"], descriptor, scales, zeros, node)
        else:
            _fail("VXREF024", f"unsupported Linear weight dtype {weight_desc.dtype}", node)

        matrix = weight.T if layout == "dout_din" else weight
        result = np.asarray(np.matmul(inputs["input"], matrix), dtype=np.float32)
        if "bias" in inputs:
            bias_name = node.input_map()["bias"]
            bias_desc = self.graph.tensors[bias_name]
            if (bias_desc.dtype not in {"float32", "float16"} or
                    inputs["bias"].shape != (result.shape[-1],)):
                _fail("VXREF025", "Linear bias must be F32/F16 with shape [d_out]", node)
            result = np.add(result, inputs["bias"].astype(np.float32), dtype=np.float32)
        return result

    def _qlinear(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Execute the canonical W8A8 dense contract exactly."""

        self._params(node, ())
        self._require_ports(node, inputs, ("input", "weight", "bias"))
        names = node.input_map()
        output_name = self._single_output_name(node)
        input_name = names["input"]
        weight_name = names["weight"]
        input_descriptor = self.graph.tensors[input_name]
        weight_descriptor = self.graph.tensors[weight_name]
        bias_descriptor = self.graph.tensors[names["bias"]]
        output_descriptor = self.graph.tensors[output_name]
        if (
            input_descriptor.dtype not in _BYTE_DTYPES
            or output_descriptor.dtype not in _BYTE_DTYPES
            or weight_descriptor.dtype not in _BYTE_DTYPES
            or bias_descriptor.dtype != "int32"
        ):
            _fail("VXREF044", "QLinear has incompatible physical storage", node)

        input_quant, input_scales, input_zeros = self._affine(input_name)
        output_quant, output_scales, output_zeros = self._affine(output_name)
        weight_quant, weight_scales, weight_zeros = self._affine(weight_name)
        self._require_per_tensor(input_quant, node)
        self._require_per_tensor(output_quant, node)
        if (
            weight_quant.scheme != "per_axis"
            or self._normalized_axis(weight_quant.axis, 2, node) != 0
        ):
            _fail("VXREF045", "QLinear weight mapping must be per-output axis 0", node)

        d_in = int(input_descriptor.shape[-1])
        d_out = int(output_descriptor.shape[-1])
        rows = int(np.prod(input_descriptor.shape[:-1], dtype=np.int64)) or 1
        source = inputs["input"].reshape(rows, d_in)
        weight = inputs["weight"].reshape(d_out, d_in)
        bias = inputs["bias"].reshape(d_out)
        result = np.empty((rows, d_out), dtype=_NUMPY_DTYPES[output_descriptor.dtype])
        input_zero = int(input_zeros.reshape(-1)[0])
        output_zero = int(output_zeros.reshape(-1)[0])
        input_scale = np.float32(input_scales.reshape(-1)[0])
        output_scale = np.float32(output_scales.reshape(-1)[0])
        output_info = np.iinfo(_NUMPY_DTYPES[output_descriptor.dtype])

        for output_channel in range(d_out):
            weight_zero = int(weight_zeros[output_channel])
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                product_scale = np.multiply(
                    input_scale,
                    np.float32(weight_scales[output_channel]),
                    dtype=np.float32,
                )
                multiplier = np.divide(
                    product_scale, output_scale, dtype=np.float32,
                )
            if not bool(np.isfinite(multiplier) and multiplier > np.float32(0.0)):
                _fail(
                    "VXREF046",
                    "QLinear requantization multiplier is not positive finite F32",
                    node,
                )
            for row in range(rows):
                accumulator = int(bias[output_channel])
                for input_channel in range(d_in):
                    accumulator += (
                        int(source[row, input_channel]) - input_zero
                    ) * (
                        int(weight[output_channel, input_channel]) - weight_zero
                    )
                    if accumulator < -(2**31) or accumulator > 2**31 - 1:
                        _fail("VXREF047", "QLinear I32 accumulator overflow", node)
                scaled = np.multiply(
                    np.float32(accumulator), multiplier, dtype=np.float32,
                )
                transformed = np.add(
                    scaled, np.float32(output_zero), dtype=np.float32,
                )
                safe = output_zero if np.isnan(transformed) else transformed
                clipped = np.clip(
                    safe, np.float32(output_info.min), np.float32(output_info.max),
                )
                result[row, output_channel] = np.rint(clipped)
        return result.reshape(tuple(int(value) for value in output_descriptor.shape))

    def _batch_matmul(
        self, node: OpNode, inputs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        self._params(node, ())
        self._require_ports(node, inputs, ("a", "b"))
        names = node.input_map()
        output = self.graph.tensors[self._single_output_name(node)]
        if {
            self.graph.tensors[names["a"]].dtype,
            self.graph.tensors[names["b"]].dtype,
            output.dtype,
        } != {"float32"}:
            _fail("VXREF069", "BatchMatMul requires F32 operands/output", node)
        try:
            result = np.matmul(inputs["a"], inputs["b"])
        except ValueError as error:
            _fail("VXREF069", f"BatchMatMul geometry is incompatible: {error}", node)
        return np.asarray(result, dtype=np.float32)

    def _q_batch_matmul(
        self, node: OpNode, inputs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        self._params(node, ())
        self._require_ports(node, inputs, ("a", "b"))
        names = node.input_map()
        output_name = self._single_output_name(node)
        descriptors = (
            self.graph.tensors[names["a"]],
            self.graph.tensors[names["b"]],
            self.graph.tensors[output_name],
        )
        if any(item.dtype not in _BYTE_DTYPES for item in descriptors):
            _fail("VXREF070", "QBatchMatMul requires byte operands/output", node)
        affines = [self._affine(name) for name in (names["a"], names["b"], output_name)]
        for descriptor, _, _ in affines:
            self._require_per_tensor(descriptor, node)
        left_zero = int(affines[0][2].reshape(-1)[0])
        right_zero = int(affines[1][2].reshape(-1)[0])
        output_zero = int(affines[2][2].reshape(-1)[0])
        width = int(descriptors[0].shape[-1])
        maximum = (
            width
            * max(abs(np.iinfo(_NUMPY_DTYPES[descriptors[0].dtype]).min - left_zero),
                  abs(np.iinfo(_NUMPY_DTYPES[descriptors[0].dtype]).max - left_zero))
            * max(abs(np.iinfo(_NUMPY_DTYPES[descriptors[1].dtype]).min - right_zero),
                  abs(np.iinfo(_NUMPY_DTYPES[descriptors[1].dtype]).max - right_zero))
        )
        if maximum > 2**31 - 1:
            _fail("VXREF070", "QBatchMatMul may overflow I32", node)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            product = np.multiply(
                np.float32(affines[0][1].reshape(-1)[0]),
                np.float32(affines[1][1].reshape(-1)[0]),
                dtype=np.float32,
            )
            multiplier = np.divide(
                product, np.float32(affines[2][1].reshape(-1)[0]),
                dtype=np.float32,
            )
        if not bool(np.isfinite(multiplier) and multiplier > np.float32(0.0)):
            _fail("VXREF070", "QBatchMatMul multiplier is invalid", node)
        left = inputs["a"].astype(np.int64) - left_zero
        right = inputs["b"].astype(np.int64) - right_zero
        accumulators = np.matmul(left, right)
        if bool(np.any(accumulators < -(2**31))) or bool(np.any(accumulators > 2**31 - 1)):
            _fail("VXREF070", "QBatchMatMul I32 accumulator overflow", node)
        transformed = np.add(
            np.multiply(accumulators.astype(np.float32), multiplier, dtype=np.float32),
            np.float32(output_zero),
            dtype=np.float32,
        )
        limits = np.iinfo(_NUMPY_DTYPES[descriptors[2].dtype])
        safe = np.where(np.isnan(transformed), np.float32(output_zero), transformed)
        return np.rint(np.clip(safe, limits.min, limits.max)).astype(
            _NUMPY_DTYPES[descriptors[2].dtype]
        )

    def _conv_parameters(self, node: OpNode, *, weight_layouts: tuple[str, ...] = ("OHWI",)) -> tuple[
        tuple[int, int], tuple[int, int], tuple[int, int, int, int], int, int, str,
    ]:
        params = self._params(node, (
            "stride", "dilation", "groups", "pads", "padding",
            "data_layout", "weight_layout", "relu",
        ))

        def pair(value: Any, length: int, *, positive: bool) -> tuple[int, ...]:
            if (
                not isinstance(value, (list, tuple))
                or len(value) != length
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    or (item <= 0 if positive else item < 0)
                    for item in value
                )
            ):
                _fail("VXREF071", "Conv2D has invalid integer geometry", node)
            return tuple(int(item) for item in value)

        weight_layout = params.get("weight_layout", "OHWI")
        if (
            params.get("data_layout", "NHWC") != "NHWC"
            or weight_layout not in weight_layouts
        ):
            _fail(
                "VXREF071",
                "Conv2D oracle supports NHWC activations with "
                f"{'/'.join(weight_layouts)} weights",
                node,
            )
        stride = pair(params.get("stride", [1, 1]), 2, positive=True)
        dilation = pair(params.get("dilation", [1, 1]), 2, positive=True)
        padding = pair(params.get("padding", [0, 0]), 2, positive=False)
        pads = pair(
            params.get("pads", [padding[0], padding[1], padding[0], padding[1]]),
            4,
            positive=False,
        )
        groups = params.get("groups", 1)
        relu = params.get("relu", 0)
        if (
            isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0
            or isinstance(relu, bool) or not isinstance(relu, int)
            or relu not in {0, 1, 2}
        ):
            _fail("VXREF071", "Conv2D groups/relu are invalid", node)
        return stride, dilation, pads, groups, relu, weight_layout

    def _conv2d(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        self._require_ports(node, inputs, ("input", "weight"), ("bias",))
        stride, dilation, pads, groups, relu, weight_layout = self._conv_parameters(
            node, weight_layouts=("OHWI", "HWIO", "HWCM"),
        )
        names = node.input_map()
        output = self.graph.tensors[self._single_output_name(node)]
        source = self.graph.tensors[names["input"]]
        weight_desc = self.graph.tensors[names["weight"]]
        if (
            source.dtype != "float32" or output.dtype != "float32"
            or weight_desc.dtype != "float32" or source.rank != 4
            or weight_desc.rank != 4 or output.rank != 4
        ):
            _fail("VXREF072", "Conv2D requires canonical rank-4 F32 storage", node)
        weight = np.asarray(inputs["weight"], dtype=np.float32)
        # The oracle indexes OHWI. HWIO/HWCM are the image-layout compute forms the
        # runtimes consume directly; fold them back so one loop covers every layout.
        if weight_layout == "HWIO":
            weight = np.ascontiguousarray(np.transpose(weight, (3, 0, 1, 2)))
        elif weight_layout == "HWCM":
            kernel_h, kernel_w, channels, multiplier = weight.shape
            weight = np.ascontiguousarray(
                np.transpose(weight, (2, 3, 0, 1))
            ).reshape(channels * multiplier, kernel_h, kernel_w, 1)
        values = np.asarray(inputs["input"], dtype=np.float32)
        batch, input_h, input_w, input_channels = values.shape
        out_channels, kernel_h, kernel_w, input_per_group = weight.shape
        out_batch, out_h, out_w, described_channels = output.shape
        if (
            batch != out_batch or out_channels != described_channels
            or input_channels != input_per_group * groups or out_channels % groups
        ):
            _fail("VXREF072", "Conv2D grouped geometry is incompatible", node)
        bias = np.zeros((out_channels,), dtype=np.float32)
        if "bias" in inputs:
            bias = np.asarray(inputs["bias"], dtype=np.float32)
            if bias.shape != (out_channels,):
                _fail("VXREF072", "Conv2D bias shape is incompatible", node)
        result = np.empty(output.shape, dtype=np.float32)
        out_per_group = out_channels // groups
        for sample in range(batch):
            for row in range(out_h):
                for column in range(out_w):
                    for channel in range(out_channels):
                        group = channel // out_per_group
                        accumulator = np.float32(bias[channel])
                        for kh in range(kernel_h):
                            ih = row * stride[0] + kh * dilation[0] - pads[0]
                            if ih < 0 or ih >= input_h:
                                continue
                            for kw in range(kernel_w):
                                iw = column * stride[1] + kw * dilation[1] - pads[1]
                                if iw < 0 or iw >= input_w:
                                    continue
                                for inner in range(input_per_group):
                                    actual = group * input_per_group + inner
                                    accumulator = np.float32(
                                        accumulator + np.float32(
                                            values[sample, ih, iw, actual]
                                            * weight[channel, kh, kw, inner]
                                        )
                                    )
                        if relu:
                            accumulator = np.maximum(accumulator, np.float32(0.0))
                            if relu == 2:
                                accumulator = np.minimum(accumulator, np.float32(6.0))
                        result[sample, row, column, channel] = accumulator
        return result

    def _qconv2d(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        self._require_ports(node, inputs, ("input", "weight"), ("bias",))
        stride, dilation, pads, groups, relu, _ = self._conv_parameters(node)
        names = node.input_map()
        output_name = self._single_output_name(node)
        source_desc = self.graph.tensors[names["input"]]
        weight_desc = self.graph.tensors[names["weight"]]
        output_desc = self.graph.tensors[output_name]
        if (
            source_desc.dtype not in _BYTE_DTYPES
            or weight_desc.dtype not in _BYTE_DTYPES
            or output_desc.dtype not in _BYTE_DTYPES
            or source_desc.rank != 4 or weight_desc.rank != 4 or output_desc.rank != 4
        ):
            _fail("VXREF073", "QConv2D requires canonical rank-4 byte storage", node)
        input_q, input_scales, input_zeros = self._affine(names["input"])
        weight_q, weight_scales, weight_zeros = self._affine(names["weight"])
        output_q, output_scales, output_zeros = self._affine(output_name)
        self._require_per_tensor(input_q, node)
        self._require_per_tensor(output_q, node)
        if weight_q.scheme != "per_axis" or self._normalized_axis(weight_q.axis, 4, node) != 0:
            _fail("VXREF073", "QConv2D weight must be per-axis OHWI axis 0", node)
        values = inputs["input"]
        weight = inputs["weight"]
        batch, input_h, input_w, input_channels = values.shape
        out_channels, kernel_h, kernel_w, input_per_group = weight.shape
        _, out_h, out_w, described_channels = output_desc.shape
        if input_channels != input_per_group * groups or described_channels != out_channels:
            _fail("VXREF073", "QConv2D grouped geometry is incompatible", node)
        bias = np.zeros((out_channels,), dtype=np.int32)
        if "bias" in inputs:
            bias = np.asarray(inputs["bias"], dtype=np.int32)
            if bias.shape != (out_channels,):
                _fail("VXREF073", "QConv2D bias must be I32 [O]", node)
        input_zero = int(input_zeros.reshape(-1)[0])
        output_zero = int(output_zeros.reshape(-1)[0])
        input_scale = np.float32(input_scales.reshape(-1)[0])
        output_scale = np.float32(output_scales.reshape(-1)[0])
        result = np.empty(output_desc.shape, dtype=_NUMPY_DTYPES[output_desc.dtype])
        limits = np.iinfo(result.dtype)
        out_per_group = out_channels // groups
        for channel in range(out_channels):
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                multiplier = np.divide(
                    np.multiply(input_scale, np.float32(weight_scales[channel]), dtype=np.float32),
                    output_scale,
                    dtype=np.float32,
                )
            if not bool(np.isfinite(multiplier) and multiplier > np.float32(0.0)):
                _fail("VXREF073", "QConv2D multiplier is invalid", node)
            group = channel // out_per_group
            weight_zero = int(weight_zeros[channel])
            for sample in range(batch):
                for row in range(out_h):
                    for column in range(out_w):
                        accumulator = int(bias[channel])
                        for kh in range(kernel_h):
                            ih = row * stride[0] + kh * dilation[0] - pads[0]
                            if ih < 0 or ih >= input_h:
                                continue
                            for kw in range(kernel_w):
                                iw = column * stride[1] + kw * dilation[1] - pads[1]
                                if iw < 0 or iw >= input_w:
                                    continue
                                for inner in range(input_per_group):
                                    actual = group * input_per_group + inner
                                    accumulator += (
                                        int(values[sample, ih, iw, actual]) - input_zero
                                    ) * (
                                        int(weight[channel, kh, kw, inner]) - weight_zero
                                    )
                        if accumulator < -(2**31) or accumulator > 2**31 - 1:
                            _fail("VXREF073", "QConv2D I32 accumulator overflow", node)
                        transformed = np.float32(
                            np.float32(np.float32(accumulator) * multiplier)
                            + np.float32(output_zero)
                        )
                        safe = output_zero if np.isnan(transformed) else transformed
                        quantized = int(np.rint(np.clip(safe, limits.min, limits.max)))
                        if relu:
                            quantized = max(quantized, output_zero)
                            if relu == 2:
                                six = int(self._quantize_values(
                                    np.asarray([6.0], dtype=np.float32), output_scale,
                                    output_zero, output_desc.dtype,
                                )[0])
                                quantized = min(quantized, six)
                        result[sample, row, column, channel] = quantized
        return result

    def _embedding(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        self._params(node, ())
        self._require_ports(node, inputs, ("input", "weight"))
        names = node.input_map()
        ids_desc = self.graph.tensors[names["input"]]
        weight_desc = self.graph.tensors[names["weight"]]
        output = self.graph.tensors[self._single_output_name(node)]
        if (
            ids_desc.dtype != "int32" or ids_desc.rank < 1
            or weight_desc.dtype != "float32" or weight_desc.rank != 2
            or output.dtype != "float32"
            or output.shape != (*ids_desc.shape, weight_desc.shape[1])
        ):
            _fail("VXREF074", "Embedding storage/geometry is incompatible", node)
        ids = inputs["input"]
        vocab = int(weight_desc.shape[0])
        if bool(np.any(ids < 0)) or bool(np.any(ids >= vocab)):
            _fail("VXREF074", "Embedding token ID is outside the vocabulary", node)
        return np.asarray(inputs["weight"][ids], dtype=np.float32)

    def _q_embedding(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        self._params(node, ())
        self._require_ports(node, inputs, ("input", "weight"))
        names = node.input_map()
        output_name = self._single_output_name(node)
        ids_desc = self.graph.tensors[names["input"]]
        weight_desc = self.graph.tensors[names["weight"]]
        output_desc = self.graph.tensors[output_name]
        if (
            ids_desc.dtype != "int32"
            or weight_desc.dtype not in _BYTE_DTYPES or weight_desc.rank != 2
            or output_desc.dtype not in _BYTE_DTYPES
            or output_desc.shape != (*ids_desc.shape, weight_desc.shape[1])
        ):
            _fail("VXREF075", "QEmbedding storage/geometry is incompatible", node)
        if embedding_ids_preflight_proof(
            self.graph, names["input"], int(weight_desc.shape[0]),
        ) is None:
            _fail(
                "VXREF075",
                "QEmbedding IDs are not public or bounded by a canonical Clip",
                node,
            )
        weight_q, row_scales, row_zeros = self._affine(names["weight"])
        output_q, output_scales, output_zeros = self._affine(output_name)
        if weight_q.scheme != "per_axis" or self._normalized_axis(weight_q.axis, 2, node) != 0:
            _fail("VXREF075", "QEmbedding weight must be per-row", node)
        self._require_per_tensor(output_q, node)
        ids = inputs["input"]
        vocab, hidden = weight_desc.shape
        if bool(np.any(ids < 0)) or bool(np.any(ids >= vocab)):
            _fail("VXREF075", "QEmbedding token ID is outside the vocabulary", node)
        output = np.empty(output_desc.shape, dtype=_NUMPY_DTYPES[output_desc.dtype])
        flat_ids = ids.reshape(-1)
        flat_output = output.reshape(-1, hidden)
        output_scale = np.float32(output_scales.reshape(-1)[0])
        output_zero = int(output_zeros.reshape(-1)[0])
        for index, token in enumerate(flat_ids):
            row = int(token)
            dequantized = np.multiply(
                inputs["weight"][row].astype(np.float32) - np.float32(int(row_zeros[row])),
                np.float32(row_scales[row]),
                dtype=np.float32,
            )
            flat_output[index] = self._quantize_values(
                dequantized, output_scale, output_zero, output_desc.dtype,
            )
        return output

    @staticmethod
    def _keep_mask_value(
        mask: np.ndarray,
        batch_index: int,
        query: int,
        key: int,
        *,
        batch: int,
        queries: int,
        keys: int,
    ) -> bool:
        if mask.ndim == 1 and mask.shape == (keys,):
            return bool(mask[key] != 0)
        if mask.ndim == 2 and mask.shape == (batch, keys):
            return bool(mask[batch_index, key] != 0)
        if mask.ndim == 2 and mask.shape == (queries, keys):
            return bool(mask[query, key] != 0)
        if mask.ndim == 3 and mask.shape == (batch, queries, keys):
            return bool(mask[batch_index, query, key] != 0)
        raise ValueError("attention keep mask has unsupported geometry")

    def _attention(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        params = self._params(node, ("heads", "causal", "scale"))
        self._require_ports(node, inputs, ("q", "k", "v"), ("mask",))
        names = node.input_map()
        output_desc = self.graph.tensors[self._single_output_name(node)]
        descriptors = [
            self.graph.tensors[names[name]] for name in ("q", "k", "v")
        ]
        if any(item.dtype != "float32" for item in descriptors) or output_desc.dtype != "float32":
            _fail("VXREF076", f"{node.op_type} requires F32 Q/K/V/output", node)
        q = np.asarray(inputs["q"], dtype=np.float32)
        k = np.asarray(inputs["k"], dtype=np.float32)
        v = np.asarray(inputs["v"], dtype=np.float32)
        if (
            q.ndim not in {2, 3} or k.ndim != q.ndim or v.ndim != q.ndim
            or q.shape[:-2] != k.shape[:-2] or k.shape != v.shape
            or q.shape[-1] != k.shape[-1] or output_desc.shape != q.shape
        ):
            _fail("VXREF076", f"{node.op_type} Q/K/V geometry is incompatible", node)
        heads = params.get("heads")
        causal = params.get("causal")
        d_model = int(q.shape[-1])
        if (
            isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0
            or d_model % heads or not isinstance(causal, bool)
        ):
            _fail("VXREF076", f"{node.op_type} heads/causal are invalid", node)
        head_dim = d_model // heads
        scale_source = params.get("scale")
        scale = np.float32(
            1.0 / math.sqrt(head_dim) if scale_source is None else scale_source
        )
        if not bool(np.isfinite(scale) and scale > np.float32(0.0)):
            _fail("VXREF076", f"{node.op_type} scale is invalid", node)
        q3 = q[None, ...] if q.ndim == 2 else q
        k3 = k[None, ...] if k.ndim == 2 else k
        v3 = v[None, ...] if v.ndim == 2 else v
        batch, queries, _ = q3.shape
        keys = int(k3.shape[1])
        result = np.zeros_like(q3, dtype=np.float32)
        mask = inputs.get("mask")
        keep: np.ndarray | None = None
        if mask is not None:
            if mask.dtype != np.int32:
                _fail("VXREF077", "attention keep mask must be I32", node)
            keep = mask
        for batch_index in range(batch):
            for head in range(heads):
                start = head * head_dim
                stop = start + head_dim
                for query in range(queries):
                    scores: list[tuple[int, np.float32]] = []
                    for key in range(keys):
                        if causal and key > query:
                            continue
                        if keep is not None and not self._keep_mask_value(
                            keep, batch_index, query, key,
                            batch=batch, queries=queries, keys=keys,
                        ):
                            continue
                        dot = np.float32(0.0)
                        for channel in range(start, stop):
                            dot = np.float32(
                                dot + np.float32(
                                    q3[batch_index, query, channel]
                                    * k3[batch_index, key, channel]
                                )
                            )
                        score = np.float32(dot * scale)
                        scores.append((key, score))
                    if not scores:
                        continue
                    maximum = max(float(score) for _, score in scores)
                    weights = [
                        np.float32(math.exp(float(np.float32(score - maximum))))
                        for _, score in scores
                    ]
                    denominator = np.float32(sum(float(value) for value in weights))
                    for channel in range(start, stop):
                        total = np.float32(0.0)
                        for (key, _), weight_value in zip(scores, weights):
                            total = np.float32(
                                total + np.float32(
                                    weight_value * v3[batch_index, key, channel]
                                )
                            )
                        result[batch_index, query, channel] = np.float32(
                            total / denominator
                        )
        return result[0] if q.ndim == 2 else result

    def _qsdpa(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        params = self._params(node, ("heads", "causal", "scale"))
        self._require_ports(node, inputs, ("q", "k", "v"), ("mask",))
        names = node.input_map()
        output_name = self._single_output_name(node)
        descriptors = {
            name: self.graph.tensors[names[name]] for name in ("q", "k", "v")
        }
        output_desc = self.graph.tensors[output_name]
        if any(item.dtype not in _BYTE_DTYPES for item in descriptors.values()) or output_desc.dtype not in _BYTE_DTYPES:
            _fail("VXREF078", "QSDPA requires byte Q/K/V/output", node)
        q = inputs["q"]
        k = inputs["k"]
        v = inputs["v"]
        if (
            q.ndim not in {2, 3} or k.ndim != q.ndim or v.ndim != q.ndim
            or q.shape[:-2] != k.shape[:-2] or k.shape != v.shape
            or q.shape[-1] != k.shape[-1] or output_desc.shape != q.shape
        ):
            _fail("VXREF078", "QSDPA Q/K/V geometry is incompatible", node)
        heads = params.get("heads")
        causal = params.get("causal")
        d_model = int(q.shape[-1])
        if (
            isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0
            or not isinstance(causal, bool) or d_model % heads
        ):
            _fail("VXREF078", "QSDPA heads/causal are invalid", node)
        head_dim = d_model // heads
        if d_model % 4 or head_dim % 4 or head_dim > 64:
            _fail("VXREF078", "QSDPA head geometry is unsupported", node)
        affines = {
            name: self._affine(names[name]) for name in ("q", "k", "v")
        }
        output_affine = self._affine(output_name)
        for descriptor, _, _ in (*affines.values(), output_affine):
            self._require_per_tensor(descriptor, node)
        q_scale = np.float32(affines["q"][1].reshape(-1)[0])
        k_scale = np.float32(affines["k"][1].reshape(-1)[0])
        v_scale = np.float32(affines["v"][1].reshape(-1)[0])
        output_scale = np.float32(output_affine[1].reshape(-1)[0])
        q_zero = int(affines["q"][2].reshape(-1)[0])
        k_zero = int(affines["k"][2].reshape(-1)[0])
        v_zero = int(affines["v"][2].reshape(-1)[0])
        output_zero = int(output_affine[2].reshape(-1)[0])
        attention_scale = np.float32(
            1.0 / math.sqrt(head_dim)
            if params.get("scale") is None else params["scale"]
        )
        score_multiplier = np.multiply(
            np.multiply(q_scale, k_scale, dtype=np.float32),
            attention_scale,
            dtype=np.float32,
        )
        if not bool(np.isfinite(score_multiplier) and score_multiplier > np.float32(0.0)):
            _fail("VXREF078", "QSDPA score multiplier is invalid", node)
        q3 = q[None, ...] if q.ndim == 2 else q
        k3 = k[None, ...] if k.ndim == 2 else k
        v3 = v[None, ...] if v.ndim == 2 else v
        batch, queries, _ = q3.shape
        keys = int(k3.shape[1])
        keep = inputs.get("mask")
        if keep is not None and keep.dtype != np.int32:
            _fail("VXREF079", "QSDPA keep mask must be I32", node)
        result = np.empty(q3.shape, dtype=_NUMPY_DTYPES[output_desc.dtype])
        limits = np.iinfo(result.dtype)
        for batch_index in range(batch):
            for head in range(heads):
                start = head * head_dim
                stop = start + head_dim
                for query in range(queries):
                    accumulators = np.zeros((head_dim,), dtype=np.float32)
                    running_max = -math.inf
                    running_sum = np.float32(0.0)
                    visible = False
                    for key in range(keys):
                        if causal and key > query:
                            continue
                        if keep is not None and not self._keep_mask_value(
                            keep, batch_index, query, key,
                            batch=batch, queries=queries, keys=keys,
                        ):
                            continue
                        raw_dot = 0
                        for channel in range(start, stop):
                            raw_dot += (
                                int(q3[batch_index, query, channel]) - q_zero
                            ) * (
                                int(k3[batch_index, key, channel]) - k_zero
                            )
                        score = np.float32(np.float32(raw_dot) * score_multiplier)
                        row = np.multiply(
                            v3[batch_index, key, start:stop].astype(np.float32)
                            - np.float32(v_zero),
                            v_scale,
                            dtype=np.float32,
                        )
                        if not visible:
                            running_max = float(score)
                            running_sum = np.float32(1.0)
                            accumulators[:] = row
                            visible = True
                        elif float(score) > running_max:
                            previous = np.float32(math.exp(float(np.float32(running_max - float(score)))))
                            running_sum = np.float32(np.float32(running_sum * previous) + np.float32(1.0))
                            accumulators[:] = np.add(
                                np.multiply(accumulators, previous, dtype=np.float32),
                                row,
                                dtype=np.float32,
                            )
                            running_max = float(score)
                        else:
                            current = (
                                np.float32(1.0) if float(score) == running_max
                                else np.float32(math.exp(float(np.float32(float(score) - running_max))))
                            )
                            running_sum = np.float32(running_sum + current)
                            accumulators[:] = np.add(
                                accumulators,
                                np.multiply(row, current, dtype=np.float32),
                                dtype=np.float32,
                            )
                    if not visible or running_sum == np.float32(0.0):
                        result[batch_index, query, start:stop] = output_zero
                        continue
                    real = np.divide(accumulators, running_sum, dtype=np.float32)
                    transformed = np.add(
                        np.divide(real, output_scale, dtype=np.float32),
                        np.float32(output_zero),
                        dtype=np.float32,
                    )
                    safe = np.where(np.isnan(transformed), np.float32(output_zero), transformed)
                    result[batch_index, query, start:stop] = np.rint(
                        np.clip(safe, limits.min, limits.max)
                    ).astype(result.dtype)
        return result[0] if q.ndim == 2 else result

    def _activation(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        self._params(node, ())
        self._require_ports(node, inputs, ("input",))
        input_desc = self.graph.tensors[node.input_map()["input"]]
        output = self.graph.tensors[self._single_output_name(node)]
        if input_desc.dtype != "float32" or output.dtype != "float32":
            _fail("VXREF026", f"{node.op_type} requires F32 edges", node)
        value = inputs["input"]
        if node.op_type in {"Relu", "ReLU"}:
            return np.where(
                value > np.float32(0.0), value, np.float32(0.0),
            ).astype(np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.asarray(
                np.float32(1.0) /
                (np.float32(1.0) + np.exp(-value, dtype=np.float32)),
                dtype=np.float32,
            )

    def _smooth_activation(
        self, node: OpNode, inputs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        allowed = ("approximate",) if node.op_type == "GELU" else ()
        params = self._params(node, allowed)
        self._require_ports(node, inputs, ("input",))
        source = self.graph.tensors[node.input_map()["input"]]
        output = self.graph.tensors[self._single_output_name(node)]
        if (
            source.dtype != "float32"
            or output.dtype != "float32"
            or source.shape != output.shape
        ):
            _fail(
                "VXREF050",
                f"{node.op_type} requires equal-shape F32 edges",
                node,
            )
        values = np.asarray(inputs["input"], dtype=np.float32)
        if node.op_type == "SiLU":
            with np.errstate(over="ignore", invalid="ignore"):
                denominator = np.add(
                    np.float32(1.0),
                    np.exp(np.negative(values, dtype=np.float32), dtype=np.float32),
                    dtype=np.float32,
                )
                return np.divide(values, denominator, dtype=np.float32)

        approximate = params.get("approximate", "none")
        if approximate not in {"none", "tanh"}:
            _fail(
                "VXREF051",
                "GELU approximate must be 'none' or 'tanh'",
                node,
            )
        result = np.empty_like(values)
        source_flat = values.reshape(-1)
        output_flat = result.reshape(-1)
        for index, raw in enumerate(source_flat):
            value = float(np.float32(raw))
            if approximate == "none":
                activated = 0.5 * value * (
                    1.0 + math.erf(value / math.sqrt(2.0))
                )
            else:
                activated = 0.5 * value * (
                    1.0 + math.tanh(
                        math.sqrt(2.0 / math.pi) *
                        (value + 0.044715 * value * value * value)
                    )
                )
            output_flat[index] = np.float32(activated)
        return result

    def _normalization(
        self, node: OpNode, inputs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        allowed = (
            ("eps", "d_model")
            if node.op_type == "LayerNorm"
            else ("num_groups", "eps", "data_layout")
        )
        params = self._params(node, allowed)
        self._require_ports(node, inputs, ("input", "weight"), ("bias",))
        names = node.input_map()
        source = self.graph.tensors[names["input"]]
        weight = self.graph.tensors[names["weight"]]
        output = self.graph.tensors[self._single_output_name(node)]
        if (
            source.dtype != "float32"
            or output.dtype != "float32"
            or source.shape != output.shape
            or source.rank < 1
        ):
            _fail(
                "VXREF052",
                f"{node.op_type} requires equal-shape ranked F32 activations",
                node,
            )
        channels = int(source.shape[-1])
        if weight.dtype != "float32" or weight.shape != (channels,):
            _fail(
                "VXREF053",
                f"{node.op_type} weight must be F32 [{channels}]",
                node,
            )
        bias_values = np.zeros((channels,), dtype=np.float32)
        if "bias" in inputs:
            bias = self.graph.tensors[names["bias"]]
            if bias.dtype != "float32" or bias.shape != (channels,):
                _fail(
                    "VXREF053",
                    f"{node.op_type} bias must be F32 [{channels}]",
                    node,
                )
            bias_values = np.asarray(inputs["bias"], dtype=np.float32)
        gamma = np.asarray(inputs["weight"], dtype=np.float32)
        if not bool(np.all(np.isfinite(gamma)) and np.all(np.isfinite(bias_values))):
            _fail("VXREF054", f"{node.op_type} affine data must be finite", node)
        epsilon_source = params.get("eps", 1e-5)
        if isinstance(epsilon_source, bool) or not isinstance(
            epsilon_source, (int, float)
        ):
            _fail("VXREF055", f"{node.op_type} eps must be positive F32", node)
        with np.errstate(over="ignore", invalid="ignore"):
            epsilon = np.float32(epsilon_source)
        if not bool(np.isfinite(epsilon) and epsilon > np.float32(0.0)):
            _fail("VXREF055", f"{node.op_type} eps must be positive F32", node)

        values = np.asarray(inputs["input"], dtype=np.float32)
        if node.op_type == "LayerNorm":
            declared = params.get("d_model", channels)
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int)
                or declared != channels
            ):
                _fail("VXREF056", "LayerNorm d_model must match D", node)
            rows = values.reshape(-1, channels)
            result = np.empty_like(rows)
            for row_index, row in enumerate(rows):
                mean = np.mean(row, dtype=np.float32)
                centered = np.subtract(row, mean, dtype=np.float32)
                variance = np.mean(
                    np.multiply(centered, centered, dtype=np.float32),
                    dtype=np.float32,
                )
                inverse = np.divide(
                    np.float32(1.0),
                    np.sqrt(np.add(variance, epsilon, dtype=np.float32), dtype=np.float32),
                    dtype=np.float32,
                )
                normalized = np.multiply(centered, inverse, dtype=np.float32)
                result[row_index] = np.add(
                    np.multiply(normalized, gamma, dtype=np.float32),
                    bias_values,
                    dtype=np.float32,
                )
            return result.reshape(values.shape)

        if source.rank != 4 or params.get("data_layout", "NHWC") not in {None, "NHWC"}:
            _fail("VXREF057", "GroupNorm requires rank-4 NHWC activation", node)
        groups = params.get("num_groups")
        if (
            isinstance(groups, bool)
            or not isinstance(groups, int)
            or groups <= 0
            or channels % groups
        ):
            _fail("VXREF058", "GroupNorm has an invalid num_groups", node)
        batch, height, width, _ = values.shape
        channels_per_group = channels // groups
        result = np.empty_like(values)
        for sample in range(batch):
            for group in range(groups):
                start = group * channels_per_group
                stop = start + channels_per_group
                region = values[sample, :, :, start:stop]
                mean = np.mean(region, dtype=np.float32)
                centered = np.subtract(region, mean, dtype=np.float32)
                variance = np.mean(
                    np.multiply(centered, centered, dtype=np.float32),
                    dtype=np.float32,
                )
                inverse = np.divide(
                    np.float32(1.0),
                    np.sqrt(np.add(variance, epsilon, dtype=np.float32), dtype=np.float32),
                    dtype=np.float32,
                )
                normalized = np.multiply(centered, inverse, dtype=np.float32)
                result[sample, :, :, start:stop] = np.add(
                    np.multiply(
                        normalized,
                        gamma[start:stop].reshape(1, 1, channels_per_group),
                        dtype=np.float32,
                    ),
                    bias_values[start:stop].reshape(1, 1, channels_per_group),
                    dtype=np.float32,
                )
        return result

    @staticmethod
    def _qgelu_erf_approx(value: np.float32) -> np.float32:
        sign = np.float32(1.0 if value >= np.float32(0.0) else -1.0)
        magnitude = np.float32(abs(value))
        denominator = np.float32(
            np.float32(1.0) + np.float32(np.float32(0.3275911) * magnitude)
        )
        ratio = np.float32(np.float32(1.0) / denominator)
        polynomial = np.float32(np.float32(1.061405429) * ratio)
        polynomial = np.float32(polynomial - np.float32(1.453152027))
        polynomial = np.float32(polynomial * ratio)
        polynomial = np.float32(polynomial + np.float32(1.421413741))
        polynomial = np.float32(polynomial * ratio)
        polynomial = np.float32(polynomial - np.float32(0.284496736))
        polynomial = np.float32(polynomial * ratio)
        polynomial = np.float32(polynomial + np.float32(0.254829592))
        polynomial = np.float32(polynomial * ratio)
        squared = np.float32(magnitude * magnitude)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            tail = np.float32(np.exp(np.float32(-squared)))
        return np.float32(
            sign * np.float32(np.float32(1.0) - np.float32(polynomial * tail))
        )

    @classmethod
    def _qgelu_value(cls, value: np.float32) -> np.float32:
        source = np.float32(value)
        erf_input = np.float32(source * np.float32(0.7071067811865476))
        cdf = np.float32(
            np.float32(0.5) * np.float32(
                np.float32(1.0) + cls._qgelu_erf_approx(erf_input)
            )
        )
        return np.float32(source * cdf)

    def _q_smooth_activation(
        self, node: OpNode, inputs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        allowed = ("approximate",) if node.op_type == "QGELU" else ()
        params = self._params(node, allowed)
        self._require_ports(node, inputs, ("input",))
        input_name = node.input_map()["input"]
        output_name = self._single_output_name(node)
        source = self.graph.tensors[input_name]
        output = self.graph.tensors[output_name]
        if (
            source.dtype not in _BYTE_DTYPES
            or output.dtype not in _BYTE_DTYPES
            or source.shape != output.shape
            or input_name == output_name
        ):
            _fail(
                "VXREF059",
                f"{node.op_type} requires distinct equal-shape I8/U8 edges",
                node,
            )
        if node.op_type == "QGELU" and params.get("approximate", "none") != "none":
            _fail(
                "VXREF060",
                "QGELU accepts only omitted or approximate='none' parameters",
                node,
            )
        input_q, input_scales, input_zeros = self._affine(input_name)
        output_q, output_scales, output_zeros = self._affine(output_name)
        self._require_per_tensor(input_q, node)
        self._require_per_tensor(output_q, node)
        input_scale = np.float32(input_scales.reshape(-1)[0])
        input_zero = int(input_zeros.reshape(-1)[0])
        output_scale = np.float32(output_scales.reshape(-1)[0])
        output_zero = int(output_zeros.reshape(-1)[0])
        activated = np.empty(source.shape, dtype=np.float32)
        source_flat = inputs["input"].reshape(-1)
        activated_flat = activated.reshape(-1)
        for index, raw in enumerate(source_flat):
            value = np.float32(np.float32(int(raw) - input_zero) * input_scale)
            if node.op_type == "QGELU":
                activated_flat[index] = self._qgelu_value(value)
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    denominator = np.float32(
                        np.float32(1.0) + np.float32(np.exp(np.float32(-value)))
                    )
                activated_flat[index] = np.float32(value / denominator)
        return self._quantize_values(
            activated, output_scale, output_zero, output.dtype,
        )

    def _q_normalization(
        self, node: OpNode, inputs: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        allowed = (
            ("eps", "d_model")
            if node.op_type == "QLayerNorm"
            else ("num_groups", "eps", "data_layout")
        )
        params = self._params(node, allowed)
        self._require_ports(node, inputs, ("input", "weight", "bias"))
        names = node.input_map()
        input_name = names["input"]
        output_name = self._single_output_name(node)
        source = self.graph.tensors[input_name]
        output = self.graph.tensors[output_name]
        if (
            source.dtype not in _BYTE_DTYPES
            or output.dtype not in _BYTE_DTYPES
            or source.shape != output.shape
            or input_name == output_name
            or source.rank < 1
        ):
            _fail(
                "VXREF061",
                f"{node.op_type} requires distinct equal-shape ranked I8/U8 edges",
                node,
            )
        channels = int(source.shape[-1])
        weight = self.graph.tensors[names["weight"]]
        bias = self.graph.tensors[names["bias"]]
        if (
            weight.dtype != "float32"
            or bias.dtype != "float32"
            or weight.shape != (channels,)
            or bias.shape != (channels,)
            or not weight.initializer
            or not bias.initializer
        ):
            _fail(
                "VXREF062",
                f"{node.op_type} requires immutable F32 weight/bias [{channels}]",
                node,
            )
        gamma = np.asarray(inputs["weight"], dtype=np.float32)
        beta = np.asarray(inputs["bias"], dtype=np.float32)
        if not bool(np.all(np.isfinite(gamma)) and np.all(np.isfinite(beta))):
            _fail("VXREF063", f"{node.op_type} affine data must be finite", node)
        epsilon_source = params.get("eps", 1e-5)
        if epsilon_source is None:
            epsilon_source = 1e-5
        if isinstance(epsilon_source, bool) or not isinstance(
            epsilon_source, (int, float)
        ):
            _fail("VXREF064", f"{node.op_type} eps must be positive F32", node)
        with np.errstate(over="ignore", invalid="ignore"):
            epsilon = np.float32(epsilon_source)
        if not bool(np.isfinite(epsilon) and epsilon > np.float32(0.0)):
            _fail("VXREF064", f"{node.op_type} eps must be positive F32", node)

        input_q, input_scales, input_zeros = self._affine(input_name)
        output_q, output_scales, output_zeros = self._affine(output_name)
        self._require_per_tensor(input_q, node)
        self._require_per_tensor(output_q, node)
        input_scale = np.float32(input_scales.reshape(-1)[0])
        input_zero = int(input_zeros.reshape(-1)[0])
        output_scale = np.float32(output_scales.reshape(-1)[0])
        output_zero = int(output_zeros.reshape(-1)[0])
        real_output = np.empty(source.shape, dtype=np.float32)

        if node.op_type == "QLayerNorm":
            if not 1 <= source.rank <= 8:
                _fail("VXREF065", "QLayerNorm requires rank 1..8", node)
            declared = params.get("d_model", channels)
            if declared is None:
                declared = channels
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int)
                or declared != channels
            ):
                _fail("VXREF066", "QLayerNorm d_model must match D", node)
            source_rows = inputs["input"].reshape(-1, channels)
            output_rows = real_output.reshape(-1, channels)
            for row_index, row in enumerate(source_rows):
                total = np.float32(0.0)
                for raw in row:
                    total = np.float32(total + np.float32(int(raw) - input_zero))
                mean_raw = np.float32(total / np.float32(channels))
                square_total = np.float32(0.0)
                for raw in row:
                    centered = np.float32(
                        np.float32(int(raw) - input_zero) - mean_raw
                    )
                    square_total = np.float32(
                        square_total + np.float32(centered * centered)
                    )
                raw_variance = np.float32(square_total / np.float32(channels))
                raw_variance = np.maximum(raw_variance, np.float32(0.0))
                scaled_variance = np.float32(raw_variance * input_scale)
                real_variance = np.maximum(
                    np.float32(scaled_variance * input_scale), np.float32(0.0),
                )
                inverse = np.float32(
                    np.float32(1.0) /
                    np.float32(np.sqrt(np.float32(real_variance + epsilon)))
                )
                for channel, raw in enumerate(row):
                    centered = np.float32(
                        np.float32(int(raw) - input_zero) - mean_raw
                    )
                    normalized = np.float32(
                        np.float32(centered * input_scale) * inverse
                    )
                    output_rows[row_index, channel] = np.float32(
                        np.float32(normalized * gamma[channel]) + beta[channel]
                    )
        else:
            if source.rank != 4 or params.get("data_layout", "NHWC") not in {
                None, "NHWC",
            }:
                _fail("VXREF067", "QGroupNorm requires rank-4 NHWC", node)
            groups = params.get("num_groups")
            if (
                isinstance(groups, bool)
                or not isinstance(groups, int)
                or groups <= 0
                or channels % groups
            ):
                _fail("VXREF068", "QGroupNorm has invalid num_groups", node)
            batch, height, width, _ = source.shape
            channels_per_group = channels // groups
            values_per_group = height * width * channels_per_group
            source_values = inputs["input"]
            for sample in range(batch):
                for group in range(groups):
                    start = group * channels_per_group
                    stop = start + channels_per_group
                    total = np.float32(0.0)
                    for row in range(height):
                        for column in range(width):
                            for channel in range(start, stop):
                                total = np.float32(
                                    total + np.float32(
                                        int(source_values[sample, row, column, channel])
                                        - input_zero
                                    )
                                )
                    mean_raw = np.float32(total / np.float32(values_per_group))
                    square_total = np.float32(0.0)
                    for row in range(height):
                        for column in range(width):
                            for channel in range(start, stop):
                                centered = np.float32(
                                    np.float32(
                                        int(source_values[sample, row, column, channel])
                                        - input_zero
                                    ) - mean_raw
                                )
                                square_total = np.float32(
                                    square_total + np.float32(centered * centered)
                                )
                    raw_variance = np.float32(
                        square_total / np.float32(values_per_group)
                    )
                    raw_variance = np.maximum(raw_variance, np.float32(0.0))
                    scaled_variance = np.float32(raw_variance * input_scale)
                    real_variance = np.maximum(
                        np.float32(scaled_variance * input_scale), np.float32(0.0),
                    )
                    inverse = np.float32(
                        np.float32(1.0) /
                        np.float32(np.sqrt(np.float32(real_variance + epsilon)))
                    )
                    for row in range(height):
                        for column in range(width):
                            for channel in range(start, stop):
                                centered = np.float32(
                                    np.float32(
                                        int(source_values[sample, row, column, channel])
                                        - input_zero
                                    ) - mean_raw
                                )
                                normalized = np.float32(
                                    np.float32(centered * input_scale) * inverse
                                )
                                real_output[sample, row, column, channel] = np.float32(
                                    np.float32(normalized * gamma[channel]) + beta[channel]
                                )

        return self._quantize_values(
            real_output, output_scale, output_zero, output.dtype,
        )

    def _reshape(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        params = self._params(node, ("shape",))
        self._require_ports(node, inputs, ("input",))
        source = self.graph.tensors[node.input_map()["input"]]
        output = self.graph.tensors[self._single_output_name(node)]
        if source.dtype != output.dtype:
            _fail("VXREF027", "Reshape requires the same input/output dtype", node)
        if source.dtype not in {"float32", "int32", "int8", "uint8"}:
            _fail("VXREF027", f"Reshape does not support {source.dtype} storage", node)
        self._require_byte_mapping_preserved(node, (source, output))
        if "shape" in params and tuple(params["shape"]) != output.shape:
            _fail("VXREF027", "Reshape params.shape must match its output descriptor", node)
        return np.reshape(inputs["input"], tuple(int(d) for d in output.shape)).copy()

    def _expand(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        params = self._params(node, ("shape",))
        self._require_ports(node, inputs, ("input",))
        source = self.graph.tensors[node.input_map()["input"]]
        output = self.graph.tensors[self._single_output_name(node)]
        if source.dtype != output.dtype:
            _fail("VXREF071", "Expand requires the same input/output dtype", node)
        if source.dtype not in {"float32", "int32", "int8", "uint8"}:
            _fail("VXREF071", f"Expand does not support {source.dtype} storage", node)
        self._require_byte_mapping_preserved(node, (source, output))
        if not 1 <= source.rank <= output.rank <= 8:
            _fail("VXREF071", "Expand requires rank-1..8 input/output tensors", node)
        offset = output.rank - source.rank
        target = tuple(int(value) for value in output.shape)
        if "shape" in params and tuple(params["shape"]) != output.shape:
            _fail("VXREF071", "Expand params.shape must match its output descriptor", node)
        for output_axis, output_dimension in enumerate(target):
            source_dimension = (
                1 if output_axis < offset
                else source.shape[output_axis - offset]
            )
            if source_dimension not in {1, output_dimension}:
                _fail("VXREF071", "Expand output is not the exact broadcast target", node)
        try:
            return np.broadcast_to(inputs["input"], target).copy()
        except ValueError as error:
            _fail("VXREF071", f"Expand broadcast failed: {error}", node)

    def _transpose(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        params = self._params(node, ("perm",))
        self._require_ports(node, inputs, ("input",))
        source = self.graph.tensors[node.input_map()["input"]]
        output = self.graph.tensors[self._single_output_name(node)]
        if source.dtype != output.dtype:
            _fail("VXREF028", "Transpose requires the same input/output dtype", node)
        if source.dtype not in {"float32", "int32", "int8", "uint8"}:
            _fail("VXREF028", f"Transpose does not support {source.dtype} storage", node)
        self._require_byte_mapping_preserved(node, (source, output))
        permutation = params.get("perm", list(reversed(range(source.rank))))
        if (
            not isinstance(permutation, (list, tuple)) or
            any(isinstance(axis, bool) or not isinstance(axis, int) for axis in permutation) or
            sorted(permutation) != list(range(source.rank))
        ):
            _fail("VXREF029", "Transpose has an invalid permutation", node)
        return np.transpose(inputs["input"], axes=tuple(permutation)).copy()

    def _concat(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        params = self._params(node, ("axis", "count"))
        if not inputs:
            _fail("VXREF030", "Concat requires at least one input", node)
        output = self.graph.tensors[self._single_output_name(node)]
        descriptors = [
            self.graph.tensors[port.value]
            for port in node.inputs if port.value is not None
        ]
        if any(descriptor.dtype != output.dtype for descriptor in descriptors):
            _fail("VXREF031", "Concat input/output dtypes differ", node)
        if output.dtype not in {"float32", "int32", "int8", "uint8"}:
            _fail("VXREF031", f"Concat does not support {output.dtype} storage", node)
        self._require_byte_mapping_preserved(node, (*descriptors, output))
        count = params.get("count", len(inputs))
        if isinstance(count, bool) or not isinstance(count, int) or count != len(inputs):
            _fail("VXREF032", "Concat count does not match its ordered inputs", node)
        axis = params.get("axis", 0)
        if isinstance(axis, bool) or not isinstance(axis, int):
            _fail("VXREF033", "Concat axis must be an integer", node)
        if axis < 0:
            axis += output.rank
        if axis < 0 or axis >= output.rank:
            _fail("VXREF033", "Concat axis is outside the output rank", node)
        ordered = [inputs[port.name] for port in node.inputs if port.value is not None]
        return np.concatenate(ordered, axis=axis)

    def _quantize(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        self._params(node, ())
        self._require_ports(node, inputs, ("input", "scale", "zero_point"))
        output_name = self._single_output_name(node)
        output = self.graph.tensors[output_name]
        source = self.graph.tensors[node.input_map()["input"]]
        if (source.dtype != "float32" or output.dtype not in _BYTE_DTYPES or
                source.shape != output.shape):
            _fail("VXREF034", "QuantizeLinear requires equal-shape F32 to I8/U8 edges", node)
        descriptor, scales, zeros = self._affine(output_name)
        self._require_per_tensor(descriptor, node)
        scale = np.float32(scales.reshape(-1)[0])
        zero = int(zeros.reshape(-1)[0])
        return self._quantize_values(inputs["input"], scale, zero, output.dtype)

    def _dequantize(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        self._params(node, ())
        self._require_ports(node, inputs, ("input", "scale", "zero_point"))
        input_name = node.input_map()["input"]
        source = self.graph.tensors[input_name]
        output = self.graph.tensors[self._single_output_name(node)]
        if (source.dtype not in _BYTE_DTYPES or output.dtype != "float32" or
                source.shape != output.shape):
            _fail("VXREF035", "DequantizeLinear requires equal-shape I8/U8 to F32 edges", node)
        descriptor, scales, zeros = self._affine(input_name)
        self._require_per_tensor(descriptor, node)
        centered = np.subtract(
            inputs["input"].astype(np.float32),
            np.float32(int(zeros.reshape(-1)[0])),
            dtype=np.float32,
        )
        return np.multiply(centered, np.float32(scales.reshape(-1)[0]), dtype=np.float32)

    def _requantize(self, node: OpNode, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        self._params(node, ())
        self._require_ports(node, inputs, ("input",))
        input_name = node.input_map()["input"]
        output_name = self._single_output_name(node)
        source = self.graph.tensors[input_name]
        output = self.graph.tensors[output_name]
        if (source.dtype not in _BYTE_DTYPES or output.dtype not in _BYTE_DTYPES or
                source.shape != output.shape):
            _fail("VXREF036", "RequantizeLinear requires equal-shape I8/U8 edges", node)
        source_q, source_scales, source_zeros = self._affine(input_name)
        output_q, output_scales, output_zeros = self._affine(output_name)
        self._require_per_tensor(source_q, node)
        self._require_per_tensor(output_q, node)
        multiplier = np.float32(
            np.float32(source_scales.reshape(-1)[0]) /
            np.float32(output_scales.reshape(-1)[0])
        )
        if not np.isfinite(multiplier) or multiplier <= 0:
            _fail("VXREF037", "RequantizeLinear has an invalid F32 scale ratio", node)
        centered = np.subtract(
            inputs["input"].astype(np.int32),
            np.int32(int(source_zeros.reshape(-1)[0])),
            dtype=np.int32,
        ).astype(np.float32)
        transformed = np.add(
            np.multiply(centered, multiplier, dtype=np.float32),
            np.float32(int(output_zeros.reshape(-1)[0])),
            dtype=np.float32,
        )
        return self._rounded_saturated(
            transformed, int(output_zeros.reshape(-1)[0]), output.dtype,
        )

    def _affine(
        self, tensor_name: str,
    ) -> tuple[AffineQuantization, np.ndarray, np.ndarray]:
        tensor = self.graph.tensors[tensor_name]
        descriptor = tensor.quantization
        if descriptor is None:
            _fail("VXREF038", f"byte tensor {tensor_name!r} has no affine descriptor")
        scales = self._initializers[descriptor.scale]
        zeros = self._initializers[descriptor.zero_point]
        if not np.all(np.isfinite(scales)) or np.any(scales <= np.float32(0.0)):
            _fail("VXREF039", f"tensor {tensor_name!r} has non-positive/non-finite F32 scales")
        return descriptor, scales, zeros

    @staticmethod
    def _require_per_tensor(descriptor: AffineQuantization, node: OpNode) -> None:
        if descriptor.scheme != "per_tensor":
            _fail("VXREF040", f"{node.op_type} requires per-tensor affine mapping", node)

    @staticmethod
    def _normalized_axis(axis: Optional[int], rank: int, node: OpNode) -> int:
        if axis is None:
            _fail("VXREF041", "per-axis affine mapping has no axis", node)
        normalized = axis + rank if axis < 0 else axis
        if normalized < 0 or normalized >= rank:
            _fail("VXREF041", "per-axis affine mapping axis is out of range", node)
        return normalized

    def _dequantized_array(
        self,
        value: np.ndarray,
        descriptor: AffineQuantization,
        scales: np.ndarray,
        zeros: np.ndarray,
        node: OpNode,
    ) -> np.ndarray:
        if descriptor.scheme == "per_tensor":
            scale_values = np.float32(scales.reshape(-1)[0])
            zero_values = np.float32(int(zeros.reshape(-1)[0]))
        else:
            axis = self._normalized_axis(descriptor.axis, value.ndim, node)
            broadcast = [1] * value.ndim
            broadcast[axis] = value.shape[axis]
            scale_values = scales.reshape(broadcast).astype(np.float32)
            zero_values = zeros.reshape(broadcast).astype(np.float32)
        centered = np.subtract(value.astype(np.float32), zero_values, dtype=np.float32)
        return np.multiply(centered, scale_values, dtype=np.float32)

    def _require_byte_mapping_preserved(
        self, node: OpNode, descriptors: Sequence[TensorValue],
    ) -> None:
        if not descriptors or descriptors[0].dtype not in _BYTE_DTYPES:
            return
        first = descriptors[0].quantization
        if first is None:
            if any(descriptor.quantization is not None
                   for descriptor in descriptors[1:]):
                _fail("VXREF042", f"{node.op_type} changes a raw byte mapping", node)
            return
        if first.scheme != "per_tensor":
            _fail(
                "VXREF042",
                f"{node.op_type} byte movement requires per-tensor affine descriptors",
                node,
            )
        _, scales, zeros = self._affine(descriptors[0].name)
        for descriptor in descriptors[1:]:
            current = descriptor.quantization
            if current is None or current.scheme != "per_tensor":
                _fail("VXREF042", f"{node.op_type} changes a byte tensor mapping", node)
            _, other_scales, other_zeros = self._affine(descriptor.name)
            if not np.array_equal(scales, other_scales) or not np.array_equal(zeros, other_zeros):
                _fail("VXREF042", f"{node.op_type} changes a byte tensor mapping", node)

    @staticmethod
    def _quantize_values(
        values: np.ndarray, scale: np.float32, zero: int, dtype: str,
    ) -> np.ndarray:
        if not np.isfinite(scale) or scale <= 0:
            _fail("VXREF043", "QuantizeLinear scale must be finite and positive")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            transformed = np.add(
                np.divide(values, scale, dtype=np.float32),
                np.float32(zero),
                dtype=np.float32,
            )
        return ReferenceExecutor._rounded_saturated(transformed, zero, dtype)

    @staticmethod
    def _rounded_saturated(
        transformed: np.ndarray, zero: int, dtype: str,
    ) -> np.ndarray:
        info = np.iinfo(_NUMPY_DTYPES[dtype])
        safe = np.where(np.isnan(transformed), np.float32(zero), transformed)
        clipped = np.clip(safe, np.float32(info.min), np.float32(info.max))
        return np.rint(clipped).astype(_NUMPY_DTYPES[dtype])


def execute_reference(
    graph: GraphIR,
    initializers: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> ReferenceExecution:
    """Convenience wrapper for a one-shot reference execution."""

    executor = ReferenceExecutor(graph, initializers)
    return executor.run(executor.bind(inputs))


__all__ = [
    "ReferenceExecution",
    "ReferenceExecutor",
    "execute_reference",
]
