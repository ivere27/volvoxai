"""Portable raw-byte operator contracts for typed RuntimeIR.

The browser loader validates hydrated tensor objects after resolving the
reference-only affine table from safetensors.  Offline publication must apply
the same graph-level rule: a byte activation with affine metadata is a typed
real-value edge, not storage that an arbitrary float operator may reinterpret.

This module mirrors ``ts/ops/quantizedGraphValidation.ts`` against the typed
RuntimeIR.  It intentionally does not choose a backend or validate ordinary
float operators; target-neutral descriptor validation and runtime backend
compilation own those separate concerns.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Callable, Mapping

from .errors import Diagnostic, ExporterError
from .ir import GraphIR, OpNode
from .quantized_embedding import embedding_ids_preflight_proof
from .quantization_storage import ResolvedQuantization


_BYTE_DTYPES = frozenset({"int8", "uint8"})
_RUNTIME_DTYPES = frozenset({"float32", "int32", "int8", "uint8"})
_DTYPE_BYTES = {"float32": 4, "int32": 4, "int8": 1, "uint8": 1}
_MAX_SAFE_INTEGER = (1 << 53) - 1
_U32_MAX = (1 << 32) - 1
_I32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class _QuantizationView:
    scheme: str
    scale: float | None = None
    zero_point: int | None = None
    axis: int | None = None
    scales: tuple[float, ...] = ()
    zero_points: tuple[int, ...] = ()


@dataclass(frozen=True)
class _TensorView:
    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    is_input: bool
    quantization: _QuantizationView | None


def _f32(value: Any) -> float:
    """Match JavaScript ``Math.fround`` for validation decisions."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, TypeError, ValueError):
        if isinstance(value, (int, float)):
            return -math.inf if value < 0 else math.inf
        return math.nan


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _is_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value) and value.is_integer()


def _is_safe_integer(value: Any) -> bool:
    return _is_integer(value) and abs(int(value)) <= _MAX_SAFE_INTEGER


def _js_truthy(value: Any) -> bool:
    """The few parameter checks that intentionally use JavaScript truthiness."""

    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        return bool(value)
    # Arrays and objects are truthy in JavaScript, including empty ones.
    return True


def _product(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def _valid_shape(tensor: _TensorView | None) -> bool:
    return tensor is not None and all(
        isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
        for dimension in tensor.shape
    )


def _same_shape(left: _TensorView | None, right: _TensorView | None) -> bool:
    return left is not None and right is not None and left.shape == right.shape


def _byte_tensor(tensor: _TensorView | None) -> bool:
    return tensor is not None and tensor.dtype in _BYTE_DTYPES


def _quantized_byte_tensor(tensor: _TensorView | None) -> bool:
    return _byte_tensor(tensor) and tensor.quantization is not None


def _per_tensor(tensor: _TensorView | None) -> bool:
    return (
        _byte_tensor(tensor)
        and tensor.quantization is not None
        and tensor.quantization.scheme == "per_tensor"
    )


def _same_byte_domain(left: _TensorView | None, right: _TensorView | None) -> bool:
    return (
        _byte_tensor(left)
        and _byte_tensor(right)
        and left.dtype == right.dtype
        and left.quantization == right.quantization
    )


def _params(node: OpNode) -> Mapping[str, Any]:
    for attribute in node.attributes:
        if attribute.name == "params" and attribute.kind == "volvox.params":
            return attribute.value if isinstance(attribute.value, Mapping) else {}
    return {}


class _Validator:
    def __init__(
        self,
        graph: GraphIR,
        resolved: Mapping[str, ResolvedQuantization],
    ) -> None:
        self.graph = graph
        quantization: dict[str, _QuantizationView] = {}
        for name, descriptor in resolved.items():
            if descriptor.scheme == "per_axis":
                quantization[name] = _QuantizationView(
                    scheme="per_axis",
                    axis=descriptor.axis,
                    scales=tuple(float(value) for value in descriptor.scales),
                    zero_points=tuple(int(value) for value in descriptor.zero_points),
                )
            else:
                quantization[name] = _QuantizationView(
                    scheme="per_tensor",
                    scale=float(descriptor.scales[0]),
                    zero_point=int(descriptor.zero_points[0]),
                )
        self.tensors = {
            name: _TensorView(
                name=name,
                shape=tuple(int(dimension) for dimension in tensor.shape),
                dtype=tensor.dtype,
                size_bytes=_product(tuple(int(dimension) for dimension in tensor.shape))
                * _DTYPE_BYTES.get(tensor.dtype, 0),
                is_input=tensor.public_input,
                quantization=quantization.get(name),
            )
            for name, tensor in graph.tensors.items()
        }

    @staticmethod
    def _reject(node: OpNode, message: str) -> None:
        raise ExporterError(Diagnostic(
            code="VXRTIR025",
            message=f"{node.op_type} node {node.name!r} {message}",
            stage="runtime-quantized-contract",
            source_node=node.name,
            source_op=node.op_type,
            constraint="portable typed I8/U8 execution contract",
        ))

    def _inputs(self, node: OpNode) -> dict[str, _TensorView]:
        return {
            port: self.tensors[name]
            for port, name in node.input_map().items()
        }

    def _outputs(self, node: OpNode) -> dict[str, _TensorView]:
        return {
            port: self.tensors[name]
            for port, name in node.output_map().items()
        }

    def _node_input(
        self,
        node: OpNode,
        inputs: Mapping[str, _TensorView],
        names: tuple[str, ...],
        label: str,
    ) -> _TensorView:
        matches = [name for name in names if name in inputs]
        if len(matches) != 1:
            self._reject(
                node,
                f"requires exactly one {label!r} input ({'/'.join(names)})",
            )
        return inputs[matches[0]]

    def _only_inputs(
        self,
        node: OpNode,
        inputs: Mapping[str, _TensorView],
        allowed: frozenset[str],
    ) -> None:
        unexpected = next((name for name in inputs if name not in allowed), None)
        if unexpected is not None:
            self._reject(
                node,
                f"has unsupported canonical W8A8 input {unexpected!r}",
            )

    def _single_output(
        self,
        node: OpNode,
        outputs: Mapping[str, _TensorView],
    ) -> _TensorView:
        if len(outputs) != 1:
            self._reject(node, "requires exactly one canonical W8A8 output")
        return next(iter(outputs.values()))

    def _require_per_axis_weight(
        self,
        node: OpNode,
        weight: _TensorView | None,
        output_channels: int,
        operation: str,
    ) -> None:
        descriptor = weight.quantization if weight is not None else None
        if (
            not _byte_tensor(weight)
            or descriptor is None
            or descriptor.scheme != "per_axis"
            or descriptor.axis != 0
            or len(descriptor.scales) != output_channels
            or len(descriptor.zero_points) != output_channels
        ):
            self._reject(
                node,
                f"{operation} requires I8/U8 weights with axis-0 per-output-channel quantization metadata",
            )

    def _qconv(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        activation = self._node_input(node, inputs, ("input", "x"), "activation")
        weight = inputs.get("weight")
        bias = inputs.get("bias")
        output = self._single_output(node, outputs)
        params = _params(node)
        self._only_inputs(node, inputs, frozenset({"input", "x", "weight", "bias"}))
        weight_quant = weight.quantization if weight is not None else None
        embedded = (
            "input_scale", "input_zero_point", "output_scale", "output_zero_point",
        )
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _byte_tensor(weight)
            or weight_quant is None
            or weight_quant.scheme != "per_axis"
            or weight_quant.axis != 0
            or not weight.shape
            or len(weight_quant.scales) != weight.shape[0]
            or len(weight_quant.zero_points) != weight.shape[0]
            or (
                bias is not None
                and (bias.dtype != "int32" or len(bias.shape) != 1 or bias.shape[0] != weight.shape[0])
            )
            or any(params.get(name) is not None for name in embedded)
            or (
                _js_truthy(params.get("data_layout"))
                and params.get("data_layout") != "NHWC"
            )
            or (
                _js_truthy(params.get("weight_layout"))
                and params.get("weight_layout") != "OHWI"
            )
        ):
            self._reject(
                node,
                "must use canonical I8/U8 NHWC/OHWI storage, per-tensor activation metadata, axis-0 per-channel weight metadata, and optional I32 bias",
            )
        if (
            len(activation.shape) != 4
            or len(weight.shape) != 4
            or len(output.shape) != 4
            or activation.shape[0] != output.shape[0]
            or weight.shape[0] != output.shape[3]
        ):
            self._reject(node, "has incompatible canonical NHWC/OHWI shapes")
        self._require_per_axis_weight(node, weight, output.shape[3], "QConv2D")

    def _qlinear(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        self._only_inputs(
            node, inputs, frozenset({"input", "x", "a", "weight", "bias"}),
        )
        activation = self._node_input(
            node, inputs, ("input", "x", "a"), "activation",
        )
        weight = inputs.get("weight")
        bias = inputs.get("bias")
        output = self._single_output(node, outputs)
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _byte_tensor(weight)
            or bias is None
            or bias.dtype != "int32"
            or len(activation.shape) < 1
            or len(output.shape) != len(activation.shape)
            or activation.shape[:-1] != output.shape[:-1]
            or weight is None
            or len(weight.shape) != 2
            or len(bias.shape) != 1
            or weight.shape[0] != output.shape[-1]
            or weight.shape[1] != activation.shape[-1]
            or bias.shape[0] != output.shape[-1]
        ):
            self._reject(
                node,
                "must use canonical typed [...,d_in] activations, [d_out,d_in] byte weights, I32 bias, and [...,d_out] output",
            )
        self._require_per_axis_weight(node, weight, output.shape[-1], node.op_type)
        input_scale = _f32(activation.quantization.scale)
        output_scale = _f32(output.quantization.scale)
        for weight_scale in weight.quantization.scales:
            multiplier = _f32(_f32(input_scale * _f32(weight_scale)) / output_scale)
            if not math.isfinite(multiplier) or multiplier <= 0:
                self._reject(
                    node,
                    "has a requantization multiplier not representable as positive F32",
                )

    @staticmethod
    def _centered_magnitude(tensor: _TensorView) -> int:
        if tensor.quantization is None or tensor.quantization.zero_point is None:
            return _MAX_SAFE_INTEGER + 1
        minimum, maximum = (-128, 127) if tensor.dtype == "int8" else (0, 255)
        return max(
            abs(minimum - tensor.quantization.zero_point),
            abs(maximum - tensor.quantization.zero_point),
        )

    def _qbatch_matmul(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        self._only_inputs(node, inputs, frozenset({"a", "b"}))
        left = inputs.get("a")
        right = inputs.get("b")
        output = self._single_output(node, outputs)
        valid_rank = lambda tensor: (
            _valid_shape(tensor) and tensor is not None and 2 <= len(tensor.shape) <= 8
        )
        if (
            not _per_tensor(left)
            or not _per_tensor(right)
            or not _per_tensor(output)
            or not valid_rank(left)
            or not valid_rank(right)
            or not valid_rank(output)
            or bool(_params(node))
        ):
            self._reject(
                node,
                "must use per-tensor I8/U8 rank-2..8 operands and no parameters",
            )
        left_batch = left.shape[:-2]
        right_batch = right.shape[:-2]
        rank = max(len(left_batch), len(right_batch))
        padded_left = (1,) * (rank - len(left_batch)) + left_batch
        padded_right = (1,) * (rank - len(right_batch)) + right_batch
        output_batch: list[int] = []
        for axis, (left_dim, right_dim) in enumerate(zip(padded_left, padded_right)):
            if left_dim != right_dim and left_dim != 1 and right_dim != 1:
                self._reject(
                    node,
                    f"has incompatible broadcast batch dimensions at axis {axis}",
                )
            output_batch.append(max(left_dim, right_dim))
        expected = (*output_batch, left.shape[-2], right.shape[-1])
        if left.shape[-1] != right.shape[-2] or output.shape != expected:
            self._reject(node, "has incompatible ONNX matrix or output dimensions")
        maximum_accumulator = (
            self._centered_magnitude(left)
            * self._centered_magnitude(right)
            * left.shape[-1]
        )
        if maximum_accumulator > _I32_MAX or maximum_accumulator > _MAX_SAFE_INTEGER:
            self._reject(node, "may overflow its defined I32 accumulator")
        multiplier = _f32(
            _f32(float(left.quantization.scale) * float(right.quantization.scale))
            / float(output.quantization.scale)
        )
        if not math.isfinite(multiplier) or multiplier <= 0:
            self._reject(
                node,
                "has a requantization multiplier not representable as positive F32",
            )

    def _qembedding(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        self._only_inputs(node, inputs, frozenset({"input", "weight"}))
        token_ids = self._node_input(node, inputs, ("input",), "token IDs")
        weight = inputs.get("weight")
        output = self._single_output(node, outputs)
        preflight_proof = (
            embedding_ids_preflight_proof(
                self.graph, token_ids.name, int(weight.shape[0]),
            )
            if weight is not None and len(weight.shape) == 2
            else None
        )
        if (
            preflight_proof is None
            or token_ids.dtype != "int32"
            or len(token_ids.shape) < 1
            or not _byte_tensor(weight)
            or not _per_tensor(output)
            or weight is None
            or len(weight.shape) != 2
            or len(output.shape) != len(token_ids.shape) + 1
            or output.shape[:-1] != token_ids.shape
            or output.shape[-1] != weight.shape[1]
        ):
            self._reject(
                node,
                "must use public or vocabulary-bounded canonical Clip I32 token IDs, [vocab,hidden] I8/U8 weight with axis-0 row metadata, and a per-tensor I8/U8 [...token,hidden] output",
            )
        self._require_per_axis_weight(node, weight, weight.shape[0], "QEmbedding")

    def _qadd(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        self._only_inputs(
            node, inputs, frozenset({"a", "input", "x", "b", "y"}),
        )
        left = self._node_input(
            node, inputs, ("a", "input", "x"), "left activation",
        )
        right = self._node_input(node, inputs, ("b", "y"), "right activation")
        output = self._single_output(node, outputs)
        params = _params(node)
        relu = params.get("relu")
        if relu is None:
            relu = 0
        if (
            not _per_tensor(left)
            or not _per_tensor(right)
            or not _per_tensor(output)
            or not _same_shape(left, right)
            or not _same_shape(left, output)
            or not _is_integer(relu)
            or not 0 <= int(relu) <= 2
        ):
            self._reject(
                node,
                "must use exact-shape canonical per-tensor I8/U8 edges and relu 0, 1, or 2",
            )

    def _qsilu(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        self._only_inputs(node, inputs, frozenset({"input", "x", "data"}))
        activation = self._node_input(
            node, inputs, ("input", "x", "data"), "activation",
        )
        output = self._single_output(node, outputs)
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _same_shape(activation, output)
            or activation.name == output.name
            or bool(_params(node))
        ):
            self._reject(
                node,
                "must use one same-shape canonical per-tensor I8/U8 activation input/output pair with distinct tensors and no parameters",
            )

    def _qgelu(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        self._only_inputs(node, inputs, frozenset({"input", "x", "data"}))
        activation = self._node_input(
            node, inputs, ("input", "x", "data"), "activation",
        )
        output = self._single_output(node, outputs)
        params = _params(node)
        canonical_params = not params or (
            set(params) == {"approximate"} and params.get("approximate") == "none"
        )
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _same_shape(activation, output)
            or activation.name == output.name
            or not canonical_params
        ):
            self._reject(
                node,
                "must use one same-shape canonical per-tensor I8/U8 activation input/output pair with distinct tensors and only omitted parameters or approximate='none'",
            )

    @staticmethod
    def _valid_affine(tensor: _TensorView | None, width: int | None) -> bool:
        return (
            tensor is not None
            and width is not None
            and tensor.dtype == "float32"
            and tensor.shape == (width,)
            and tensor.size_bytes == width * 4
        )

    def _qgroup_norm(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        activation = inputs.get("input")
        weight = inputs.get("weight")
        bias = inputs.get("bias")
        output = self._single_output(node, outputs)
        params = _params(node)
        channels = activation.shape[3] if activation is not None and len(activation.shape) == 4 else None
        epsilon_source = params.get("eps")
        if epsilon_source is None:
            epsilon_source = 1e-5
        epsilon = _f32(epsilon_source)
        num_groups = params.get("num_groups")
        valid_groups = _is_integer(num_groups) and int(num_groups) > 0
        if (
            set(inputs) != {"input", "weight", "bias"}
            or not _per_tensor(activation)
            or not _per_tensor(output)
            or activation is None
            or activation.name == output.name
            or len(activation.shape) != 4
            or len(output.shape) != 4
            or not _same_shape(activation, output)
            or not self._valid_affine(weight, channels)
            or not self._valid_affine(bias, channels)
            or not set(params) <= {"num_groups", "eps", "data_layout"}
            or not valid_groups
            or channels is None
            or channels % int(num_groups) != 0
            or (params.get("data_layout") is not None and params.get("data_layout") != "NHWC")
            or not _is_number(epsilon_source)
            or not math.isfinite(float(epsilon_source))
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            self._reject(
                node,
                "must use exact input/weight/bias inputs, rank-4 NHWC per-tensor I8/U8 activation edges, F32 [C] affine tensors, and positive num_groups/eps parameters",
            )

    def _qlayer_norm(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        activation = inputs.get("input")
        weight = inputs.get("weight")
        bias = inputs.get("bias")
        output = self._single_output(node, outputs)
        params = _params(node)
        d_model = activation.shape[-1] if activation is not None and activation.shape else None
        epsilon_source = params.get("eps")
        if epsilon_source is None:
            epsilon_source = 1e-5
        epsilon = _f32(epsilon_source)
        declared_d_model = params.get("d_model")
        valid_d_model = (
            declared_d_model is None
            or (_is_integer(declared_d_model) and int(declared_d_model) == d_model)
        )
        if (
            set(inputs) != {"input", "weight", "bias"}
            or not _per_tensor(activation)
            or not _per_tensor(output)
            or activation is None
            or activation.name == output.name
            or output.name in {
                weight.name if weight is not None else "",
                bias.name if bias is not None else "",
            }
            or len(activation.shape) < 1
            or not _same_shape(activation, output)
            or not self._valid_affine(weight, d_model)
            or not self._valid_affine(bias, d_model)
            or not set(params) <= {"eps", "d_model"}
            or not valid_d_model
            or not _is_number(epsilon_source)
            or not math.isfinite(float(epsilon_source))
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            self._reject(
                node,
                "must use exact input/weight/bias inputs, same-shape rank-at-least-1 per-tensor I8/U8 activation edges, F32 [D] affine tensors, and positive eps with optional d_model matching D",
            )

    def _valid_per_tensor_values(self, tensor: _TensorView | None) -> bool:
        if not _per_tensor(tensor):
            return False
        descriptor = tensor.quantization
        minimum, maximum = (-128, 127) if tensor.dtype == "int8" else (0, 255)
        encoded_scale = _f32(descriptor.scale)
        return (
            _is_number(descriptor.scale)
            and math.isfinite(float(descriptor.scale))
            and float(descriptor.scale) > 0
            and math.isfinite(encoded_scale)
            and encoded_scale > 0
            and _is_integer(descriptor.zero_point)
            and minimum <= int(descriptor.zero_point) <= maximum
        )

    def _qmasked_mean(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        activation = inputs.get("input")
        mask = inputs.get("mask")
        output = outputs.get("out")
        params = _params(node)
        input_elements = _product(activation.shape) if _valid_shape(activation) else -1
        mask_elements = _product(mask.shape) if _valid_shape(mask) else -1
        output_elements = _product(output.shape) if _valid_shape(output) else -1
        batch = activation.shape[0] if activation is not None and len(activation.shape) >= 1 else None
        sequence = activation.shape[1] if activation is not None and len(activation.shape) >= 2 else None
        width = activation.shape[2] if activation is not None and len(activation.shape) >= 3 else None
        multiplier = math.nan
        maximum_sum = _MAX_SAFE_INTEGER + 1
        if (
            self._valid_per_tensor_values(activation)
            and self._valid_per_tensor_values(output)
        ):
            multiplier = _f32(
                _f32(activation.quantization.scale)
                / _f32(output.quantization.scale)
            )
            minimum, maximum = (-128, 127) if activation.dtype == "int8" else (0, 255)
            centered = max(
                abs(minimum - int(activation.quantization.zero_point)),
                abs(maximum - int(activation.quantization.zero_point)),
            )
            if sequence is not None:
                maximum_sum = sequence * centered
        if (
            set(inputs) != {"input", "mask"}
            or set(outputs) != {"out"}
            or output is None
            or bool(params)
            or not self._valid_per_tensor_values(activation)
            or not self._valid_per_tensor_values(output)
            or activation is None
            or activation.name == output.name
            or not _valid_shape(activation)
            or not _valid_shape(mask)
            or not _valid_shape(output)
            or len(activation.shape) != 3
            or mask is None
            or mask.dtype != "int32"
            or mask.quantization is not None
            or len(mask.shape) != 2
            or len(output.shape) != 2
            or mask.shape != (batch, sequence)
            or output.shape != (batch, width)
            or not _is_safe_integer(input_elements)
            or not _is_safe_integer(mask_elements)
            or not _is_safe_integer(output_elements)
            or activation.size_bytes != input_elements
            or mask.size_bytes != mask_elements * 4
            or output.size_bytes != output_elements
            or input_elements != batch * sequence * width
            or mask_elements != batch * sequence
            or output_elements != batch * width
            or not _is_safe_integer(maximum_sum)
            or maximum_sum > _I32_MAX
            or not math.isfinite(multiplier)
            or multiplier <= 0
        ):
            self._reject(
                node,
                "must use exactly { input, mask } -> { out }: immutable per-tensor I8/U8 [B,S,D] and [B,D] edges, an unquantized I32 [B,S] keep mask, no parameters, and an I32-safe centered sum",
            )

    def _qsdpa(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        q = inputs.get("q")
        k = inputs.get("k")
        v = inputs.get("v")
        has_mask = "mask" in inputs
        mask = inputs.get("mask")
        output = self._single_output(node, outputs)
        params = _params(node)
        rank = len(q.shape) if q is not None else None
        geometry_ranked = (
            rank in {2, 3}
            and k is not None
            and v is not None
            and len(k.shape) == rank
            and len(v.shape) == rank
            and len(output.shape) == rank
        )
        batch = 1 if rank == 2 else (q.shape[0] if q is not None and rank == 3 else None)
        queries = q.shape[-2] if q is not None and rank in {2, 3} else None
        keys = k.shape[-2] if k is not None and rank in {2, 3} else None
        d_model = q.shape[-1] if q is not None and rank in {2, 3} else None
        k_batch = 1 if rank == 2 else (k.shape[0] if k is not None and rank == 3 else None)
        v_batch = 1 if rank == 2 else (v.shape[0] if v is not None and rank == 3 else None)
        valid_mask = not has_mask
        if has_mask and mask is not None and keys is not None:
            valid_mask = (
                mask.dtype == "int32"
                and _valid_shape(mask)
                and mask.size_bytes == _product(mask.shape) * 4
                and (
                    (len(mask.shape) == 1 and mask.shape[0] == keys)
                    or (
                        len(mask.shape) == 2
                        and mask.shape[1] == keys
                        and mask.shape[0] in {batch, queries}
                    )
                    or (
                        len(mask.shape) == 3
                        and mask.shape == (batch, queries, keys)
                    )
                )
            )
        heads = params.get("heads")
        valid_heads = _is_integer(heads) and int(heads) > 0
        head_dim = d_model / int(heads) if d_model is not None and valid_heads else math.nan
        scale_source = params.get("scale")
        if scale_source is None:
            scale_source = 1 / math.sqrt(head_dim) if head_dim > 0 else math.nan
        scale = _f32(scale_source)
        score_multiplier = math.nan
        maximum_score = math.nan
        if (
            self._valid_per_tensor_values(q)
            and self._valid_per_tensor_values(k)
        ):
            q_scale = _f32(q.quantization.scale)
            k_scale = _f32(k.quantization.scale)
            score_multiplier = _f32(_f32(q_scale * k_scale) * scale)
            if _is_integer(head_dim):
                maximum_score = _f32(
                    _f32(
                        int(head_dim)
                        * self._centered_magnitude(q)
                        * self._centered_magnitude(k)
                    )
                    * score_multiplier
                )
        exact_inputs = set(inputs) in ({"q", "k", "v"}, {"q", "k", "v", "mask"})
        if (
            not exact_inputs
            or not self._valid_per_tensor_values(q)
            or not self._valid_per_tensor_values(k)
            or not self._valid_per_tensor_values(v)
            or not self._valid_per_tensor_values(output)
            or q is None
            or k is None
            or v is None
            or output.name in {q.name, k.name, v.name, mask.name if mask is not None else ""}
            or not _valid_shape(q)
            or not _valid_shape(k)
            or not _valid_shape(v)
            or not _valid_shape(output)
            or not geometry_ranked
            or not _same_shape(q, output)
            or k_batch != batch
            or v_batch != batch
            or k.shape[-1] != d_model
            or v.shape[-1] != d_model
            or v.shape[-2] != keys
            or not valid_mask
            or not set(params) <= {"heads", "causal", "scale"}
            or not valid_heads
            or not _is_integer(head_dim)
            or int(head_dim) <= 0
            or d_model % 4 != 0
            or int(head_dim) % 4 != 0
            or int(head_dim) > 64
            or "heads" not in params
            or "causal" not in params
            or not isinstance(params.get("causal"), bool)
            or not _is_number(scale_source)
            or not math.isfinite(float(scale_source))
            or float(scale_source) <= 0
            or not math.isfinite(scale)
            or scale <= 0
            or not math.isfinite(score_multiplier)
            or score_multiplier <= 0
            or not math.isfinite(maximum_score)
        ):
            self._reject(
                node,
                "must use exact q/k/v and optional I32 mask inputs, rank-2/3 per-tensor I8/U8 tensors, D/head dimensions divisible by 4 with head_dim <= 64, and explicit heads/causal plus finite score scaling",
            )

    def _qargmax(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        outputs = self._outputs(node)
        activation = inputs.get("input")
        output = outputs.get("out")
        params = _params(node)
        rank = len(activation.shape) if activation is not None else None
        axis_source = params.get("axis")
        axis = int(axis_source) if _is_integer(axis_source) else None
        if axis is not None and axis < 0 and rank is not None:
            axis += rank
        input_elements = _product(activation.shape) if _valid_shape(activation) else -1
        output_elements = _product(output.shape) if _valid_shape(output) else -1
        axis_size = (
            activation.shape[axis]
            if activation is not None and axis is not None and 0 <= axis < len(activation.shape)
            else None
        )
        outer = (
            _product(activation.shape[:axis])
            if activation is not None and axis is not None and 0 <= axis < len(activation.shape)
            else -1
        )
        inner = (
            _product(activation.shape[axis + 1:])
            if activation is not None and axis is not None and 0 <= axis < len(activation.shape)
            else -1
        )
        expected = (
            activation.shape[:axis] + activation.shape[axis + 1:]
            if activation is not None and axis is not None and 0 <= axis < len(activation.shape)
            else None
        )
        if (
            set(inputs) != {"input"}
            or set(outputs) != {"out"}
            or activation is None
            or output is None
            or activation.name == output.name
            or not self._valid_per_tensor_values(activation)
            or not _valid_shape(activation)
            or not _valid_shape(output)
            or rank is None
            or rank < 2
            or rank > 8
            or set(params) != {"axis"}
            or not _is_integer(axis_source)
            or axis is None
            or axis < 0
            or axis >= rank
            or axis_size is None
            or axis_size <= 0
            or axis_size > _I32_MAX
            or not _is_safe_integer(outer)
            or outer <= 0
            or not _is_safe_integer(inner)
            or inner <= 0
            or outer > _U32_MAX
            or inner > _U32_MAX
            or not _is_safe_integer(input_elements)
            or input_elements <= 0
            or input_elements > _U32_MAX
            or not _is_safe_integer(output_elements)
            or output_elements <= 0
            or output_elements > _U32_MAX
            or activation.size_bytes != input_elements
            or output.dtype != "int32"
            or output.quantization is not None
            or output.size_bytes != output_elements * 4
            or output_elements != outer * inner
            or input_elements != outer * axis_size * inner
            or output.shape != expected
        ):
            self._reject(
                node,
                "must use exactly { input } -> { out }, immutable per-tensor rank-2..8 I8/U8 input, exact { axis: integer } parameters, and an unquantized I32 output with the reduced axis removed",
            )

    def _requantize(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        output = self._single_output(node, self._outputs(node))
        self._only_inputs(node, inputs, frozenset({"input", "x", "data"}))
        activation = self._node_input(
            node, inputs, ("input", "x", "data"), "activation",
        )
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _same_shape(activation, output)
        ):
            self._reject(
                node, "requires equal-shape canonical per-tensor I8/U8 edges",
            )

    def _quantize(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        output = self._single_output(node, self._outputs(node))
        self._only_inputs(
            node,
            inputs,
            frozenset({"input", "x", "data", "scale", "zero_point"}),
        )
        activation = self._node_input(
            node, inputs, ("input", "x", "data"), "F32 activation",
        )
        scale = inputs.get("scale")
        zero = inputs.get("zero_point")
        if (
            activation.dtype != "float32"
            or scale is None
            or scale.dtype != "float32"
            or scale.size_bytes != 4
            or not _per_tensor(output)
            or not _same_shape(activation, output)
            or (
                zero is not None
                and (zero.dtype != output.dtype or zero.size_bytes != 1)
            )
        ):
            self._reject(
                node,
                "requires F32 input, scalar F32 scale, optional scalar matching zero point, and per-tensor I8/U8 output",
            )

    def _dequantize(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        output = self._single_output(node, self._outputs(node))
        self._only_inputs(
            node,
            inputs,
            frozenset({"input", "x", "data", "scale", "zero_point"}),
        )
        activation = self._node_input(
            node, inputs, ("input", "x", "data"), "activation",
        )
        scale = inputs.get("scale")
        zero = inputs.get("zero_point")
        supported_zero = zero is None or (
            zero.dtype in {"float32", "int32"} and zero.size_bytes == 4
        ) or (
            zero.dtype in _BYTE_DTYPES and zero.size_bytes == 1
        )
        if (
            activation.dtype not in _RUNTIME_DTYPES
            or scale is None
            or scale.dtype != "float32"
            or scale.size_bytes != 4
            or output.dtype != "float32"
            or not _same_shape(activation, output)
            or not supported_zero
        ):
            self._reject(
                node,
                "requires equal-shape F32/I32/I8/U8 input and F32 output, scalar F32 scale, and an optional scalar F32/I32/I8/U8 zero point",
            )

    def _shape_copy(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        output = self._single_output(node, self._outputs(node))
        self._only_inputs(node, inputs, frozenset({"input"}))
        activation = self._node_input(node, inputs, ("input",), "activation")
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _same_byte_domain(activation, output)
            or activation.size_bytes != output.size_bytes
        ):
            self._reject(
                node,
                "requires equal-size I8/U8 storage with identical per-tensor quantization metadata",
            )

    def _transpose(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        output = self._single_output(node, self._outputs(node))
        self._only_inputs(node, inputs, frozenset({"input", "x", "data"}))
        activation = self._node_input(
            node, inputs, ("input", "x", "data"), "activation",
        )
        params = _params(node)
        rank = len(activation.shape)
        permutation = params.get("perm")
        if permutation is None:
            permutation = list(reversed(range(rank)))
        valid_permutation = (
            isinstance(permutation, list)
            and len(permutation) == rank
            and all(_is_integer(axis) for axis in permutation)
            and len({int(axis) for axis in permutation}) == rank
            and all(0 <= int(axis) < rank for axis in permutation)
        )
        expected = (
            tuple(activation.shape[int(axis)] for axis in permutation)
            if valid_permutation else None
        )
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _same_byte_domain(activation, output)
            or rank < 1
            or rank > 8
            or len(output.shape) != rank
            or activation.size_bytes != output.size_bytes
            or not set(params) <= {"perm"}
            or not valid_permutation
            or output.shape != expected
        ):
            self._reject(
                node, "requires a descriptor-preserving rank-1..8 I8/U8 permutation",
            )

    def _slice(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        output = self._single_output(node, self._outputs(node))
        self._only_inputs(node, inputs, frozenset({"input", "x", "data"}))
        activation = self._node_input(
            node, inputs, ("input", "x", "data"), "activation",
        )
        rank = len(activation.shape)
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _same_byte_domain(activation, output)
            or rank < 1
            or rank > 8
            or len(output.shape) != rank
            or output.size_bytes > activation.size_bytes
        ):
            self._reject(
                node,
                "requires a descriptor-preserving rank-1..8 I8/U8 selection no larger than its input",
            )

    def _expand(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        output = self._single_output(node, self._outputs(node))
        params = _params(node)
        self._only_inputs(node, inputs, frozenset({"input", "x", "data"}))
        activation = self._node_input(
            node, inputs, ("input", "x", "data"), "activation",
        )
        input_rank = len(activation.shape)
        output_rank = len(output.shape)
        offset = output_rank - input_rank
        exact_broadcast = (
            1 <= input_rank <= output_rank <= 8
            and all(
                (
                    1 if output_axis < offset
                    else activation.shape[output_axis - offset]
                ) in {1, output.shape[output_axis]}
                for output_axis in range(output_rank)
            )
        )
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _same_byte_domain(activation, output)
            or not exact_broadcast
            or output.size_bytes < activation.size_bytes
            or activation.name == output.name
            or set(params) != {"shape"}
            or not isinstance(params.get("shape"), (list, tuple))
            or tuple(params["shape"]) != output.shape
        ):
            self._reject(
                node,
                "requires a distinct descriptor-preserving rank-1..8 I8/U8 exact broadcast with matching params.shape",
            )

    @staticmethod
    def _false_or_absent(value: Any) -> bool:
        return value is None or value is False or (
            _is_number(value) and float(value) == 0.0
        )

    def _nearest_mapping(self, node: OpNode, params: Mapping[str, Any]) -> None:
        mode = params.get("mode")
        if node.op_type == "Resize":
            if mode != "nearest":
                self._reject(node, 'supports raw I8/U8 Resize only with mode "nearest"')
        elif mode is not None and mode != "nearest":
            self._reject(
                node,
                'supports raw I8/U8 ResizeNearest2D only with mode "nearest"',
            )
        if params.get("coordinate_transform_mode") is not None:
            self._reject(
                node,
                "does not define coordinate_transform_mode; use coordinate_transformation_mode",
            )
        transform = params.get("coordinate_transformation_mode")
        if transform is not None and transform != "asymmetric":
            self._reject(
                node,
                'supports raw I8/U8 nearest resize only with coordinate_transformation_mode "asymmetric"',
            )
        nearest = params.get("nearest_mode")
        if nearest is not None and nearest != "floor":
            self._reject(
                node,
                'supports raw I8/U8 nearest resize only with nearest_mode "floor"',
            )
        if (
            not self._false_or_absent(params.get("align_corners"))
            or not self._false_or_absent(params.get("antialias"))
        ):
            self._reject(
                node,
                "does not support align_corners or antialias for raw I8/U8 nearest resize",
            )

    def _resize(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        output = self._single_output(node, self._outputs(node))
        self._only_inputs(node, inputs, frozenset({"input"}))
        activation = self._node_input(node, inputs, ("input",), "activation")
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _same_byte_domain(activation, output)
            or len(activation.shape) != 4
            or len(output.shape) != 4
            or activation.shape[0] != output.shape[0]
            or activation.shape[3] != output.shape[3]
        ):
            self._reject(
                node,
                "requires descriptor-preserving rank-4 NHWC I8/U8 input/output tensors",
            )
        self._nearest_mapping(node, _params(node))

    def _integer_pair(
        self,
        node: OpNode,
        params: Mapping[str, Any],
        name: str,
        fallback: int,
        minimum: int,
        *,
        required: bool = False,
    ) -> tuple[int, int]:
        source = params.get(name)
        if source is None:
            if required:
                self._reject(
                    node,
                    f"requires {name!r} for canonical W8A8 execution",
                )
            return fallback, fallback
        values = source if isinstance(source, list) else [source]
        if (
            not 1 <= len(values) <= 2
            or any(not _is_integer(value) or int(value) < minimum for value in values)
        ):
            self._reject(
                node,
                f"requires {name!r} to be one or two integers >= {minimum}",
            )
        return int(values[0]), int(values[1] if len(values) == 2 else values[0])

    def _maxpool(self, node: OpNode) -> None:
        inputs = self._inputs(node)
        output = self._single_output(node, self._outputs(node))
        self._only_inputs(node, inputs, frozenset({"input"}))
        activation = self._node_input(node, inputs, ("input",), "activation")
        params = _params(node)
        if (
            not _per_tensor(activation)
            or not _per_tensor(output)
            or not _same_byte_domain(activation, output)
            or len(activation.shape) != 4
            or len(output.shape) != 4
            or activation.shape[0] != output.shape[0]
            or activation.shape[3] != output.shape[3]
        ):
            self._reject(
                node,
                "requires descriptor-preserving rank-4 NHWC I8/U8 input/output tensors",
            )
        if not self._false_or_absent(params.get("ceil_mode")):
            self._reject(node, "does not support ceil_mode for raw I8/U8 storage")
        kernel = self._integer_pair(node, params, "kernel", 1, 1, required=True)
        stride = self._integer_pair(node, params, "stride", 1, 1)
        dilation = self._integer_pair(node, params, "dilation", 1, 1)
        padding = self._integer_pair(node, params, "padding", 0, 0)
        raw_pads = params.get("pads")
        if raw_pads is None:
            pads = (padding[0], padding[1], padding[0], padding[1])
        elif (
            isinstance(raw_pads, list)
            and len(raw_pads) == 4
            and all(_is_integer(value) and int(value) >= 0 for value in raw_pads)
        ):
            pads = tuple(int(value) for value in raw_pads)
        else:
            self._reject(
                node, "requires four non-negative top/left/bottom/right pads",
            )
        if dilation != (1, 1):
            self._reject(
                node, "supports raw I8/U8 MaxPool2D only with unit dilation",
            )
        expected_height = (
            (activation.shape[1] + pads[0] + pads[2] - kernel[0]) // stride[0]
        ) + 1
        expected_width = (
            (activation.shape[2] + pads[1] + pads[3] - kernel[1]) // stride[1]
        ) + 1
        if output.shape[1:3] != (expected_height, expected_width):
            self._reject(
                node,
                "has an output shape incompatible with canonical I8/U8 MaxPool2D parameters",
            )

    def _concat(self, node: OpNode) -> None:
        inputs = list(self._inputs(node).values())
        output = self._single_output(node, self._outputs(node))
        params = _params(node)
        axis_source = params.get("axis")
        if axis_source is None:
            axis_source = 0
        axis = int(axis_source) if _is_integer(axis_source) else None
        if axis is not None and axis < 0:
            axis += len(output.shape)
        if (
            not _per_tensor(output)
            or not inputs
            or axis is None
            or axis < 0
            or axis >= len(output.shape)
            or not self._false_or_absent(params.get("sigmoid"))
        ):
            self._reject(
                node,
                "requires canonical descriptor-preserving I8/U8 inputs, a valid axis, and no fused sigmoid",
            )
        axis_sum = 0
        for activation in inputs:
            if (
                not _per_tensor(activation)
                or not _same_byte_domain(activation, output)
                or len(activation.shape) != len(output.shape)
                or any(
                    dimension != output.shape[index]
                    for index, dimension in enumerate(activation.shape)
                    if index != axis
                )
            ):
                self._reject(
                    node,
                    "requires same-dtype inputs with identical per-tensor metadata and compatible shapes",
                )
            axis_sum += activation.shape[axis]
        if axis_sum != output.shape[axis]:
            self._reject(node, "has input axes that do not match its output axis")

    def assert_graph(self) -> None:
        handlers: dict[str, Callable[[OpNode], None]] = {
            "QConv2D": self._qconv,
            "QLinear": self._qlinear,
            "QMatMul": self._qlinear,
            "QGemm": self._qlinear,
            "QBatchMatMul": self._qbatch_matmul,
            "QEmbedding": self._qembedding,
            "QAdd": self._qadd,
            "QGELU": self._qgelu,
            "QGroupNorm": self._qgroup_norm,
            "QLayerNorm": self._qlayer_norm,
            "QMaskedMean": self._qmasked_mean,
            "QSDPA": self._qsdpa,
            "QArgMax": self._qargmax,
            "QSiLU": self._qsilu,
            "RequantizeLinear": self._requantize,
            "QuantizeLinear": self._quantize,
            "DequantizeLinear": self._dequantize,
            "MaxPool2D": self._maxpool,
            "Resize": self._resize,
            "ResizeNearest2D": self._resize,
            "Reshape": self._shape_copy,
            "Flatten": self._shape_copy,
            "Squeeze": self._shape_copy,
            "Unsqueeze": self._shape_copy,
            "Identity": self._shape_copy,
            "Transpose": self._transpose,
            "Slice": self._slice,
            "Expand": self._expand,
            "Concat": self._concat,
        }
        metadata_only = frozenset({
            "weight", "bias", "scale", "weight_scale", "zero_point",
            "weight_zero_point", "input_scale", "input_zero_point",
            "output_scale", "output_zero_point",
        })
        explicit = frozenset({
            "QConv2D", "QLinear", "QMatMul", "QGemm", "QBatchMatMul",
            "QAdd", "QGELU", "QGroupNorm", "QLayerNorm", "QMaskedMean",
            "QSDPA", "QArgMax", "QSiLU", "QEmbedding",
            "RequantizeLinear", "QuantizeLinear", "DequantizeLinear",
        })
        generic_arithmetic = frozenset({"Add", "Mul", "Sub", "Div"})
        float_boundary_activations = frozenset({
            "ReLU", "Sigmoid", "HardSwish", "HardSigmoid", "SiLU",
            "Swish", "Tanh",
        })
        for node in self.graph.nodes:
            inputs = self._inputs(node)
            outputs = self._outputs(node)
            activation_tensors = [
                tensor for name, tensor in inputs.items() if name not in metadata_only
            ] + list(outputs.values())
            typed_activations = [
                tensor for tensor in activation_tensors if _quantized_byte_tensor(tensor)
            ]
            byte_activations = [
                tensor for tensor in activation_tensors if _byte_tensor(tensor)
            ]
            handler = handlers.get(node.op_type)
            if node.op_type in generic_arithmetic and byte_activations:
                self._reject(
                    node,
                    "routes I8/U8 storage to an unsupported generic operator; use a typed quantized operator or an explicit F32 boundary",
                )
            if node.op_type in explicit:
                if handler is None:
                    self._reject(
                        node, "does not have a canonical typed validation handler",
                    )
                handler(node)
            elif typed_activations:
                if handler is None:
                    self._reject(
                        node,
                        "routes canonical raw I8/U8 storage to an unsupported generic operator; insert an explicit DequantizeLinear boundary or implement a typed operator",
                    )
                handler(node)
            if node.op_type in float_boundary_activations:
                activation = (
                    inputs.get("input") or inputs.get("x") or inputs.get("data")
                )
                output = outputs.get("out") or next(iter(outputs.values()), None)
                if _byte_tensor(activation) or _byte_tensor(output):
                    self._reject(
                        node,
                        "requires an explicit F32 boundary; it cannot reinterpret raw quantized bytes",
                    )
            if node.op_type == "Concat" and _js_truthy(_params(node).get("sigmoid")):
                output = outputs.get("out") or next(iter(outputs.values()), None)
                if _byte_tensor(output):
                    self._reject(
                        node,
                        "cannot fuse sigmoid on quantized Concat; insert an explicit F32 boundary",
                    )


def validate_portable_quantized_graph(
    graph: GraphIR,
    resolved: Mapping[str, ResolvedQuantization],
) -> None:
    """Reject a RuntimeIR graph outside the shared typed I8/U8 subset.

    ``resolved`` must come from the package's canonical safetensors-backed
    affine table.  The function stores no numeric affine data in RuntimeIR or
    graph JSON.
    """

    if not isinstance(graph, GraphIR):
        raise TypeError("graph must be a typed GraphIR")
    if not isinstance(resolved, Mapping):
        raise TypeError("resolved quantization must be a mapping")
    declared = {
        name: tensor.quantization
        for name, tensor in graph.tensors.items()
        if tensor.quantization is not None
    }
    if set(declared) != set(resolved):
        raise ExporterError(Diagnostic(
            code="VXRTIR025",
            message=(
                "RuntimeIR affine targets disagree with the resolved "
                "safetensors quantization table"
            ),
            stage="runtime-quantized-contract",
            constraint="one central affine descriptor per typed byte tensor",
        ))
    for name, affine in declared.items():
        descriptor = resolved[name]
        if (
            affine.scheme != descriptor.scheme
            or affine.axis != descriptor.axis
        ):
            raise ExporterError(Diagnostic(
                code="VXRTIR025",
                message=(
                    f"RuntimeIR affine target {name!r} disagrees with its "
                    "resolved safetensors scheme or axis"
                ),
                stage="runtime-quantized-contract",
                source_node=name,
                constraint="one central affine descriptor per typed byte tensor",
            ))
    _Validator(graph, resolved).assert_graph()
