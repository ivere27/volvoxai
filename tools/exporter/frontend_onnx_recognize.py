"""Pattern recognition for the ONNX frontend.

The `_recognize_*` family and its helpers: identity/ABI casts, W8A8 quantized
subgraphs, linear and attention shapes, and the feature probes they share.
Recognition decides which VolvoxAI operator an imported subgraph becomes;
emission is in frontend_onnx_emit.py.
"""

from __future__ import annotations

from __future__ import annotations
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
import numpy as np
from .capabilities import classify_package
from .errors import Diagnostic, ExporterError
from .importers.onnx import import_onnx_source
from .optimizer.typed_pipeline import (
    optimize_runtime_package,
    serialize_pipeline_report,
)
from .operator_shape_contracts import prove_operator_shape_domain
from .quantization_storage import externalize_quantization
from .shape_system import ShapeEnvironment
from .frontend_onnx_emit import OnnxEmissionMixin

from .frontend_onnx_common import (
    _Attention,
    _GeluRegion,
    _GroupNormRegion,
    _LinearBiasRegion,
    _QArgMaxRegion,
    _QBatchMatMulRegion,
    _QConvRegion,
    _QLinearRegion,
    _attribute,
    _broadcast_shape,
    _normalize_axis,
    _onnx_dtype_name,
    _runtime_dtype_for_array,
    _source_name,
    _to_i32_checked,
)


class OnnxRecognitionMixin:
    """Mixin for OnnxCompiler; see frontend_onnx.py."""

    def _recognize_identity_abi_casts(self) -> None:
        """Record source identity Casts before runtime dtype legalization."""

        for index, node in enumerate(self.model.graph.node):
            if node.op_type != "Cast" or not node.input or not node.output:
                continue
            target = _onnx_dtype_name(int(_attribute(node, "to")))
            if (
                target == "bool"
                and self.dtype_of(node.input[0]) == target
                and self.dtype_of(node.output[0]) == target
            ):
                self.identity_bool_casts.add(index)
            elif (
                target == "int64"
                and self.dtype_of(node.input[0]) == target
                and self.dtype_of(node.output[0]) == target
            ):
                self.identity_int64_casts.add(index)

    @staticmethod
    def _byte_range(dtype: str) -> tuple[int, int]:
        return (-128, 127) if dtype == "int8" else (0, 255)

    def _per_tensor_byte_descriptor(
        self,
        *,
        scale_name: str,
        zero_name: str,
        storage_dtype: str,
    ) -> Optional[dict[str, Any]]:
        """Return a runtime-exact affine descriptor, or None for a hybrid edge.

        Canonical byte-domain operators use immutable F32 scale metadata.  A
        non-F32 scale can still be represented by an explicit ONNX
        DequantizeLinear/QuantizeLinear node, but it must not be advertised as
        the canonical W8A8 tensor contract.
        """

        if storage_dtype not in {"int8", "uint8"}:
            return None
        scale = self._array(scale_name)
        if scale is None or np.asarray(scale).dtype != np.dtype(np.float32) or np.asarray(scale).size != 1:
            return None
        scale_value = float(np.asarray(scale, dtype=np.float32).reshape(-1)[0])
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            return None
        if zero_name:
            zero = self._array(zero_name)
            if zero is None or _runtime_dtype_for_array(np.asarray(zero)) != storage_dtype or np.asarray(zero).size != 1:
                return None
            zero_value = int(np.asarray(zero).reshape(-1)[0])
        else:
            zero_value = 0
        minimum, maximum = self._byte_range(storage_dtype)
        if zero_value < minimum or zero_value > maximum:
            return None
        return {
            "scheme": "per_tensor",
            "scale": scale_value,
            "zero_point": zero_value,
        }

    def _per_axis_linear_weight(
        self,
        entry,
        *,
        d_in: int,
        d_out: int,
    ) -> Optional[tuple[str, np.ndarray, dict[str, Any]]]:
        dq_index, dq_node = entry
        del dq_index
        if dq_node.op_type != "DequantizeLinear" or len(dq_node.input) < 2:
            return None
        raw_name = self.resolve(dq_node.input[0])
        raw = self._array(raw_name)
        scale = self._array(dq_node.input[1])
        zero_name = self.resolve(dq_node.input[2]) if len(dq_node.input) > 2 and dq_node.input[2] else ""
        if raw is None or scale is None:
            return None
        raw = np.asarray(raw)
        scale = np.asarray(scale)
        storage_dtype = _runtime_dtype_for_array(raw)
        if storage_dtype != "int8" or raw.shape != (d_in, d_out):
            return None
        if scale.dtype != np.dtype(np.float32) or scale.ndim != 1 or scale.shape != (d_out,):
            return None
        normalized_axis = int(_attribute(dq_node, "axis", 1))
        if normalized_axis < 0:
            normalized_axis += raw.ndim
        if normalized_axis != 1:
            return None
        scales = np.asarray(scale, dtype=np.float32)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            return None
        if zero_name:
            zero = self._array(zero_name)
            if zero is None:
                return None
            zero = np.asarray(zero)
            if _runtime_dtype_for_array(zero) != storage_dtype or zero.ndim != 1 or zero.shape != (d_out,):
                return None
            zero_points = zero.astype(np.int64)
        else:
            zero_points = np.zeros(d_out, dtype=np.int64)
        minimum, maximum = self._byte_range(storage_dtype)
        if np.any(zero_points < minimum) or np.any(zero_points > maximum):
            return None
        descriptor = {
            "scheme": "per_axis",
            "axis": 0,
            "scales": [float(value) for value in scales],
            "zero_points": [int(value) for value in zero_points],
        }
        return raw_name, np.ascontiguousarray(raw.T), descriptor

    def _per_axis_gemm_weight(
        self,
        entry,
        *,
        d_in: int,
        d_out: int,
    ) -> Optional[tuple[str, np.ndarray, dict[str, Any]]]:
        """Validate a canonical S8 Gemm B stored physically as [out,in]."""

        _dq_index, dq_node = entry
        if dq_node.op_type != "DequantizeLinear" or len(dq_node.input) < 2:
            return None
        raw_name = self.resolve(dq_node.input[0])
        raw = self._array(raw_name)
        scale = self._array(dq_node.input[1])
        zero_name = (
            self.resolve(dq_node.input[2])
            if len(dq_node.input) > 2 and dq_node.input[2]
            else ""
        )
        if raw is None or scale is None:
            return None
        raw = np.asarray(raw)
        scales = np.asarray(scale)
        if (
            raw.dtype != np.dtype(np.int8)
            or raw.shape != (d_out, d_in)
            or scales.dtype != np.dtype(np.float32)
            or scales.ndim != 1
            or scales.shape != (d_out,)
        ):
            return None
        axis = int(_attribute(dq_node, "axis", 1))
        axis = axis + raw.ndim if axis < 0 else axis
        if axis != 0 or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            return None
        if zero_name:
            zero = self._array(zero_name)
            if (
                zero is None
                or np.asarray(zero).dtype != np.dtype(np.int8)
                or np.asarray(zero).shape != (d_out,)
            ):
                return None
            zero_points = np.asarray(zero, dtype=np.int64)
        else:
            zero_points = np.zeros(d_out, dtype=np.int64)
        descriptor = {
            "scheme": "per_axis",
            "axis": 0,
            "scales": [float(value) for value in scales],
            "zero_points": [int(value) for value in zero_points],
        }
        return raw_name, np.ascontiguousarray(raw), descriptor

    def _per_axis_conv_weight(
        self,
        entry,
        *,
        input_channels: int,
        output_channels: int,
    ) -> Optional[tuple[str, np.ndarray, dict[str, Any]]]:
        """Validate S8 OIHW storage and normalize it to physical OHWI."""

        _dq_index, dq_node = entry
        if dq_node.op_type != "DequantizeLinear" or len(dq_node.input) < 2:
            return None
        raw_name = self.resolve(dq_node.input[0])
        raw = self._array(raw_name)
        scale = self._array(dq_node.input[1])
        zero_name = (
            self.resolve(dq_node.input[2])
            if len(dq_node.input) > 2 and dq_node.input[2]
            else ""
        )
        if raw is None or scale is None:
            return None
        raw = np.asarray(raw)
        scales = np.asarray(scale)
        if (
            raw.dtype != np.dtype(np.int8)
            or raw.ndim != 4
            or raw.shape[0] != output_channels
            or raw.shape[1] != input_channels
            or scales.dtype != np.dtype(np.float32)
            or scales.shape != (output_channels,)
        ):
            return None
        axis = int(_attribute(dq_node, "axis", 1))
        axis = axis + raw.ndim if axis < 0 else axis
        if axis != 0 or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            return None
        if zero_name:
            zero = self._array(zero_name)
            if (
                zero is None
                or np.asarray(zero).dtype != np.dtype(np.int8)
                or np.asarray(zero).shape != (output_channels,)
            ):
                return None
            zero_points = np.asarray(zero, dtype=np.int64)
        else:
            zero_points = np.zeros(output_channels, dtype=np.int64)
        descriptor = {
            "scheme": "per_axis",
            "axis": 0,
            "scales": [float(value) for value in scales],
            "zero_points": [int(value) for value in zero_points],
        }
        return (
            raw_name,
            np.ascontiguousarray(np.transpose(raw, (0, 2, 3, 1))),
            descriptor,
        )

    def _per_axis_i32_bias(
        self,
        entry,
        *,
        d_out: int,
        accumulator_scales: np.ndarray,
    ) -> Optional[tuple[str, np.ndarray]]:
        """Return raw Gemm bias only when its DQ descriptor is accumulator-exact."""

        _dq_index, dq_node = entry
        if dq_node.op_type != "DequantizeLinear" or len(dq_node.input) < 2:
            return None
        raw_name = self.resolve(dq_node.input[0])
        raw = self._array(raw_name)
        scale = self._array(dq_node.input[1])
        zero_name = (
            self.resolve(dq_node.input[2])
            if len(dq_node.input) > 2 and dq_node.input[2]
            else ""
        )
        if raw is None or scale is None:
            return None
        raw = np.asarray(raw)
        scales = np.asarray(scale)
        axis = int(_attribute(dq_node, "axis", 1))
        axis = axis + raw.ndim if axis < 0 else axis
        if (
            raw.dtype != np.dtype(np.int32)
            or raw.shape != (d_out,)
            or scales.dtype != np.dtype(np.float32)
            or scales.shape != (d_out,)
            or axis != 0
            or not np.all(np.isfinite(scales))
            or np.any(scales <= 0)
            or not np.array_equal(scales, accumulator_scales)
        ):
            return None
        if zero_name:
            zero = self._array(zero_name)
            if (
                zero is None
                or np.asarray(zero).dtype != np.dtype(np.int32)
                or np.asarray(zero).shape != (d_out,)
                or np.any(np.asarray(zero) != 0)
            ):
                return None
        return raw_name, np.ascontiguousarray(raw)

    def _qlinear_accumulator_is_safe(
        self,
        *,
        input_dtype: str,
        input_descriptor: Mapping[str, Any],
        weight_out_in: np.ndarray,
        weight_descriptor: Mapping[str, Any],
        bias_i32: np.ndarray,
    ) -> bool:
        """Prove the exact centered I32 row-accumulator bound."""

        d_out = int(weight_out_in.shape[0])
        input_minimum, input_maximum = self._byte_range(input_dtype)
        input_zero = int(input_descriptor["zero_point"])
        maximum_input_magnitude = max(
            abs(input_minimum - input_zero),
            abs(input_maximum - input_zero),
        )
        weight_zero = np.asarray(
            weight_descriptor["zero_points"], dtype=np.int64,
        ).reshape(d_out, 1)
        centered_weight = weight_out_in.astype(np.int64) - weight_zero
        product_bounds = maximum_input_magnitude * np.sum(
            np.abs(centered_weight), axis=1, dtype=np.int64,
        )
        accumulator_bounds = product_bounds + np.abs(
            bias_i32.astype(np.int64)
        )
        return bool(np.all(accumulator_bounds <= 2**31 - 1))

    def _qbatch_matmul_geometry(
        self,
        left_shape: list[int | str],
        right_shape: list[int | str],
        output_shape: list[int | str],
        *,
        source_node: str,
    ) -> Optional[list[int]]:
        if (
            len(left_shape) < 2
            or len(right_shape) < 2
            or len(left_shape) > 8
            or len(right_shape) > 8
            or not isinstance(left_shape[-1], int)
            or isinstance(left_shape[-1], bool)
            or left_shape[-1] <= 0
            or left_shape[-1] != right_shape[-2]
        ):
            return None
        try:
            batch = _broadcast_shape(
                left_shape[:-2], right_shape[:-2], node=source_node
            )
        except ExporterError:
            return None
        expected = [*batch, left_shape[-2], right_shape[-1]]
        return expected if expected == output_shape else None

    def _qbatch_accumulator_is_safe(
        self,
        *,
        k: int,
        left_dtype: str,
        left_descriptor: Mapping[str, Any],
        right_dtype: str,
        right_descriptor: Mapping[str, Any],
    ) -> bool:
        left_minimum, left_maximum = self._byte_range(left_dtype)
        right_minimum, right_maximum = self._byte_range(right_dtype)
        left_zero = int(left_descriptor["zero_point"])
        right_zero = int(right_descriptor["zero_point"])
        maximum_left = max(
            abs(left_minimum - left_zero), abs(left_maximum - left_zero)
        )
        maximum_right = max(
            abs(right_minimum - right_zero), abs(right_maximum - right_zero)
        )
        return k * maximum_left * maximum_right <= 2**31 - 1

    @staticmethod
    def _product_multipliers_are_safe(
        left_descriptor: Mapping[str, Any],
        product_scales: Iterable[float],
        output_descriptor: Mapping[str, Any],
    ) -> bool:
        for scale in product_scales:
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                product_scale = np.multiply(
                    np.float32(left_descriptor["scale"]),
                    np.float32(scale),
                    dtype=np.float32,
                )
                multiplier = np.divide(
                    product_scale,
                    np.float32(output_descriptor["scale"]),
                    dtype=np.float32,
                )
            if not bool(np.isfinite(multiplier) and multiplier > 0):
                return False
        return True

    def _register_activation_quantization(
        self,
        name: str,
        descriptor: Mapping[str, Any],
    ) -> bool:
        resolved = self.resolve(name)
        canonical = dict(descriptor)
        previous = self.activation_quantization.get(resolved)
        if previous is not None and previous != canonical:
            return False
        self.activation_quantization[resolved] = canonical
        return True

    def _recognize_w8a8(self) -> None:
        """Recognize bounded, canonical ONNX QDQ activation islands.

        This pass deliberately accepts only descriptors that map one-for-one
        to the strict quantized runtime contracts.  Everything else is left
        for explicit Q/DQ lowering, keeping preserve mode semantic and
        ensuring require-w8a8 cannot succeed on a partial island.
        """

        producers = self._producer_map()
        consumers = self._single_consumer_map()

        # A DQ node may fan out to several independently collapsible regions.
        # Track which consumers stop needing each decoded value, then remove
        # the DQ only when every live use has been replaced.  This keeps a
        # mixed recognized/unrecognized fan-out valid instead of publishing a
        # dangling float edge.
        collapsed_dq_consumers: dict[int, set[int]] = defaultdict(set)

        # Canonical QDQ MatMul/Gemm regions.  A constant S8 RHS becomes
        # QLinear/QGemm; two dynamic byte operands become QBatchMatMul.
        for quantize_index, quantize in enumerate(self.model.graph.node):
            if (
                quantize.op_type != "QuantizeLinear"
                or len(quantize.input) < 2
                or not quantize.output
            ):
                continue
            source_value = self.resolve(quantize.input[0])
            source_entry = producers.get(source_value)

            output_zero = (
                self.resolve(quantize.input[2])
                if len(quantize.input) > 2 and quantize.input[2]
                else ""
            )
            output_dtype = self.dtype_of(quantize.output[0])
            output_descriptor = self._per_tensor_byte_descriptor(
                scale_name=self.resolve(quantize.input[1]),
                zero_name=output_zero,
                storage_dtype=output_dtype,
            )
            if (
                output_dtype != "uint8"
                or output_descriptor is None
                or source_value in self.graph_output_names
            ):
                continue

            # DQ(U8 NCHW activation), DQ(S8 axis-0 OIHW weight), exact Conv,
            # and Q(U8 NCHW) become byte-preserving layout transposes around
            # one canonical NHWC/OHWI QConv2D.
            if source_entry and source_entry[1].op_type == "Conv":
                conv_index, conv = source_entry
                entries = [
                    producers.get(self.resolve(name)) for name in conv.input
                ]
                if (
                    len(conv.input) != 2
                    or any(
                        entry is None
                        or entry[1].op_type != "DequantizeLinear"
                        or len(entry[1].input) < 2
                        for entry in entries
                    )
                ):
                    continue
                activation_entry, weight_entry = entries
                assert activation_entry and weight_entry
                activation_index, activation_dq = activation_entry
                weight_index, weight_dq = weight_entry
                raw_input = self.resolve(activation_dq.input[0])
                input_shape = self.shape_of(raw_input)
                output_shape = self.shape_of(quantize.output[0])
                conv_shape = self.shape_of(conv.output[0])
                auto_pad = _attribute(conv, "auto_pad", b"NOTSET")
                if isinstance(auto_pad, bytes):
                    auto_pad = auto_pad.decode("utf-8")
                pads = [
                    int(value)
                    for value in _attribute(conv, "pads", [0, 0, 0, 0])
                ]
                strides = [
                    int(value)
                    for value in _attribute(conv, "strides", [1, 1])
                ]
                dilations = [
                    int(value)
                    for value in _attribute(conv, "dilations", [1, 1])
                ]
                batch_dimension = input_shape[0] if input_shape else None
                if (
                    len(input_shape) != 4
                    or len(output_shape) != 4
                    or conv_shape != output_shape
                    or input_shape[0] != output_shape[0]
                    or not (
                        (
                            isinstance(batch_dimension, int)
                            and not isinstance(batch_dimension, bool)
                            and batch_dimension > 0
                        )
                        or (
                            isinstance(batch_dimension, str)
                            and bool(batch_dimension)
                        )
                    )
                    or any(
                        isinstance(dimension, bool)
                        or not isinstance(dimension, int)
                        or dimension <= 0
                        for dimension in input_shape[1:] + output_shape[1:]
                    )
                    or auto_pad not in {"", "NOTSET"}
                    or pads != [1, 1, 1, 1]
                    or strides not in ([1, 1], [2, 2])
                    or dilations != [1, 1]
                    or int(_attribute(conv, "group", 1)) != 1
                    or consumers.get(conv.output[0], []) != [quantize_index]
                ):
                    continue
                input_dtype = self.dtype_of(raw_input)
                input_descriptor = self._per_tensor_byte_descriptor(
                    scale_name=self.resolve(activation_dq.input[1]),
                    zero_name=(
                        self.resolve(activation_dq.input[2])
                        if len(activation_dq.input) > 2
                        and activation_dq.input[2]
                        else ""
                    ),
                    storage_dtype=input_dtype,
                )
                weight = self._per_axis_conv_weight(
                    weight_entry,
                    input_channels=input_shape[1],
                    output_channels=output_shape[1],
                )
                if (
                    self._array(raw_input) is not None
                    or input_dtype != "uint8"
                    or input_descriptor is None
                    or weight is None
                ):
                    continue
                raw_weight, weight_ohwi, weight_descriptor = weight
                kernel_shape = [
                    int(value)
                    for value in _attribute(
                        conv,
                        "kernel_shape",
                        [weight_ohwi.shape[1], weight_ohwi.shape[2]],
                    )
                ]
                if kernel_shape != [
                    weight_ohwi.shape[1],
                    weight_ohwi.shape[2],
                ]:
                    continue
                expected_height = (
                    input_shape[2]
                    + pads[0]
                    + pads[2]
                    - dilations[0] * (kernel_shape[0] - 1)
                    - 1
                ) // strides[0] + 1
                expected_width = (
                    input_shape[3]
                    + pads[1]
                    + pads[3]
                    - dilations[1] * (kernel_shape[1] - 1)
                    - 1
                ) // strides[1] + 1
                if output_shape != [
                    input_shape[0],
                    weight_ohwi.shape[0],
                    expected_height,
                    expected_width,
                ]:
                    continue
                bias_i32 = np.zeros(weight_ohwi.shape[0], dtype=np.int32)
                if not self._qlinear_accumulator_is_safe(
                    input_dtype=input_dtype,
                    input_descriptor=input_descriptor,
                    weight_out_in=weight_ohwi.reshape(
                        weight_ohwi.shape[0], -1
                    ),
                    weight_descriptor=weight_descriptor,
                    bias_i32=bias_i32,
                ) or not self._product_multipliers_are_safe(
                    input_descriptor,
                    weight_descriptor["scales"],
                    output_descriptor,
                ):
                    continue
                if not self._register_activation_quantization(
                    raw_input, input_descriptor
                ) or not self._register_activation_quantization(
                    quantize.output[0], output_descriptor
                ):
                    continue
                params = {
                    "stride": strides,
                    "dilation": dilations,
                    "groups": 1,
                    "pads": pads,
                    "padding": pads[:2],
                    "data_layout": "NHWC",
                    "weight_layout": "OHWI",
                }
                indices = {
                    activation_index,
                    weight_index,
                    conv_index,
                    quantize_index,
                }
                region = _QConvRegion(
                    source_node=_source_name(quantize, quantize_index),
                    raw_input=raw_input,
                    raw_weight=raw_weight,
                    output=quantize.output[0],
                    weight_ohwi=weight_ohwi,
                    bias_i32=bias_i32,
                    input_quantization=input_descriptor,
                    weight_quantization=weight_descriptor,
                    output_quantization=output_descriptor,
                    output_dtype=output_dtype,
                    input_nhwc_shape=[
                        input_shape[0],
                        input_shape[2],
                        input_shape[3],
                        input_shape[1],
                    ],
                    output_nchw_shape=output_shape,
                    output_nhwc_shape=[
                        output_shape[0],
                        output_shape[2],
                        output_shape[3],
                        output_shape[1],
                    ],
                    params=params,
                    skip=frozenset(indices),
                )
                self.qconv_replacements[quantize_index] = region
                self.w8a8_skip.add(conv_index)
                collapsed_dq_consumers[activation_index].add(conv_index)
                collapsed_dq_consumers[weight_index].add(conv_index)
                self.features["w8a8_qconv2d"].append({
                    "source_node": region.source_node
                })
                continue

            # DQ(U8 activation), DQ(S8 axis-0 [out,in] weight), and
            # DQ(I32 axis-0 bias) -> exact Gemm -> Q(U8).
            if (
                source_entry
                and source_entry[1].op_type == "Gemm"
                and len(source_entry[1].input) == 3
                and all(source_entry[1].input)
            ):
                gemm_index, gemm = source_entry
                attributes = {
                    "transA": int(_attribute(gemm, "transA", 0)),
                    "transB": int(_attribute(gemm, "transB", 0)),
                    "alpha": float(_attribute(gemm, "alpha", 1.0)),
                    "beta": float(_attribute(gemm, "beta", 1.0)),
                }
                if attributes != {
                    "transA": 0,
                    "transB": 1,
                    "alpha": 1.0,
                    "beta": 1.0,
                }:
                    continue
                entries = [
                    producers.get(self.resolve(name)) for name in gemm.input
                ]
                if any(
                    entry is None
                    or entry[1].op_type != "DequantizeLinear"
                    or len(entry[1].input) < 2
                    for entry in entries
                ):
                    continue
                activation_entry, weight_entry, bias_entry = entries
                assert activation_entry and weight_entry and bias_entry
                activation_index, activation_dq = activation_entry
                weight_index, weight_dq = weight_entry
                bias_index, bias_dq = bias_entry
                raw_input = self.resolve(activation_dq.input[0])
                input_shape = self.shape_of(raw_input)
                output_shape = self.shape_of(quantize.output[0])
                gemm_shape = self.shape_of(gemm.output[0])
                if (
                    len(input_shape) != 2
                    or len(output_shape) != 2
                    or gemm_shape != output_shape
                    or input_shape[0] != output_shape[0]
                    or not isinstance(input_shape[1], int)
                    or not isinstance(output_shape[1], int)
                    or input_shape[1] <= 0
                    or output_shape[1] <= 0
                    or consumers.get(gemm.output[0], []) != [quantize_index]
                    or gemm.output[0] in self.graph_output_names
                ):
                    continue
                d_in = input_shape[1]
                d_out = output_shape[1]
                input_dtype = self.dtype_of(raw_input)
                input_descriptor = self._per_tensor_byte_descriptor(
                    scale_name=self.resolve(activation_dq.input[1]),
                    zero_name=(
                        self.resolve(activation_dq.input[2])
                        if len(activation_dq.input) > 2
                        and activation_dq.input[2]
                        else ""
                    ),
                    storage_dtype=input_dtype,
                )
                weight = self._per_axis_gemm_weight(
                    weight_entry, d_in=d_in, d_out=d_out
                )
                if (
                    self._array(raw_input) is not None
                    or input_dtype != "uint8"
                    or input_descriptor is None
                    or weight is None
                ):
                    continue
                raw_weight, weight_out_in, weight_descriptor = weight
                accumulator_scales = np.multiply(
                    np.float32(input_descriptor["scale"]),
                    np.asarray(weight_descriptor["scales"], dtype=np.float32),
                    dtype=np.float32,
                )
                bias = self._per_axis_i32_bias(
                    bias_entry,
                    d_out=d_out,
                    accumulator_scales=accumulator_scales,
                )
                if bias is None:
                    continue
                _raw_bias, bias_i32 = bias
                if not self._qlinear_accumulator_is_safe(
                    input_dtype=input_dtype,
                    input_descriptor=input_descriptor,
                    weight_out_in=weight_out_in,
                    weight_descriptor=weight_descriptor,
                    bias_i32=bias_i32,
                ):
                    continue
                if not self._register_activation_quantization(
                    raw_input, input_descriptor
                ) or not self._register_activation_quantization(
                    quantize.output[0], output_descriptor
                ):
                    continue
                indices = {
                    activation_index,
                    weight_index,
                    bias_index,
                    gemm_index,
                    quantize_index,
                }
                region = _QLinearRegion(
                    source_node=_source_name(quantize, quantize_index),
                    op_type="QGemm",
                    source_op="QDQ-Gemm-QuantizeLinear",
                    raw_input=raw_input,
                    raw_weight=raw_weight,
                    output=quantize.output[0],
                    weight_out_in=weight_out_in,
                    bias_i32=bias_i32,
                    input_quantization=input_descriptor,
                    weight_quantization=weight_descriptor,
                    output_quantization=output_descriptor,
                    output_dtype=output_dtype,
                    skip=frozenset(indices),
                )
                self.qlinear_replacements[quantize_index] = region
                self.w8a8_skip.add(gemm_index)
                for dq_index in (
                    activation_index,
                    weight_index,
                    bias_index,
                ):
                    collapsed_dq_consumers[dq_index].add(gemm_index)
                self.features["w8a8_qgemm"].append({
                    "source_node": region.source_node
                })
                continue

            add_index = None
            bias_name = ""
            if source_entry and source_entry[1].op_type == "Add" and len(source_entry[1].input) == 2:
                add_index, add_node = source_entry
                matmul_entry = None
                for matmul_side, bias_side in ((0, 1), (1, 0)):
                    candidate = producers.get(self.resolve(add_node.input[matmul_side]))
                    if candidate and candidate[1].op_type == "MatMul":
                        matmul_entry = candidate
                        bias_name = self.resolve(add_node.input[bias_side])
                        break
            elif source_entry and source_entry[1].op_type == "MatMul":
                matmul_entry = source_entry
            else:
                matmul_entry = None
            if not matmul_entry:
                continue
            matmul_index, matmul = matmul_entry
            if len(matmul.input) != 2:
                continue
            activation_entry = producers.get(self.resolve(matmul.input[0]))
            weight_entry = producers.get(self.resolve(matmul.input[1]))
            if not activation_entry or not weight_entry:
                continue
            activation_index, activation_dq = activation_entry
            weight_index, weight_dq = weight_entry
            if activation_dq.op_type != "DequantizeLinear" or weight_dq.op_type != "DequantizeLinear":
                continue
            if len(activation_dq.input) < 2 or len(weight_dq.input) < 2:
                continue
            raw_input = self.resolve(activation_dq.input[0])
            raw_right = self.resolve(weight_dq.input[0])
            input_shape = self.shape_of(raw_input)
            matmul_shape = self.shape_of(matmul.output[0])
            output_shape = self.shape_of(quantize.output[0])
            if not input_shape or not matmul_shape or matmul_shape != output_shape:
                continue
            d_in = input_shape[-1]
            d_out = matmul_shape[-1]
            if (
                not isinstance(d_in, int)
                or isinstance(d_in, bool)
                or d_in <= 0
            ):
                continue
            expected_matmul_consumer = add_index if add_index is not None else quantize_index
            if consumers.get(matmul.output[0], []) != [expected_matmul_consumer]:
                continue
            if add_index is not None and consumers.get(self.model.graph.node[add_index].output[0], []) != [quantize_index]:
                continue
            if matmul.output[0] in self.graph_output_names:
                continue

            input_dtype = self.dtype_of(raw_input)
            input_descriptor = self._per_tensor_byte_descriptor(
                scale_name=self.resolve(activation_dq.input[1]),
                zero_name=self.resolve(activation_dq.input[2]) if len(activation_dq.input) > 2 and activation_dq.input[2] else "",
                storage_dtype=input_dtype,
            )
            # Both decoded operands are dynamic byte tensors: preserve exact
            # ONNX rank-N MatMul broadcasting as a QBatchMatMul descriptor.
            if self._array(raw_input) is None and self._array(raw_right) is None:
                if add_index is not None:
                    continue
                right_dtype = self.dtype_of(raw_right)
                right_descriptor = self._per_tensor_byte_descriptor(
                    scale_name=self.resolve(weight_dq.input[1]),
                    zero_name=(
                        self.resolve(weight_dq.input[2])
                        if len(weight_dq.input) > 2 and weight_dq.input[2]
                        else ""
                    ),
                    storage_dtype=right_dtype,
                )
                right_shape = self.shape_of(raw_right)
                expected_shape = self._qbatch_matmul_geometry(
                    input_shape,
                    right_shape,
                    output_shape,
                    source_node=_source_name(matmul, matmul_index),
                )
                if (
                    input_descriptor is None
                    or right_descriptor is None
                    or expected_shape is None
                    or not self._product_multipliers_are_safe(
                        input_descriptor,
                        [right_descriptor["scale"]],
                        output_descriptor,
                    )
                    or not self._qbatch_accumulator_is_safe(
                        k=d_in,
                        left_dtype=input_dtype,
                        left_descriptor=input_descriptor,
                        right_dtype=right_dtype,
                        right_descriptor=right_descriptor,
                    )
                ):
                    continue
                if not self._register_activation_quantization(
                    raw_input, input_descriptor
                ) or not self._register_activation_quantization(
                    raw_right, right_descriptor
                ) or not self._register_activation_quantization(
                    quantize.output[0], output_descriptor
                ):
                    continue
                indices = {
                    activation_index,
                    weight_index,
                    matmul_index,
                    quantize_index,
                }
                region = _QBatchMatMulRegion(
                    source_node=_source_name(quantize, quantize_index),
                    raw_a=raw_input,
                    raw_b=raw_right,
                    output=quantize.output[0],
                    a_quantization=input_descriptor,
                    b_quantization=right_descriptor,
                    output_quantization=output_descriptor,
                    output_dtype=output_dtype,
                    output_shape=expected_shape,
                    skip=frozenset(indices),
                )
                self.qbatch_matmul_replacements[quantize_index] = region
                self.w8a8_skip.add(matmul_index)
                collapsed_dq_consumers[activation_index].add(matmul_index)
                collapsed_dq_consumers[weight_index].add(matmul_index)
                self.features["w8a8_qbatch_matmul"].append({
                    "source_node": region.source_node
                })
                continue

            if (
                not isinstance(d_out, int)
                or isinstance(d_out, bool)
                or d_out <= 0
            ):
                continue
            weight = self._per_axis_linear_weight(
                weight_entry, d_in=d_in, d_out=d_out
            )

            if (
                input_dtype != "uint8"
                or self._array(raw_input) is not None
                or input_descriptor is None
                or weight is None
                or matmul_shape[:-1] != input_shape[:-1]
            ):
                continue
            raw_weight, weight_out_in, weight_descriptor = weight

            accumulator_scales = np.asarray(weight_descriptor["scales"], dtype=np.float32)
            accumulator_scales = np.multiply(
                np.float32(input_descriptor["scale"]), accumulator_scales,
                dtype=np.float32,
            )
            if not np.all(np.isfinite(accumulator_scales)) or np.any(accumulator_scales <= 0):
                continue
            if bias_name:
                bias = self._array(bias_name)
                if bias is None or np.asarray(bias).dtype != np.dtype(np.float32) or np.asarray(bias).shape != (d_out,):
                    continue
                bias = np.asarray(bias, dtype=np.float32)
                ratios = np.divide(bias, accumulator_scales, dtype=np.float32)
                rounded = np.rint(ratios)
                if not np.all(np.isfinite(rounded)) or np.any(rounded < -(2**31)) or np.any(rounded > 2**31 - 1):
                    continue
                bias_i32 = rounded.astype(np.int32)
                reconstructed = np.multiply(bias_i32.astype(np.float32), accumulator_scales, dtype=np.float32)
                if not np.array_equal(reconstructed, bias):
                    continue
            else:
                bias_i32 = np.zeros(d_out, dtype=np.int32)

            if not self._qlinear_accumulator_is_safe(
                input_dtype=input_dtype,
                input_descriptor=input_descriptor,
                weight_out_in=weight_out_in,
                weight_descriptor=weight_descriptor,
                bias_i32=bias_i32,
            ):
                continue

            if not self._register_activation_quantization(raw_input, input_descriptor):
                continue
            if not self._register_activation_quantization(quantize.output[0], output_descriptor):
                continue
            indices = {activation_index, weight_index, matmul_index, quantize_index}
            if add_index is not None:
                indices.add(add_index)
            region = _QLinearRegion(
                source_node=_source_name(quantize, quantize_index),
                op_type="QLinear",
                source_op="QDQ-MatMul-QuantizeLinear",
                raw_input=raw_input,
                raw_weight=raw_weight,
                output=quantize.output[0],
                weight_out_in=weight_out_in,
                bias_i32=np.ascontiguousarray(bias_i32),
                input_quantization=input_descriptor,
                weight_quantization=weight_descriptor,
                output_quantization=output_descriptor,
                output_dtype=output_dtype,
                skip=frozenset(indices),
            )
            self.qlinear_replacements[quantize_index] = region
            self.w8a8_skip.add(matmul_index)
            if add_index is not None:
                self.w8a8_skip.add(add_index)
            collapsed_dq_consumers[activation_index].add(matmul_index)
            collapsed_dq_consumers[weight_index].add(matmul_index)
            self.features["w8a8_qlinear"].append({"source_node": region.source_node})

        for dq_index, replaced_consumers in collapsed_dq_consumers.items():
            dq_node = self.model.graph.node[dq_index]
            live_consumers = set(consumers.get(dq_node.output[0], ()))
            if (
                live_consumers
                and live_consumers <= replaced_consumers
                and dq_node.output[0] not in self.graph_output_names
            ):
                self.w8a8_skip.add(dq_index)

        # DQ(per-tensor byte logits) -> terminal first-tie ArgMax.  Positive
        # affine scaling preserves order, so comparing the stored bytes is
        # exactly equivalent and avoids a decoded logits tensor.
        for argmax_index, argmax in enumerate(self.model.graph.node):
            if argmax.op_type != "ArgMax" or not argmax.input or not argmax.output:
                continue
            dq_entry = producers.get(self.resolve(argmax.input[0]))
            if not dq_entry or dq_entry[1].op_type != "DequantizeLinear":
                continue
            dq_index, dq_node = dq_entry
            if len(dq_node.input) < 2 or consumers.get(dq_node.output[0], []) != [argmax_index]:
                continue
            if dq_node.output[0] in self.graph_output_names:
                continue
            if consumers.get(argmax.output[0], []) or argmax.output[0] not in self.graph_output_names:
                continue
            if int(_attribute(argmax, "keepdims", 1)) != 0 or int(_attribute(argmax, "select_last_index", 0)) != 0:
                continue
            raw_input = self.resolve(dq_node.input[0])
            input_shape = self.shape_of(raw_input)
            if len(input_shape) < 2 or len(input_shape) > 8:
                continue
            axis = _normalize_axis(
                int(_attribute(argmax, "axis", 0)), len(input_shape),
                node=_source_name(argmax, argmax_index),
            )
            expected_shape = [*input_shape[:axis], *input_shape[axis + 1:]]
            if not expected_shape or self.shape_of(argmax.output[0]) != expected_shape:
                continue
            descriptor = self._per_tensor_byte_descriptor(
                scale_name=self.resolve(dq_node.input[1]),
                zero_name=self.resolve(dq_node.input[2]) if len(dq_node.input) > 2 and dq_node.input[2] else "",
                storage_dtype=self.dtype_of(raw_input),
            )
            if descriptor is None or not self._register_activation_quantization(raw_input, descriptor):
                continue
            region = _QArgMaxRegion(
                source_node=_source_name(argmax, argmax_index),
                raw_input=raw_input,
                output=argmax.output[0],
                axis=axis,
                input_quantization=descriptor,
                skip=frozenset({dq_index, argmax_index}),
            )
            self.qargmax_replacements[argmax_index] = region
            self.w8a8_skip.update(region.skip)
            self.features["w8a8_qargmax"].append({"source_node": region.source_node})

    def _scalar_constant(self, name: str, expected: Optional[float] = None, *, tolerance: float = 1e-6):
        value = self._array(name)
        if value is None or np.asarray(value).size != 1:
            return None
        scalar = float(np.asarray(value).reshape(-1)[0])
        if expected is not None and not math.isclose(scalar, expected, rel_tol=tolerance, abs_tol=tolerance):
            return None
        return scalar

    def _mul_constant_and_value(self, node, expected: Optional[float] = None):
        if node.op_type != "Mul" or len(node.input) != 2:
            return None
        for constant_index, value_index in ((0, 1), (1, 0)):
            scalar = self._scalar_constant(node.input[constant_index], expected)
            if scalar is not None:
                return scalar, self.resolve(node.input[value_index])
        return None

    def _recognize_exact_gelu(self) -> None:
        """Collapse the normative ONNX erf GELU expansion, never an approximation."""
        producers = self._producer_map()
        consumers = self._single_consumer_map()
        for final_index, final in enumerate(self.model.graph.node):
            if final.op_type != "Mul" or len(final.input) != 2 or final_index in self.folded:
                continue
            candidates: list[tuple[str, str, set[int]]] = []
            half = self._mul_constant_and_value(final, 0.5)
            if half is not None:
                product_entry = producers.get(half[1])
                if not product_entry or product_entry[1].op_type != "Mul":
                    continue
                product_index, product = product_entry
                for x_side, add_side in ((0, 1), (1, 0)):
                    add_value = self.resolve(product.input[add_side])
                    add_entry = producers.get(add_value)
                    if add_entry and add_entry[1].op_type == "Add":
                        candidates.append((
                            self.resolve(product.input[x_side]),
                            add_value,
                            {final_index, product_index},
                        ))
            else:
                # The other exact associations are
                # (0.5 * X) * (1 + erf(...)) and
                # X * (0.5 * (1 + erf(...))).  Keep both candidates until the
                # Erf argument proves which Add is the GELU factor; X itself
                # may legitimately also be produced by an Add.
                for inner_side, outer_side in ((0, 1), (1, 0)):
                    inner_entry = producers.get(
                        self.resolve(final.input[inner_side])
                    )
                    if not inner_entry or inner_entry[1].op_type != "Mul":
                        continue
                    inner_value = self._mul_constant_and_value(
                        inner_entry[1], 0.5
                    )
                    if inner_value is None:
                        continue
                    dynamic_inner = inner_value[1]
                    outer_value = self.resolve(final.input[outer_side])
                    for x, add_value in (
                        (dynamic_inner, outer_value),
                        (outer_value, dynamic_inner),
                    ):
                        add_entry = producers.get(add_value)
                        if add_entry and add_entry[1].op_type == "Add":
                            candidates.append((
                                x,
                                add_value,
                                {final_index, inner_entry[0]},
                            ))

            for x, add_value, candidate_indices in candidates:
                indices = set(candidate_indices)
                add_index, add = producers[add_value]
                indices.add(add_index)
                erf_value = None
                for erf_side, one_side in ((0, 1), (1, 0)):
                    if self._scalar_constant(add.input[one_side], 1.0) is None:
                        continue
                    erf_entry = producers.get(self.resolve(add.input[erf_side]))
                    if erf_entry and erf_entry[1].op_type == "Erf":
                        erf_value = self.resolve(add.input[erf_side])
                        break
                if erf_value is None:
                    continue
                erf_index, erf = producers[erf_value]
                indices.add(erf_index)
                normalized_entry = producers.get(self.resolve(erf.input[0]))
                if (
                    not normalized_entry
                    or normalized_entry[1].op_type not in {"Div", "Mul"}
                ):
                    continue
                normalized_index, normalized = normalized_entry
                indices.add(normalized_index)
                normalized_x = None
                if normalized.op_type == "Div" and len(normalized.input) == 2:
                    divisor = self._scalar_constant(
                        normalized.input[1], math.sqrt(2.0)
                    )
                    if divisor is not None:
                        normalized_x = self.resolve(normalized.input[0])
                elif normalized.op_type == "Mul":
                    scaled = self._mul_constant_and_value(
                        normalized, 1.0 / math.sqrt(2.0)
                    )
                    if scaled is not None:
                        normalized_x = scaled[1]
                if normalized_x != x:
                    continue
                safe = True
                for region_index in indices:
                    if region_index == final_index:
                        continue
                    for output in self.model.graph.node[region_index].output:
                        if any(
                            consumer not in indices
                            for consumer in consumers.get(output, ())
                        ):
                            safe = False
                if not safe:
                    continue
                region = _GeluRegion(
                    source_node=_source_name(final, final_index),
                    input=x,
                    output=final.output[0],
                    skip=frozenset(indices),
                )
                self.gelu_replacements[final_index] = region
                self.gelu_skip.update(indices)
                self.features["gelu"].append({
                    "source_node": region.source_node,
                    "approximate": "none",
                    "source_pattern": "erf",
                })
                break

    def _binary_dynamic_constant(self, node):
        if node.op_type not in {"Add", "Mul"} or len(node.input) != 2:
            return None
        for dynamic_side, constant_side in ((0, 1), (1, 0)):
            value = self._array(node.input[constant_side])
            if value is not None:
                return self.resolve(node.input[dynamic_side]), self.resolve(node.input[constant_side]), np.asarray(value)
        return None

    def _recognize_linear_bias(self) -> None:
        """Bind `MatMul -> Add(immutable [d_out])` as one biased Linear.

        This is what `Gemm` already spells, and producers differ only in
        whether their exporter emitted the fused form. Lowering the two nodes
        separately is numerically identical in F32 but not in the byte domain:
        an unbound bias is broadcast to the full activation shape, which hides
        its per-output-channel structure from bias folding and leaves the PTQ
        author quantizing the pre-bias accumulator and the bias sum separately
        instead of folding the bias into one I32 accumulator.
        """

        producers = self._producer_map()
        consumers = self._single_consumer_map()
        for index, node in enumerate(self.model.graph.node):
            if node.op_type != "Add" or index in self.folded or len(node.input) != 2:
                continue
            if index in self.groupnorm_skip or index in self.gelu_skip:
                continue
            parts = self._binary_dynamic_constant(node)
            if parts is None:
                continue
            dynamic_name, bias_name, bias = parts
            entry = producers.get(dynamic_name)
            if not entry or entry[1].op_type != "MatMul" or entry[0] in self.folded:
                continue
            matmul_index, matmul = entry
            if matmul_index in self.linear_bias_fusions:
                continue
            # The MatMul result must be consumed only here, otherwise the
            # unbiased value is still live and cannot be replaced.
            if list(consumers.get(matmul.output[0], ())) != [index]:
                continue
            if matmul.output[0] in self.graph_output_names:
                continue
            # Only the static-RHS form lowers to Linear; a dynamic RHS becomes
            # BatchMatMul, which has no bias port.
            weight = self._array(self.resolve(matmul.input[1]))
            if weight is None or np.asarray(weight).ndim != 2:
                continue
            output_shape = self.shape_of(node.output[0])
            width = output_shape[-1] if output_shape else None
            values = np.asarray(bias)
            if not isinstance(width, int) or values.ndim != 1 or values.shape[0] != width:
                continue
            if self.shape_of(matmul.output[0]) != output_shape:
                continue
            self.linear_bias_fusions[matmul_index] = _LinearBiasRegion(
                source_node=_source_name(node, index),
                bias=bias_name,
                output=node.output[0],
                add_index=index,
            )
            self.linear_bias_skip.add(index)
            self.features["linear_bias"].append({
                "source_node": _source_name(node, index),
                "width": int(width),
            })

    def _recognize_group_norm(self) -> None:
        """Recognize PyTorch's reshape/InstanceNormalization GroupNorm export."""
        producers = self._producer_map()
        consumers = self._single_consumer_map()
        for final_index, final in enumerate(self.model.graph.node):
            if final.op_type != "Add" or final_index in self.folded:
                continue
            final_parts = self._binary_dynamic_constant(final)
            if final_parts is None:
                continue
            mul_value, bias_name, bias = final_parts
            mul_entry = producers.get(mul_value)
            if not mul_entry or mul_entry[1].op_type != "Mul":
                continue
            mul_index, mul = mul_entry
            mul_parts = self._binary_dynamic_constant(mul)
            if mul_parts is None:
                continue
            restored_value, weight_name, weight = mul_parts
            restore_entry = producers.get(restored_value)
            if not restore_entry or restore_entry[1].op_type != "Reshape":
                continue
            restore_index, restore = restore_entry
            instance_entry = producers.get(self.resolve(restore.input[0]))
            if not instance_entry or instance_entry[1].op_type != "InstanceNormalization":
                continue
            instance_index, instance = instance_entry
            grouped_entry = producers.get(self.resolve(instance.input[0]))
            if not grouped_entry or grouped_entry[1].op_type != "Reshape":
                continue
            grouped_index, grouped = grouped_entry
            source = self.resolve(grouped.input[0])
            source_shape = self.shape_of(source)
            grouped_shape = self.shape_of(grouped.output[0])
            if len(source_shape) != 4 or len(grouped_shape) != 3:
                continue
            channels = source_shape[1]
            groups = grouped_shape[1]
            if groups <= 0 or channels % groups:
                continue
            instance_scale = self._array(instance.input[1]) if len(instance.input) > 1 else None
            instance_bias = self._array(instance.input[2]) if len(instance.input) > 2 else None
            if (
                instance_scale is None
                or instance_bias is None
                or np.asarray(instance_scale).shape != (groups,)
                or np.asarray(instance_bias).shape != (groups,)
                or not np.allclose(instance_scale, 1.0, rtol=0, atol=0)
                or not np.allclose(instance_bias, 0.0, rtol=0, atol=0)
            ):
                continue
            if np.asarray(weight).size != channels or np.asarray(bias).size != channels:
                continue
            epsilon = float(_attribute(instance, "epsilon", 1e-5))
            if not math.isfinite(epsilon) or epsilon <= 0:
                continue
            indices = {grouped_index, instance_index, restore_index, mul_index, final_index}
            # The restore shape is commonly produced by Shape(source).
            if len(restore.input) > 1:
                shape_entry = producers.get(self.resolve(restore.input[1]))
                if shape_entry and shape_entry[1].op_type == "Shape":
                    indices.add(shape_entry[0])
            safe = True
            for region_index in indices:
                if region_index == final_index:
                    continue
                for output in self.model.graph.node[region_index].output:
                    if any(consumer not in indices for consumer in consumers.get(output, ())):
                        safe = False
            if not safe:
                continue
            derived_weight = f"{weight_name}__groupnorm_channel"
            derived_bias = f"{bias_name}__groupnorm_channel"
            self.arrays[derived_weight] = np.asarray(weight, dtype=np.float32).reshape(channels)
            self.arrays[derived_bias] = np.asarray(bias, dtype=np.float32).reshape(channels)
            self.shape_map[derived_weight] = [channels]
            self.shape_map[derived_bias] = [channels]
            self.dtype_map[derived_weight] = "float32"
            self.dtype_map[derived_bias] = "float32"
            region = _GroupNormRegion(
                source_node=_source_name(final, final_index),
                input=source,
                output=final.output[0],
                weight=derived_weight,
                bias=derived_bias,
                groups=int(groups),
                epsilon=epsilon,
                skip=frozenset(indices),
            )
            self.groupnorm_replacements[final_index] = region
            self.groupnorm_skip.update(indices)
            self.features["group_norm"].append({
                "source_node": region.source_node,
                "groups": region.groups,
                "source_pattern": "reshape-instance-normalization",
            })

    def _linear_input(self, node):
        return self.resolve(node.input[0]) if node.op_type in {"MatMul", "Gemm"} and node.input else None

    def _effective_linear_weight(self, node) -> Optional[np.ndarray]:
        if node.op_type not in {"MatMul", "Gemm"} or len(node.input) < 2:
            return None
        weight = self._array(node.input[1])
        if weight is None or np.asarray(weight).ndim != 2:
            return None
        effective = np.asarray(weight)
        if node.op_type == "Gemm":
            if int(_attribute(node, "transA", 0)) != 0:
                return None
            if int(_attribute(node, "transB", 0)) != 0:
                effective = effective.T
        return effective

    def _unwrap_biased_linear(self, value: str, producers):
        """Return a linear producer and an optional constant-bias Add wrapper."""
        entry = producers.get(self.resolve(value))
        if entry and entry[1].op_type in {"MatMul", "Gemm"}:
            return entry, None
        if not entry or entry[1].op_type != "Add" or len(entry[1].input) != 2:
            return None, None
        for linear_side, bias_side in ((0, 1), (1, 0)):
            linear_entry = producers.get(self.resolve(entry[1].input[linear_side]))
            bias = self._array(entry[1].input[bias_side])
            if not linear_entry or linear_entry[1].op_type not in {"MatMul", "Gemm"} or bias is None:
                continue
            output_shape = self.shape_of(value)
            if output_shape and np.asarray(bias).size in {1, output_shape[-1]}:
                return linear_entry, entry
        return None, None

    def _unwrap_scaled_linear(self, value: str, producers):
        scalar = 1.0
        current = self.resolve(value)
        producer = producers.get(current)
        if producer and producer[1].op_type == "Mul":
            mul = producer[1]
            constants = [(name, self.arrays.get(self.resolve(name))) for name in mul.input]
            constant_positions = [(name, array) for name, array in constants if array is not None and np.asarray(array).size == 1]
            if len(constant_positions) == 1:
                constant_name, array = constant_positions[0]
                scalar = float(np.asarray(array).reshape(-1)[0])
                current = self.resolve(mul.input[1] if mul.input[0] == constant_name else mul.input[0])
                producer = producers.get(current)
            elif producer:
                other_nodes = [producers.get(self.resolve(name)) for name in mul.input]
                if any(item and item[1].op_type in {"MatMul", "Gemm"} for item in other_nodes):
                    return None, None, None, "dynamic scale"
        if not producer or producer[1].op_type not in {"MatMul", "Gemm"}:
            return None, None, None, None
        second = producer[1]
        first_value = self.resolve(second.input[0])
        first_producer = producers.get(first_value)
        if not first_producer or first_producer[1].op_type not in {"MatMul", "Gemm"}:
            return None, None, None, None
        return first_producer[1], second, scalar, None

    def _router_feature(self, node, index: int, producers):
        """Recognize a bounded, all-masked-safe FP32 masked-mean router."""

        if not node.input or not node.output or node.output[0] not in self.graph_output_names:
            return None
        logits_shape = self.shape_of(node.input[0])
        if (
            len(logits_shape) != 2
            or logits_shape[0] != 1
            or not isinstance(logits_shape[1], int)
            or logits_shape[1] <= 1
        ):
            return None
        axis = _normalize_axis(
            int(_attribute(node, "axis", 0)), len(logits_shape),
            node=_source_name(node, index),
        )
        if (
            axis != len(logits_shape) - 1
            or int(_attribute(node, "keepdims", 1)) != 0
            or int(_attribute(node, "select_last_index", 0)) != 0
        ):
            return None

        head_entry, _ = self._unwrap_biased_linear(node.input[0], producers)
        if not head_entry:
            return None
        head = head_entry[1]
        pooled = self.resolve(head.input[0])
        div_entry = producers.get(pooled)
        if not div_entry or div_entry[1].op_type != "Div" or len(div_entry[1].input) != 2:
            return None
        numerator_name = self.resolve(div_entry[1].input[0])
        denominator_name = self.resolve(div_entry[1].input[1])

        # count.clamp_min(1) is required: a raw divide by mask.sum() has
        # undefined all-masked behavior and cannot be advertised as a router.
        clip_entry = producers.get(denominator_name)
        if not clip_entry or clip_entry[1].op_type != "Clip":
            return None
        clip = clip_entry[1]
        minimum = self._array(clip.input[1]) if len(clip.input) > 1 and clip.input[1] else None
        maximum = self._array(clip.input[2]) if len(clip.input) > 2 and clip.input[2] else None
        if minimum is None:
            attribute_minimum = _attribute(clip, "min")
            minimum_value = float(attribute_minimum) if attribute_minimum is not None else None
        elif np.asarray(minimum).size == 1:
            minimum_value = float(np.asarray(minimum).reshape(-1)[0])
        else:
            return None
        if minimum_value != 1.0:
            return None
        if maximum is not None:
            if np.asarray(maximum).size != 1 or float(np.asarray(maximum).reshape(-1)[0]) < 1.0:
                return None
        else:
            attribute_maximum = _attribute(clip, "max")
            if attribute_maximum is not None and float(attribute_maximum) < 1.0:
                return None
        count_name = self.resolve(clip.input[0])
        count_entry = producers.get(count_name)
        if not count_entry or count_entry[1].op_type != "ReduceSum":
            return None
        count = count_entry[1]
        if self._reduction_axes(count) != [1] or int(_attribute(count, "keepdims", 1)) != 1:
            return None
        mask_name = self.resolve(count.input[0])
        mask_shape = self.shape_of(mask_name)
        public_inputs = {
            value.name for value in self.model.graph.input
            if value.name not in self.initializer_names
        }
        if (
            mask_name not in public_inputs
            or mask_shape[:1] != [1]
            or len(mask_shape) != 2
            or self.execution_dtype_of(mask_name) != "float32"
        ):
            return None

        numerator_entry = producers.get(numerator_name)
        if not numerator_entry or numerator_entry[1].op_type != "ReduceSum":
            return None
        numerator = numerator_entry[1]
        if self._reduction_axes(numerator) != [1] or int(_attribute(numerator, "keepdims", 1)) != 0:
            return None
        multiply_entry = producers.get(self.resolve(numerator.input[0]))
        if not multiply_entry or multiply_entry[1].op_type != "Mul":
            return None
        multiply = multiply_entry[1]

        token_name = None
        for mask_side, token_side in ((0, 1), (1, 0)):
            expanded_entry = producers.get(self.resolve(multiply.input[mask_side]))
            if not expanded_entry or expanded_entry[1].op_type != "Unsqueeze":
                continue
            expanded = expanded_entry[1]
            if self.resolve(expanded.input[0]) != mask_name:
                continue
            axes = _attribute(expanded, "axes")
            if axes is None and len(expanded.input) > 1:
                axes_array = self._array(expanded.input[1])
                axes = list(np.asarray(axes_array).reshape(-1)) if axes_array is not None else None
            if [int(value) for value in (axes or [])] != [2]:
                continue
            token_name = self.resolve(multiply.input[token_side])
            break
        if token_name is None:
            return None
        token_shape = self.shape_of(token_name)
        if (
            len(token_shape) != 3
            or token_shape[0] != 1
            or token_shape[1] != mask_shape[1]
            or self.execution_dtype_of(token_name) != "float32"
        ):
            return None
        return {
            "source_node": _source_name(node, index),
            "output": self.names.get(node.output[0]),
            "tie_policy": "first-index",
            "semantic_mask_inputs": [self.names.get(mask_name)],
            "all_masked": "zero-vector-via-clamp-min-one",
        }

    def _recognize_features(self) -> None:
        producers = self._producer_map()
        consumers = self._single_consumer_map()
        for index, node in enumerate(self.model.graph.node):
            if node.op_type != "Add" or len(node.input) != 2:
                continue
            node_name = _source_name(node, index)
            for base_side, delta_side in ((0, 1), (1, 0)):
                base_entry, base_bias_entry = self._unwrap_biased_linear(node.input[base_side], producers)
                if not base_entry:
                    continue
                first, second, scale, error = self._unwrap_scaled_linear(node.input[delta_side], producers)
                if error:
                    raise ExporterError(Diagnostic(
                        "VXLORA_SCALE",
                        f"{node_name} resembles LoRA but uses a dynamic scale",
                        "recognize-lora",
                        source_node=node_name,
                        source_op="Add",
                        constraint="finite compile-time LoRA scale",
                    ))
                if not first or not second:
                    continue
                base = base_entry[1]
                source = self.resolve(base.input[0])
                if self.resolve(first.input[0]) != source:
                    continue
                a_weight = self._effective_linear_weight(first)
                b_weight = self._effective_linear_weight(second)
                base_weight = self._effective_linear_weight(base)
                if any(weight is None or np.asarray(weight).ndim != 2 for weight in (base_weight, a_weight, b_weight)):
                    continue
                d_in, rank = np.asarray(a_weight).shape
                rank_b, d_out = np.asarray(b_weight).shape
                if rank <= 0 or rank_b != rank or np.asarray(base_weight).shape != (d_in, d_out):
                    raise ExporterError(Diagnostic(
                        "VXLORA_SHAPE",
                        f"{node_name} has incompatible base/A/B LoRA dimensions",
                        "recognize-lora",
                        source_node=node_name,
                        constraint="d_in->r->d_out with r > 0",
                    ))
                if not math.isfinite(float(scale)):
                    raise ExporterError(Diagnostic(
                        "VXLORA_SCALE", f"{node_name} LoRA scale is not finite", "recognize-lora", source_node=node_name
                    ))
                if self.shape_of(node.input[0]) != self.shape_of(node.input[1]):
                    raise ExporterError(Diagnostic(
                        "VXLORA_RESIDUAL",
                        f"{node_name} LoRA residual add is broadcast rather than exact-shape",
                        "recognize-lora",
                        source_node=node_name,
                        constraint="exact-shape residual",
                    ))
                self.features["lora"].append({
                    "source_node": node_name,
                    "input": source,
                    "rank": int(rank),
                    "scale": float(scale),
                    "base": base.name or base.output[0],
                    "base_bias": (base_bias_entry[1].name or base_bias_entry[1].output[0]) if base_bias_entry else None,
                    "a": first.name or first.output[0],
                    "b": second.name or second.output[0],
                })
                break

            # Fixed adapter: X + Up(GELU(Down(X))).
            for residual_side, adapter_side in ((0, 1), (1, 0)):
                up_entry, up_bias_entry = self._unwrap_biased_linear(node.input[adapter_side], producers)
                if not up_entry:
                    continue
                activation_entry = producers.get(self.resolve(up_entry[1].input[0]))
                if not activation_entry:
                    continue
                if activation_entry[0] in self.gelu_replacements:
                    activation_input = self.gelu_replacements[activation_entry[0]].input
                elif activation_entry[1].op_type in {"Gelu", "GELU"}:
                    activation_input = self.resolve(activation_entry[1].input[0])
                else:
                    continue
                down_entry, down_bias_entry = self._unwrap_biased_linear(activation_input, producers)
                if not down_entry:
                    continue
                if self.resolve(down_entry[1].input[0]) != self.resolve(node.input[residual_side]):
                    continue
                if self.shape_of(node.input[0]) != self.shape_of(node.input[1]):
                    raise ExporterError(Diagnostic(
                        "VXADAPTER_RESIDUAL",
                        f"{node_name} adapter residual add is broadcast rather than exact-shape",
                        "recognize-adapter",
                        source_node=node_name,
                    ))
                self.features["static_adapter"].append({
                    "source_node": node_name,
                    "input": self.resolve(node.input[residual_side]),
                    "down": down_entry[1].name or down_entry[1].output[0],
                    "up": up_entry[1].name or up_entry[1].output[0],
                    "down_bias": (down_bias_entry[1].name or down_bias_entry[1].output[0]) if down_bias_entry else None,
                    "up_bias": (up_bias_entry[1].name or up_bias_entry[1].output[0]) if up_bias_entry else None,
                })
                break

        for index, node in enumerate(self.model.graph.node):
            if node.op_type != "ArgMax":
                continue
            feature = self._router_feature(node, index, producers)
            if feature is not None:
                self.features["router"].append(feature)

            if node.output:
                dynamic_consumers = consumers.get(node.output[0], [])
                dispatch_consumers = []
                for consumer_index in dynamic_consumers:
                    consumer = self.model.graph.node[consumer_index]
                    if (
                        consumer.op_type in {"Gather", "GatherElements"}
                        and len(consumer.input) > 1
                        and self.resolve(consumer.input[1]) == self.resolve(node.output[0])
                    ):
                        dispatch_consumers.append(consumer_index)
                if dispatch_consumers:
                    labels = ", ".join(
                        _source_name(self.model.graph.node[item], item) for item in dispatch_consumers
                    )
                    raise ExporterError(Diagnostic(
                        "VXROUTER_DISPATCH",
                        f"{_source_name(node, index)} ArgMax feeds data-dependent graph dispatch at {labels}",
                        "recognize-router",
                        source_node=_source_name(node, index),
                        source_op="ArgMax",
                        constraint="router ArgMax must be terminal and host-dispatched",
                    ))

    def _single_consumer_map(self):
        consumers: dict[str, list[int]] = defaultdict(list)
        for index, node in enumerate(self.model.graph.node):
            for name in node.input:
                if name:
                    consumers[name].append(index)
        return consumers

    def _attention_head_path(self, value: str, producers, *, key: bool = False):
        """Return the pre-head-split rank-3 value and structural node indices."""
        current = self.resolve(value)
        entry = producers.get(current)
        if not entry or entry[1].op_type != "Transpose":
            return None
        transpose_index, transpose = entry
        permutation = list(_attribute(transpose, "perm", ()))
        expected = [0, 2, 3, 1] if key else [0, 2, 1, 3]
        if permutation != expected:
            return None
        reshape_entry = producers.get(self.resolve(transpose.input[0]))
        if not reshape_entry or reshape_entry[1].op_type != "Reshape":
            return None
        reshape_index, reshape = reshape_entry
        split_shape = self.shape_of(reshape.output[0])
        base = self.resolve(reshape.input[0])
        base_shape = self.shape_of(base)
        if len(split_shape) != 4 or len(base_shape) != 3:
            return None
        batch, sequence, heads, head_dim = split_shape
        if base_shape != [batch, sequence, heads * head_dim] or heads <= 0 or head_dim <= 0:
            return None
        return {
            "base": base,
            "heads": heads,
            "head_dim": head_dim,
            "indices": {reshape_index, transpose_index},
        }

    def _attention_scale(self, value: str, producers):
        current = self.resolve(value)
        entry = producers.get(current)
        if not entry or entry[1].op_type not in {"Mul", "Div"}:
            return current, 1.0, set()
        index, node = entry
        if len(node.input) != 2:
            return current, 1.0, set()
        left = self._array(node.input[0])
        right = self._array(node.input[1])
        if node.op_type == "Mul":
            if left is not None and np.asarray(left).size == 1:
                return self.resolve(node.input[1]), float(np.asarray(left).reshape(-1)[0]), {index}
            if right is not None and np.asarray(right).size == 1:
                return self.resolve(node.input[0]), float(np.asarray(right).reshape(-1)[0]), {index}
        if node.op_type == "Div" and right is not None and np.asarray(right).size == 1:
            divisor = float(np.asarray(right).reshape(-1)[0])
            if divisor != 0:
                return self.resolve(node.input[0]), 1.0 / divisor, {index}
        return current, 1.0, set()

    @staticmethod
    def _is_attention_negative(value: float) -> bool:
        # A finite additive sentinel (for example -1e4 or -1e9) is not
        # mathematically identical to a hard exclusion unless the producer
        # also proves a bound on every QK score.  This frontend has no such
        # range proof, so only exact negative infinity may become a keep mask.
        return math.isinf(value) and value < 0

    def _constant_attention_mask(self, name: str, *, batch: int, queries: int, keys: int):
        array = self._array(name)
        if array is None:
            return None
        values = np.asarray(array)
        try:
            # The ONNX score tensor is [B,H,Q,K].  VolvoxAI masks are
            # head-independent, so broadcasting to H=1 both proves source
            # geometry and rejects a head-specific mask.
            expanded = np.broadcast_to(values, (batch, 1, queries, keys))[:, 0, :, :]
        except ValueError:
            return None
        keep = np.empty(expanded.shape, dtype=np.int32)
        iterator = np.nditer(expanded, flags=["multi_index"])
        for value in iterator:
            numeric = float(value)
            if numeric == 0.0:
                keep[iterator.multi_index] = 1
            elif self._is_attention_negative(numeric):
                keep[iterator.multi_index] = 0
            else:
                return None
        causal = np.fromfunction(
            lambda row, column: column <= row, (queries, keys), dtype=int
        ).astype(np.int32)
        if queries == keys and np.array_equal(keep, np.broadcast_to(causal, keep.shape)):
            return {"causal": True, "mask": None}
        if np.all(keep == keep[0:1, 0:1, :]):
            keep = keep[0, 0, :]
        elif np.all(keep == keep[:, 0:1, :]):
            keep = keep[:, 0, :]
        elif batch != queries and np.all(keep == keep[0:1, :, :]):
            keep = keep[0, :, :]
        derived = f"{name}__keep_i32"
        self.arrays[derived] = np.ascontiguousarray(keep)
        self.shape_map[derived] = list(keep.shape)
        self.dtype_map[derived] = "int32"
        return {"causal": False, "mask": derived}

    def _where_attention_mask(self, name: str, producers, *, batch: int, queries: int, keys: int):
        entry = producers.get(self.resolve(name))
        if not entry or entry[1].op_type != "Where":
            return None
        index, node = entry
        if len(node.input) != 3:
            return None
        true_value = self._array(node.input[1])
        false_value = self._array(node.input[2])
        if true_value is None or false_value is None or np.asarray(true_value).size != 1 or np.asarray(false_value).size != 1:
            return None
        true_scalar = float(np.asarray(true_value).reshape(-1)[0])
        false_scalar = float(np.asarray(false_value).reshape(-1)[0])
        if true_scalar != 0.0 or not self._is_attention_negative(false_scalar):
            return None
        condition = self.resolve(node.input[0])
        structural = {index}
        while True:
            condition_entry = producers.get(condition)
            if not condition_entry or condition_entry[1].op_type not in {"Unsqueeze", "Squeeze", "Expand", "Reshape"}:
                break
            structural.add(condition_entry[0])
            condition = self.resolve(condition_entry[1].input[0])
        if self.dtype_of(condition) != "int32":
            return None
        shape = self.shape_of(condition)
        accepted = (
            shape == [keys]
            or shape == [batch, keys]
            or shape == [queries, keys] and batch != queries
            or shape == [batch, queries, keys]
        )
        if not accepted:
            return None
        return {"causal": False, "mask": condition, "indices": structural}

    def _attention_mask(self, value: str, producers, *, batch: int, queries: int, keys: int):
        constant = self._constant_attention_mask(value, batch=batch, queries=queries, keys=keys)
        if constant is not None:
            constant["indices"] = set()
            return constant
        return self._where_attention_mask(
            value, producers, batch=batch, queries=queries, keys=keys
        )

    def _recognize_attention(self) -> None:
        """Recognize the common explicit QK-softmax-V ONNX attention DAG."""
        producers = self._producer_map()
        consumers = self._single_consumer_map()
        for context_index, context in enumerate(self.model.graph.node):
            if context_index in self.folded or context.op_type != "MatMul" or len(context.input) != 2:
                continue
            probability_entry = producers.get(self.resolve(context.input[0]))
            if not probability_entry or probability_entry[1].op_type != "Softmax":
                continue
            softmax_index, softmax = probability_entry
            softmax_shape = self.shape_of(softmax.input[0])
            if not softmax_shape:
                continue
            softmax_default_axis = 1 if self.opset <= 12 else -1
            softmax_axis = int(_attribute(softmax, "axis", softmax_default_axis))
            softmax_axis = softmax_axis + len(softmax_shape) if softmax_axis < 0 else softmax_axis
            if softmax_axis != len(softmax_shape) - 1:
                continue
            v_path = self._attention_head_path(context.input[1], producers, key=False)
            if v_path is None:
                continue
            score_value = self.resolve(softmax.input[0])
            mask_name = None
            region_indices = {context_index, softmax_index, *v_path["indices"]}
            score_entry = producers.get(score_value)
            if score_entry and score_entry[1].op_type == "Add":
                add_index, add = score_entry
                candidates = []
                for score_side, mask_side in ((0, 1), (1, 0)):
                    unscaled, scale, scale_indices = self._attention_scale(add.input[score_side], producers)
                    qk_entry = producers.get(unscaled)
                    if qk_entry and qk_entry[1].op_type == "MatMul":
                        candidates.append((qk_entry, scale, scale_indices, add.input[mask_side]))
                if len(candidates) != 1:
                    continue
                qk_entry, scale, scale_indices, mask_name = candidates[0]
                region_indices.add(add_index)
            else:
                unscaled, scale, scale_indices = self._attention_scale(score_value, producers)
                qk_entry = producers.get(unscaled)
                if not qk_entry or qk_entry[1].op_type != "MatMul":
                    continue
            qk_index, qk = qk_entry
            q_path = self._attention_head_path(qk.input[0], producers, key=False)
            k_path = self._attention_head_path(qk.input[1], producers, key=True)
            if q_path is None or k_path is None:
                continue
            if q_path["heads"] != k_path["heads"] or q_path["heads"] != v_path["heads"]:
                continue
            if q_path["head_dim"] != k_path["head_dim"] or q_path["head_dim"] != v_path["head_dim"]:
                continue
            q_shape = self.shape_of(q_path["base"])
            k_shape = self.shape_of(k_path["base"])
            v_shape = self.shape_of(v_path["base"])
            if len(q_shape) != 3 or len(k_shape) != 3 or k_shape != v_shape or q_shape[0] != k_shape[0] or q_shape[2] != k_shape[2]:
                continue
            queries, keys = q_shape[1], k_shape[1]
            causal = False
            runtime_mask = None
            mask_indices: set[int] = set()
            if mask_name is not None:
                if any(
                    not isinstance(dimension, int)
                    for dimension in (q_shape[0], queries, keys)
                ):
                    continue
                mask = self._attention_mask(
                    mask_name,
                    producers,
                    batch=q_shape[0],
                    queries=queries,
                    keys=keys,
                )
                if mask is None:
                    continue
                causal = bool(mask["causal"])
                runtime_mask = mask["mask"]
                mask_indices = set(mask.get("indices", ()))
            # Context is normally restored [B,H,Q,Dh] -> [B,Q,H,Dh] -> [B,Q,D].
            final_index = context_index
            final_output = context.output[0]
            context_consumers = consumers.get(context.output[0], [])
            if len(context_consumers) == 1:
                transpose_index = context_consumers[0]
                transpose = self.model.graph.node[transpose_index]
                if transpose.op_type == "Transpose" and list(_attribute(transpose, "perm", ())) == [0, 2, 1, 3]:
                    transpose_consumers = consumers.get(transpose.output[0], [])
                    if len(transpose_consumers) == 1 and self.model.graph.node[transpose_consumers[0]].op_type == "Reshape":
                        final_index = transpose_consumers[0]
                        final_output = self.model.graph.node[final_index].output[0]
                        region_indices.update({transpose_index, final_index})
            if len(self.shape_of(final_output)) != 3:
                continue
            region_indices.update({qk_index, *scale_indices, *q_path["indices"], *k_path["indices"], *mask_indices})
            # Every eliminated intermediate must be local to the region.
            safe = True
            for region_index in region_indices:
                region_node = self.model.graph.node[region_index]
                if region_index == final_index:
                    continue
                for produced in region_node.output:
                    if any(consumer not in region_indices for consumer in consumers.get(produced, ())):
                        safe = False
            if not safe:
                continue
            attention = _Attention(
                source_node=_source_name(context, context_index),
                q=q_path["base"],
                k=k_path["base"],
                v=v_path["base"],
                output=final_output,
                heads=int(q_path["heads"]),
                scale=float(scale),
                causal=causal,
                mask=runtime_mask,
                skip=frozenset(region_indices),
            )
            if not math.isfinite(attention.scale) or attention.scale <= 0:
                raise ExporterError(Diagnostic(
                    "VXATTENTION_SCALE",
                    f"{attention.source_node} attention scale must be finite and positive",
                    "recognize-attention",
                    source_node=attention.source_node,
                ))
            self.attention_replacements[final_index] = attention
            self.attention_skip.update(region_indices)
            self.features["attention"].append({
                "source_node": attention.source_node,
                "heads": attention.heads,
                "causal": attention.causal,
                "kind": "self" if self._attention_is_self(attention) else "separate-qkv",
            })

    def _float_storage(self) -> str:
        requested = (self.weight_dtype or "auto").lower()
        if requested in {"float16", "fp16", "f16"}:
            return "float16"
        if requested in {"float32", "fp32", "f32"}:
            return "float32"
        if requested != "auto":
            raise ExporterError(Diagnostic("VXWEIGHT_DTYPE", f"unsupported weight dtype {requested!r}", "usage"))
        if any(np.asarray(value).dtype == np.float16 for value in self.arrays.values()):
            return "float16"
        filename = Path(self.model_path).name.lower()
        return "float16" if any(token in filename for token in ("float16", "fp16", "f16")) else "float32"

    def _array(self, name: str) -> Optional[np.ndarray]:
        return self.arrays.get(self.resolve(name))

    def ensure_weight(
        self,
        name: str,
        *,
        array: Optional[np.ndarray] = None,
        preferred: Optional[str] = None,
        force_float32: bool = False,
    ) -> str:
        source = self.resolve(name)
        exported = self.names.get(source if array is None else f"{source}::{preferred or 'derived'}", weight=True, preferred=preferred)
        if exported in self.weights:
            if force_float32 and self.weights[exported].dtype != np.dtype(np.float32):
                replacement = np.asarray(self.arrays.get(source) if array is None else array, dtype=np.float32)
                if replacement.ndim == 0:
                    replacement = replacement.reshape(1)
                self.weights[exported] = np.ascontiguousarray(replacement)
            return exported
        value = np.asarray(self.arrays.get(source) if array is None else array)
        if value.dtype == np.bool_:
            value = value.astype(np.int32)
        elif value.dtype == np.int64:
            value = _to_i32_checked(value, label=source)
        elif value.dtype.kind == "f":
            value = value.astype(
                np.float32 if force_float32 or self._float_storage() == "float32" else np.float16
            )
        elif value.dtype not in (np.dtype(np.int32), np.dtype(np.int8), np.dtype(np.uint8)):
            raise ExporterError(Diagnostic(
                "VXWEIGHT_STORAGE",
                f"constant {source!r} has unsupported storage dtype {value.dtype}",
                "dtype-legalize",
                source_node=source,
            ))
        if value.ndim == 0:
            value = value.reshape(1)
        self.weights[exported] = np.ascontiguousarray(value)
        return exported

