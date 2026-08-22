"""Constant loading, shape specialization and constant folding for the ONNX
frontend.

Resolves initializers and statically known shapes, evaluates the parts of the
graph that do not depend on runtime inputs, and folds them away before
recognition runs.
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
from .frontend_onnx_recognize import OnnxRecognitionMixin

from .frontend_onnx_common import (
    _attribute,
    _broadcast_shape,
    _concat_shape,
    _onnx_dtype_name,
    _product,
    _runtime_dtype_for_array,
    _source_name,
)


class OnnxConstantFoldingMixin:
    """Mixin for OnnxCompiler; see frontend_onnx.py."""

    def _load_constants(self) -> None:
        for index, node in enumerate(self.model.graph.node):
            if node.op_type != "Constant" or not node.output:
                continue
            value = _attribute(node, "value")
            if value is not None:
                array = self.numpy_helper.to_array(value)
            elif _attribute(node, "value_float") is not None:
                array = np.asarray(_attribute(node, "value_float"), dtype=np.float32)
            elif _attribute(node, "value_int") is not None:
                array = np.asarray(_attribute(node, "value_int"), dtype=np.int64)
            elif _attribute(node, "value_floats") is not None:
                array = np.asarray(_attribute(node, "value_floats"), dtype=np.float32)
            elif _attribute(node, "value_ints") is not None:
                array = np.asarray(_attribute(node, "value_ints"), dtype=np.int64)
            else:
                raise ExporterError(Diagnostic(
                    "VXONNX_CONSTANT",
                    f"{_source_name(node, index)} uses an unsupported Constant encoding",
                    "frontend",
                    source_node=_source_name(node, index),
                    source_op="Constant",
                ))
            self.arrays[node.output[0]] = np.asarray(array)
            self.shape_map[node.output[0]] = list(np.asarray(array).shape)
            self.dtype_map[node.output[0]] = _runtime_dtype_for_array(np.asarray(array))
            self.folded.add(index)

    def _prepare_public_names(self) -> None:
        graph_inputs = [value for value in self.model.graph.input if value.name not in self.initializer_names]
        for index, value in enumerate(graph_inputs):
            self.names.bind(value.name, f"input{index}")
        outputs = list(self.model.graph.output)
        requested = self.output_names_requested or [f"output{index}" for index in range(len(outputs))]
        if len(requested) != len(outputs):
            raise ExporterError(Diagnostic(
                "VXOUTPUT_NAMES",
                f"expected {len(outputs)} output name(s), received {len(requested)}",
                "usage",
            ))
        if any(not isinstance(name, str) or not name.strip() for name in requested) or len(set(requested)) != len(requested):
            raise ExporterError(Diagnostic("VXOUTPUT_NAMES", "output names must be unique non-empty strings", "usage"))
        for output, name in zip(outputs, requested):
            self.names.bind(output.name, name)

    def _validate_dtype_binding_names(self) -> None:
        graph_inputs = [
            value for value in self.model.graph.input
            if value.name not in self.initializer_names
        ]
        allowed_inputs = {
            name
            for index, value in enumerate(graph_inputs)
            for name in (value.name, f"input{index}")
        }
        allowed_outputs = {
            name
            for value in self.model.graph.output
            for name in (value.name, self.names.get(value.name))
        }
        unknown_inputs = sorted(set(self.input_dtype_bindings) - allowed_inputs)
        unknown_outputs = sorted(set(self.output_dtype_bindings) - allowed_outputs)
        if unknown_inputs:
            raise ExporterError(Diagnostic(
                "VXINPUT_DTYPE_UNKNOWN",
                f"input dtype binding names unknown input(s): {', '.join(unknown_inputs)}",
                "usage",
            ))
        if unknown_outputs:
            raise ExporterError(Diagnostic(
                "VXOUTPUT_DTYPE_UNKNOWN",
                f"output dtype binding names unknown output(s): {', '.join(unknown_outputs)}",
                "usage",
            ))

    def _specialization_input(self, graph_inputs, key: str):
        for index, value in enumerate(graph_inputs):
            if key in {value.name, f"input{index}"}:
                return value, index
        return None, -1

    def _static_shape_value(self, name: str) -> Optional[list[int]]:
        """Return a fully known tensor shape without materializing its data."""

        resolved = self.resolve(name)
        if resolved in self.arrays:
            return list(np.asarray(self.arrays[resolved]).shape)
        for candidate in (resolved, name):
            if candidate not in self.shape_map:
                continue
            shape = list(self.shape_map[candidate])
            # Shape inference represents both symbolic and otherwise unknown
            # dimensions as zero, so zero cannot safely be folded as a value.
            if all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in shape
            ):
                return [int(dimension) for dimension in shape]
        return None

    @staticmethod
    def _evaluate_shape(node, shape: list[int]) -> np.ndarray:
        rank = len(shape)
        start = int(_attribute(node, "start", 0))
        end = int(_attribute(node, "end", rank))
        return np.asarray(shape[slice(start, end)], dtype=np.int64)

    def _static_reshape_output_shape(self, node) -> Optional[list[int]]:
        if len(node.input) < 2:
            return None
        target = self._array(node.input[1])
        if target is None or np.asarray(target).ndim != 1:
            return None
        requested = [int(value) for value in np.asarray(target).reshape(-1)]
        allowzero = bool(int(_attribute(node, "allowzero", 0)))
        source_shape = self._static_shape_value(node.input[0])
        output_shape: list[int] = []
        inferred_axis = None
        for axis, dimension in enumerate(requested):
            if dimension > 0:
                output_shape.append(dimension)
            elif dimension == 0 and not allowzero:
                if source_shape is None or axis >= len(source_shape):
                    return None
                output_shape.append(source_shape[axis])
            elif dimension == -1 and inferred_axis is None:
                inferred_axis = axis
                output_shape.append(-1)
            else:
                return None
        if inferred_axis is not None:
            if source_shape is None:
                return None
            known_product = _product(
                dimension
                for axis, dimension in enumerate(output_shape)
                if axis != inferred_axis
            )
            source_product = _product(source_shape)
            if known_product <= 0 or source_product % known_product:
                return None
            output_shape[inferred_axis] = source_product // known_product
        return output_shape if all(dimension > 0 for dimension in output_shape) else None

    def _propagate_static_output_shape(self, node) -> Optional[list[int]]:
        """Infer one fully concrete output shape after shape-value folding."""

        if not node.output or not node.input:
            return None
        op = node.op_type
        input_shape = self._static_shape_value(node.input[0])
        if input_shape is None:
            return None
        if op == "Reshape":
            return self._static_reshape_output_shape(node)
        if op in {
            "Identity", "Cast", "Clip", "Not", "Relu", "Sigmoid", "Tanh",
            "Erf", "LayerNormalization", "Softmax", "LogSoftmax",
            "QuantizeLinear", "DequantizeLinear",
        }:
            return input_shape
        if op == "Transpose":
            permutation = list(
                _attribute(node, "perm", list(reversed(range(len(input_shape)))))
            )
            if sorted(permutation) != list(range(len(input_shape))):
                return None
            return [input_shape[axis] for axis in permutation]
        if op == "Unsqueeze":
            axes_value = (
                self._array(node.input[1])
                if len(node.input) > 1 and node.input[1]
                else None
            )
            axes = list(_attribute(node, "axes", ())) or (
                [int(value) for value in np.asarray(axes_value).reshape(-1)]
                if axes_value is not None
                else []
            )
            output_rank = len(input_shape) + len(axes)
            normalized = sorted(
                axis + output_rank if axis < 0 else axis for axis in axes
            )
            if (
                len(set(normalized)) != len(normalized)
                or any(axis < 0 or axis >= output_rank for axis in normalized)
            ):
                return None
            result = list(input_shape)
            for axis in normalized:
                result.insert(axis, 1)
            return result
        if op == "Squeeze":
            axes_value = (
                self._array(node.input[1])
                if len(node.input) > 1 and node.input[1]
                else None
            )
            axes = list(_attribute(node, "axes", ())) or (
                [int(value) for value in np.asarray(axes_value).reshape(-1)]
                if axes_value is not None
                else None
            )
            if axes is None:
                return [dimension for dimension in input_shape if dimension != 1]
            normalized = {
                axis + len(input_shape) if axis < 0 else axis for axis in axes
            }
            if (
                len(normalized) != len(axes)
                or any(
                    axis < 0
                    or axis >= len(input_shape)
                    or input_shape[axis] != 1
                    for axis in normalized
                )
            ):
                return None
            return [
                dimension
                for axis, dimension in enumerate(input_shape)
                if axis not in normalized
            ]
        if op == "Flatten":
            axis = int(_attribute(node, "axis", 1))
            axis = axis + len(input_shape) if axis < 0 else axis
            if axis < 0 or axis > len(input_shape):
                return None
            return [_product(input_shape[:axis]), _product(input_shape[axis:])]
        if op == "Expand" and len(node.input) > 1:
            target = self._array(node.input[1])
            if target is None:
                return None
            shape = [int(value) for value in np.asarray(target).reshape(-1)]
            if not shape or any(dimension <= 0 for dimension in shape):
                return None
            return _broadcast_shape(input_shape, shape, node=_source_name(node, -1))
        if op in {"Add", "Sub", "Mul", "Div", "Equal", "GreaterOrEqual"}:
            if len(node.input) < 2:
                return None
            right_shape = self._static_shape_value(node.input[1])
            if right_shape is None:
                return None
            return _broadcast_shape(input_shape, right_shape, node=_source_name(node, -1))
        if op == "Where" and len(node.input) == 3:
            x_shape = self._static_shape_value(node.input[1])
            y_shape = self._static_shape_value(node.input[2])
            if x_shape is None or y_shape is None:
                return None
            values_shape = _broadcast_shape(x_shape, y_shape, node=_source_name(node, -1))
            return _broadcast_shape(input_shape, values_shape, node=_source_name(node, -1))
        if op == "Concat":
            shapes = [self._static_shape_value(name) for name in node.input]
            if any(shape is None for shape in shapes):
                return None
            rank = len(input_shape)
            axis = int(_attribute(node, "axis", 0))
            axis = axis + rank if axis < 0 else axis
            if axis < 0 or axis >= rank or any(len(shape or ()) != rank for shape in shapes):
                return None
            result = list(input_shape)
            result[axis] = 0
            for shape in shapes:
                assert shape is not None
                if any(
                    dimension != input_shape[index]
                    for index, dimension in enumerate(shape)
                    if index != axis
                ):
                    return None
                result[axis] += shape[axis]
            return result
        if op == "Gather" and len(node.input) > 1:
            indices_shape = self._static_shape_value(node.input[1])
            if indices_shape is None:
                return None
            axis = int(_attribute(node, "axis", 0))
            axis = axis + len(input_shape) if axis < 0 else axis
            if axis < 0 or axis >= len(input_shape):
                return None
            return [*input_shape[:axis], *indices_shape, *input_shape[axis + 1:]]
        if op == "MatMul" and len(node.input) > 1:
            right_shape = self._static_shape_value(node.input[1])
            if (
                right_shape is None
                or len(input_shape) < 2
                or len(right_shape) < 2
                or input_shape[-1] != right_shape[-2]
            ):
                return None
            batch = _broadcast_shape(
                input_shape[:-2], right_shape[:-2], node=_source_name(node, -1)
            )
            return [*batch, input_shape[-2], right_shape[-1]]
        if op == "Gemm" and len(node.input) > 1:
            right_shape = self._static_shape_value(node.input[1])
            trans_a = int(_attribute(node, "transA", 0))
            trans_b = int(_attribute(node, "transB", 0))
            if (
                right_shape is None
                or len(input_shape) != 2
                or len(right_shape) != 2
                or trans_a not in {0, 1}
                or trans_b not in {0, 1}
            ):
                return None
            rows = input_shape[1] if trans_a else input_shape[0]
            inner_left = input_shape[0] if trans_a else input_shape[1]
            inner_right = right_shape[1] if trans_b else right_shape[0]
            columns = right_shape[0] if trans_b else right_shape[1]
            if inner_left != inner_right:
                return None
            return [rows, columns]
        if op in {"ReduceSum", "ReduceMean"}:
            axes = _attribute(node, "axes")
            if axes is None and len(node.input) > 1 and node.input[1]:
                axes_value = self._array(node.input[1])
                axes = (
                    [int(value) for value in np.asarray(axes_value).reshape(-1)]
                    if axes_value is not None
                    else None
                )
            if axes is None:
                axes = list(range(len(input_shape)))
            normalized = {
                axis + len(input_shape) if axis < 0 else axis
                for axis in (int(value) for value in axes)
            }
            if any(axis < 0 or axis >= len(input_shape) for axis in normalized):
                return None
            keepdims = bool(int(_attribute(node, "keepdims", 1)))
            return [
                (1 if keepdims and axis in normalized else dimension)
                for axis, dimension in enumerate(input_shape)
                if keepdims or axis not in normalized
            ]
        return None

    def _apply_specializations_and_fold(self) -> None:
        graph_inputs = [value for value in self.model.graph.input if value.name not in self.initializer_names]
        specialized_sources: set[str] = set()
        specialized_values: dict[str, int] = {}
        for key, integer in self.specializations.items():
            value, index = self._specialization_input(graph_inputs, key)
            if value is None:
                raise ExporterError(Diagnostic(
                    "VXSPEC_UNKNOWN", f"specialization names unknown public input {key!r}", "specialize-inputs"
                ))
            dtype = self.dtype_map.get(value.name)
            shape = self.shape_map.get(value.name, [])
            if dtype not in {"int32", "int64"} or (_product(shape) if shape else 1) != 1:
                raise ExporterError(Diagnostic(
                    "VXSPEC_TYPE",
                    f"specialized input {value.name!r} must be an integer scalar or one-element tensor",
                    "specialize-inputs",
                    source_node=value.name,
                ))
            if integer < -(2**31) or integer > 2**31 - 1:
                raise ExporterError(Diagnostic(
                    "VXSPEC_RANGE", f"specialized value for {value.name!r} is outside int32 range", "specialize-inputs"
                ))
            previous = specialized_values.get(value.name)
            if previous is not None:
                if previous != integer:
                    raise ExporterError(Diagnostic(
                        "VXSPEC_CONFLICT",
                        f"specialization aliases for {value.name!r} disagree ({previous} versus {integer})",
                        "specialize-inputs",
                        source_node=value.name,
                        constraint="one unambiguous compile-time value per public input",
                    ))
                continue
            specialized_values[value.name] = integer
            array_shape = shape if shape else []
            self.arrays[value.name] = np.asarray(integer, dtype=np.int64 if dtype == "int64" else np.int32).reshape(array_shape)
            specialized_sources.add(value.name)
            self.abi_changes.append({
                "kind": "specialize-input",
                "name": value.name,
                "source": {"dtype": dtype, "shape": shape},
                "exported": {"removed": True, "value": integer},
            })

        changed = True
        while changed:
            changed = False
            for index, node in enumerate(self.model.graph.node):
                if index in self.folded or not node.output:
                    continue
                output_shape = self._propagate_static_output_shape(node)
                if (
                    output_shape is not None
                    and self.shape_map.get(node.output[0]) != output_shape
                ):
                    self.shape_map[node.output[0]] = output_shape
                    changed = True
                if node.op_type == "QuantizeLinear":
                    zero_name = (
                        self.resolve(node.input[2])
                        if len(node.input) > 2 and node.input[2]
                        else ""
                    )
                    output_dtype = (
                        self.dtype_of(zero_name) if zero_name else "uint8"
                    )
                    if self.dtype_map.get(node.output[0]) != output_dtype:
                        self.dtype_map[node.output[0]] = output_dtype
                        changed = True
                elif node.op_type == "DequantizeLinear" and len(node.input) > 1:
                    output_dtype = self.dtype_of(self.resolve(node.input[1]))
                    if self.dtype_map.get(node.output[0]) != output_dtype:
                        self.dtype_map[node.output[0]] = output_dtype
                        changed = True
                inputs = [self.resolve(name) for name in node.input if name]
                if node.op_type == "Shape" and node.input:
                    static_shape = self._static_shape_value(node.input[0])
                    result = (
                        self._evaluate_shape(node, static_shape)
                        if static_shape is not None
                        else None
                    )
                elif inputs and all(name in self.arrays for name in inputs):
                    result = self._evaluate_constant(
                        node, [self.arrays[name] for name in inputs]
                    )
                else:
                    result = None
                if result is None:
                    continue
                results = result if isinstance(result, tuple) else (result,)
                if len(results) != len(node.output):
                    continue
                for name, array in zip(node.output, results):
                    self.arrays[name] = np.asarray(array)
                    self.shape_map[name] = list(np.asarray(array).shape)
                    self.dtype_map[name] = _runtime_dtype_for_array(np.asarray(array))
                self.folded.add(index)
                changed = True

        if specialized_sources:
            remaining = []
            for index, node in enumerate(self.model.graph.node):
                if index in self.folded:
                    continue
                if any(self.resolve(name) in specialized_sources or name in specialized_sources for name in node.input):
                    remaining.append(_source_name(node, index))
            if remaining:
                raise ExporterError(Diagnostic(
                    "VXSPEC_INCOMPLETE",
                    f"specialized selector remains live at: {', '.join(remaining)}",
                    "specialize-inputs",
                    constraint="every specialized use must constant-fold before lowering",
                ))

    def _evaluate_constant(self, node, values: list[np.ndarray]):
        op = node.op_type
        try:
            if op == "Identity":
                return values[0]
            if op == "Cast":
                target = _onnx_dtype_name(int(_attribute(node, "to")))
                dtype = {"float32": np.float32, "int32": np.int32, "int64": np.int64, "bool": np.bool_}.get(target)
                return values[0].astype(dtype) if dtype is not None else None
            if op == "Transpose":
                permutation = list(_attribute(node, "perm", list(reversed(range(values[0].ndim)))))
                return np.transpose(values[0], permutation)
            if op == "Reshape":
                return np.reshape(values[0], tuple(int(value) for value in values[1].reshape(-1)))
            if op == "Squeeze":
                axes = list(_attribute(node, "axes", ())) or (list(values[1].reshape(-1)) if len(values) > 1 else None)
                return np.squeeze(values[0], axis=tuple(int(axis) for axis in axes) if axes is not None else None)
            if op == "Unsqueeze":
                axes = list(_attribute(node, "axes", ())) or (list(values[1].reshape(-1)) if len(values) > 1 else [])
                result = values[0]
                for axis in sorted(int(axis) for axis in axes):
                    result = np.expand_dims(result, axis)
                return result
            if op == "Shape":
                return self._evaluate_shape(node, list(values[0].shape))
            if op == "Gather":
                return np.take(values[0], values[1], axis=int(_attribute(node, "axis", 0)))
            if op == "Concat":
                return np.concatenate(values, axis=int(_attribute(node, "axis", 0)))
            if op in {"Add", "Sub", "Mul", "Div"}:
                return {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}[op](values[0], values[1])
            if op == "Pow":
                # A shape program reaches Pow through the conventional
                # `Cast(Shape(...)) ** -0.5` attention scale.  Fold it in F64
                # and keep only exactly representable finite results: a
                # negative base with a fractional exponent is complex, and
                # overflow would silently substitute an infinite scale.
                base = np.asarray(values[0], dtype=np.float64)
                exponent = np.asarray(values[1], dtype=np.float64)
                with np.errstate(all="ignore"):
                    folded = np.power(base, exponent)
                if not np.all(np.isfinite(folded)):
                    return None
                return folded.astype(np.float32)
            if op == "Equal":
                return np.equal(values[0], values[1])
            if op == "GreaterOrEqual":
                return np.greater_equal(values[0], values[1])
            if op == "Not":
                return np.logical_not(values[0])
            if op == "Where":
                return np.where(values[0], values[1], values[2])
            if op == "Expand":
                target = tuple(int(value) for value in values[1].reshape(-1))
                return np.broadcast_to(
                    values[0], np.broadcast_shapes(values[0].shape, target)
                )
            if op == "Trilu":
                diagonal = int(values[1].reshape(-1)[0]) if len(values) > 1 else 0
                return (
                    np.triu(values[0], diagonal)
                    if int(_attribute(node, "upper", 1))
                    else np.tril(values[0], diagonal)
                )
            if op == "Split":
                axis = int(_attribute(node, "axis", 0))
                if axis < 0:
                    axis += values[0].ndim
                split = (
                    [int(value) for value in values[1].reshape(-1)]
                    if len(values) > 1
                    else [int(value) for value in _attribute(node, "split", ())]
                )
                if split:
                    boundaries = np.cumsum(split, dtype=np.int64)[:-1]
                    return tuple(np.split(values[0], boundaries, axis=axis))
                return tuple(np.array_split(values[0], len(node.output), axis=axis))
            if op == "ConstantOfShape":
                tensor = _attribute(node, "value")
                fill = self.numpy_helper.to_array(tensor).reshape(-1)[0] if tensor is not None else np.float32(0)
                return np.full(tuple(int(value) for value in values[0].reshape(-1)), fill)
            if op == "Range":
                return np.arange(values[0].item(), values[1].item(), values[2].item(), dtype=values[0].dtype)
            if op == "Slice":
                starts = values[1].reshape(-1).astype(int)
                ends = values[2].reshape(-1).astype(int)
                axes = values[3].reshape(-1).astype(int) if len(values) > 3 else np.arange(len(starts))
                steps = values[4].reshape(-1).astype(int) if len(values) > 4 else np.ones(len(starts), dtype=int)
                slices = [slice(None)] * values[0].ndim
                for start, end, axis, step in zip(starts, ends, axes, steps):
                    slices[int(axis)] = slice(int(start), int(end), int(step))
                return values[0][tuple(slices)]
        except (IndexError, TypeError, ValueError, OverflowError):
            return None
        return None

    def _producer_map(self):
        producers = {}
        for index, node in enumerate(self.model.graph.node):
            for output in node.output:
                producers[output] = (index, node)
        return producers

    @staticmethod
    def _shape_extent(value: object) -> int | str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            return int(value)
        if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
            integer = int(value)
            if float(value) == integer:
                return integer
        return None

    def _shape_products_equal(
        self,
        left: Iterable[int | str],
        right: Iterable[int | str],
    ) -> bool:
        """Compare tensor element counts without inventing symbolic algebra."""

        def factors(shape: Iterable[int | str]):
            fixed = 1
            symbols: dict[str, int] = defaultdict(int)
            for dimension in shape:
                if isinstance(dimension, int):
                    if dimension <= 0:
                        return None
                    fixed *= dimension
                elif isinstance(dimension, str):
                    constraint = self.shape_environment.get(dimension)
                    if constraint is not None and constraint.min == constraint.max:
                        fixed *= constraint.min
                    else:
                        symbols[dimension] += 1
                else:
                    return None
            return fixed, dict(symbols)

        return factors(left) == factors(right)

    def _canonical_affine_shape_extent(
        self,
        source: str,
        *,
        scale: int = 1,
        offset: int = 0,
    ) -> int | str | None:
        """Resolve one scalar affine shape expression to an existing symbol."""

        constraint = self.shape_environment.get(source)
        if constraint is None or scale <= 0:
            return None

        def progression(item):
            step = item.multiple_of or 1
            remainder = item.min % step
            first = item.min + (0 if remainder == 0 else step - remainder)
            last = item.max - (item.max % step)
            if first > last:
                return None
            return first, last, step, ((last - first) // step) + 1

        source_progression = progression(constraint)
        if source_progression is None:
            return None
        first, last, step, count = source_progression
        expected = (
            first * scale + offset,
            last * scale + offset,
            step * scale,
            count,
        )
        if count == 1:
            return expected[0] if expected[0] > 0 else None
        matches = [
            item.name
            for item in self.shape_environment.dimensions
            if progression(item) == expected
        ]
        return matches[0] if len(matches) == 1 else None

    def _evaluate_symbolic_shape_arithmetic(
        self,
        op: str,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray | None:
        """Evaluate scalar affine arithmetic without publishing expressions."""

        try:
            left_values, right_values = np.broadcast_arrays(left, right)
        except ValueError:
            return None
        result = np.empty(left_values.shape, dtype=object)
        for index in np.ndindex(result.shape):
            a = self._shape_extent(left_values[index])
            b = self._shape_extent(right_values[index])
            if a is None or b is None:
                return None

            # Singleton bounded symbols are exact constants inside arithmetic,
            # while their direct Shape value remains the authored ABI symbol.
            if isinstance(a, str):
                constraint = self.shape_environment.get(a)
                if constraint is not None and constraint.min == constraint.max:
                    a = constraint.min
            if isinstance(b, str):
                constraint = self.shape_environment.get(b)
                if constraint is not None and constraint.min == constraint.max:
                    b = constraint.min

            value: int | str | None
            if isinstance(a, int) and isinstance(b, int):
                if op == "Add":
                    value = a + b
                elif op == "Sub":
                    value = a - b
                elif op == "Mul":
                    value = a * b
                elif op == "Div" and b != 0:
                    quotient = abs(a) // abs(b)
                    value = quotient if (a < 0) == (b < 0) else -quotient
                else:
                    value = None
            elif op == "Add" and isinstance(a, str) and isinstance(b, int):
                value = self._canonical_affine_shape_extent(a, offset=b)
            elif op == "Add" and isinstance(a, int) and isinstance(b, str):
                value = self._canonical_affine_shape_extent(b, offset=a)
            elif op == "Sub" and isinstance(a, str) and isinstance(b, int):
                value = self._canonical_affine_shape_extent(a, offset=-b)
            elif op == "Mul" and isinstance(a, str) and isinstance(b, int):
                value = self._canonical_affine_shape_extent(a, scale=b)
            elif op == "Mul" and isinstance(a, int) and isinstance(b, str):
                value = self._canonical_affine_shape_extent(b, scale=a)
            else:
                value = None
            if value is None or (isinstance(value, int) and value <= 0):
                return None
            result[index] = value
        return result

    def _evaluate_symbolic_shape_equal(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray | None:
        """Prove an Equal result is invariant over the complete shape domain."""

        try:
            left_values, right_values = np.broadcast_arrays(left, right)
        except ValueError:
            return None
        result = np.empty(left_values.shape, dtype=bool)
        for index in np.ndindex(result.shape):
            a = self._shape_extent(left_values[index])
            b = self._shape_extent(right_values[index])
            if a is None or b is None:
                return None
            if isinstance(a, int) and isinstance(b, int):
                result[index] = a == b
                continue
            if isinstance(a, str) and isinstance(b, str):
                if a == b:
                    result[index] = True
                    continue
                left_constraint = self.shape_environment.get(a)
                right_constraint = self.shape_environment.get(b)
                if left_constraint is None or right_constraint is None:
                    return None
                if (
                    left_constraint.max < right_constraint.min
                    or right_constraint.max < left_constraint.min
                ):
                    result[index] = False
                    continue
                # Two distinct symbols with overlapping domains have no
                # authored correlation proof. Their equality is runtime data.
                return None

            symbol = a if isinstance(a, str) else b
            integer = b if isinstance(a, str) else a
            assert isinstance(symbol, str) and isinstance(integer, int)
            constraint = self.shape_environment.get(symbol)
            if constraint is None:
                return None
            if constraint.min == constraint.max:
                result[index] = constraint.min == integer
                continue
            multiple = constraint.multiple_of or 1
            if (
                integer < constraint.min
                or integer > constraint.max
                or integer % multiple != 0
            ):
                result[index] = False
                continue
            # The comparison changes across the admitted runtime domain.
            return None
        return result

    def _evaluate_symbolic_shape_value(
        self,
        name: str,
        *,
        producers,
        memo: dict[str, tuple[np.ndarray, frozenset[int]] | None],
        visiting: set[str],
    ) -> tuple[np.ndarray, frozenset[int]] | None:
        """Evaluate an ONNX shape-tensor program over bounded symbols.

        Values are object arrays whose scalar members are either exact integers
        or existing ONNX dimension symbols.  Deliberately, this evaluator does
        not create expression strings: an affine runtime dimension must still
        be authored as one caller-bounded canonical symbol.
        """

        resolved = self.resolve(name)
        if resolved in memo:
            return memo[resolved]
        array = self._array(resolved)
        if array is not None:
            result = (np.asarray(array, dtype=object), frozenset())
            memo[resolved] = result
            return result
        if resolved in visiting:
            memo[resolved] = None
            return None
        produced = producers.get(resolved)
        if produced is None:
            memo[resolved] = None
            return None
        index, node = produced
        if len(node.output) != 1:
            memo[resolved] = None
            return None

        visiting.add(resolved)

        def operand(position: int):
            if position >= len(node.input) or not node.input[position]:
                return None
            return self._evaluate_symbolic_shape_value(
                node.input[position],
                producers=producers,
                memo=memo,
                visiting=visiting,
            )

        value: np.ndarray | None = None
        dependencies: set[int] = set()
        op = node.op_type
        try:
            if op == "Shape" and node.input:
                source_shape = self.shape_of(node.input[0])
                if source_shape and all(
                    self._shape_extent(dimension) is not None
                    for dimension in source_shape
                ):
                    rank = len(source_shape)
                    start = int(_attribute(node, "start", 0))
                    end = int(_attribute(node, "end", rank))
                    value = np.asarray(
                        source_shape[slice(start, end)], dtype=object
                    )
            elif op == "Identity":
                source = operand(0)
                if source is not None:
                    value = np.asarray(source[0], dtype=object)
                    dependencies.update(source[1])
            elif op == "Cast":
                source = operand(0)
                source_dtype = self.dtype_of(node.input[0]) if node.input else ""
                target_dtype = _onnx_dtype_name(int(_attribute(node, "to")))
                # Shape-program erasure must preserve the source program for
                # every bounded extent.  Even apparently harmless casts to
                # BOOL, float, or a narrower integer can change a dimension
                # value (truth conversion, rounding, or overflow).  Keep those
                # programs executable/fail closed; only an authored integer
                # identity cast is representation preserving without another
                # value-domain proof.
                if (
                    source is not None
                    and source_dtype == target_dtype
                    and target_dtype in {"int32", "int64"}
                ):
                    value = np.asarray(source[0], dtype=object)
                    dependencies.update(source[1])
            elif op == "Gather":
                source = operand(0)
                indices = operand(1)
                if source is not None and indices is not None:
                    index_values = np.asarray(indices[0]).reshape(-1)
                    if all(self._shape_extent(item) is not None for item in index_values):
                        numeric_indices = np.asarray(
                            [int(item) for item in index_values], dtype=np.int64
                        ).reshape(np.asarray(indices[0]).shape)
                        value = np.take(
                            source[0],
                            numeric_indices,
                            axis=int(_attribute(node, "axis", 0)),
                        )
                        dependencies.update(source[1])
                        dependencies.update(indices[1])
            elif op in {"Squeeze", "Unsqueeze"}:
                source = operand(0)
                axes_source = operand(1) if len(node.input) > 1 else None
                axes = list(_attribute(node, "axes", ()))
                if not axes and axes_source is not None:
                    axes = [int(item) for item in axes_source[0].reshape(-1)]
                if source is not None:
                    value = np.asarray(source[0], dtype=object)
                    if op == "Squeeze":
                        value = np.squeeze(
                            value,
                            axis=(
                                tuple(int(axis) for axis in axes)
                                if axes else None
                            ),
                        )
                    elif axes:
                        output_rank = value.ndim + len(axes)
                        normalized = sorted(
                            int(axis) + output_rank if int(axis) < 0 else int(axis)
                            for axis in axes
                        )
                        for axis in normalized:
                            value = np.expand_dims(value, axis)
                    else:
                        value = None
                    dependencies.update(source[1])
                    if axes_source is not None:
                        dependencies.update(axes_source[1])
            elif op == "Concat":
                sources = [operand(position) for position in range(len(node.input))]
                if sources and all(source is not None for source in sources):
                    value = np.concatenate(
                        [source[0] for source in sources if source is not None],
                        axis=int(_attribute(node, "axis", 0)),
                    )
                    for source in sources:
                        assert source is not None
                        dependencies.update(source[1])
            elif op == "Slice":
                sources = [operand(position) for position in range(len(node.input))]
                if len(sources) >= 3 and all(
                    source is not None for source in sources[:3]
                ):
                    starts = [int(item) for item in sources[1][0].reshape(-1)]
                    ends = [int(item) for item in sources[2][0].reshape(-1)]
                    axes = (
                        [int(item) for item in sources[3][0].reshape(-1)]
                        if len(sources) > 3 and sources[3] is not None
                        else list(range(len(starts)))
                    )
                    steps = (
                        [int(item) for item in sources[4][0].reshape(-1)]
                        if len(sources) > 4 and sources[4] is not None
                        else [1] * len(starts)
                    )
                    slices = [slice(None)] * sources[0][0].ndim
                    for start, end, axis, step in zip(starts, ends, axes, steps):
                        slices[axis] = slice(start, end, step)
                    value = sources[0][0][tuple(slices)]
                    for source in sources:
                        if source is not None:
                            dependencies.update(source[1])
            elif op == "Reshape":
                source = operand(0)
                target = operand(1)
                if source is not None and target is not None:
                    requested = tuple(int(item) for item in target[0].reshape(-1))
                    value = np.reshape(source[0], requested)
                    dependencies.update(source[1])
                    dependencies.update(target[1])
            elif op == "ConstantOfShape":
                source = operand(0)
                tensor = _attribute(node, "value")
                fill = (
                    self.numpy_helper.to_array(tensor).reshape(-1)
                    if tensor is not None else None
                )
                if (
                    source is not None
                    and self.dtype_of(node.input[0]) in {"int32", "int64"}
                    and fill is not None
                    and fill.size == 1
                    and fill.dtype.kind in {"i", "u"}
                ):
                    extents = [
                        self._shape_extent(item) for item in source[0].reshape(-1)
                    ]
                    if (
                        all(isinstance(item, int) and item >= 0 for item in extents)
                        and math.prod(int(item) for item in extents) <= 4096
                    ):
                        value = np.full(
                            tuple(int(item) for item in extents),
                            fill[0],
                            dtype=fill.dtype,
                        )
                        dependencies.update(source[1])
            elif op == "Equal":
                left = operand(0)
                right = operand(1)
                if (
                    left is not None
                    and right is not None
                    and self.dtype_of(node.input[0]) in {"int32", "int64"}
                    and self.dtype_of(node.input[1]) in {"int32", "int64"}
                ):
                    value = self._evaluate_symbolic_shape_equal(left[0], right[0])
                    dependencies.update(left[1])
                    dependencies.update(right[1])
            elif op == "Where":
                condition = operand(0)
                when_true = operand(1)
                when_false = operand(2)
                if (
                    condition is not None
                    and when_true is not None
                    and when_false is not None
                    and self.dtype_of(node.input[0]) == "bool"
                    and self.dtype_of(node.input[1]) in {"int32", "int64"}
                    and self.dtype_of(node.input[1]) == self.dtype_of(node.input[2])
                    and all(
                        isinstance(item, (bool, np.bool_))
                        for item in condition[0].reshape(-1)
                    )
                    and all(
                        self._shape_extent(item) is not None
                        for branch in (when_true[0], when_false[0])
                        for item in branch.reshape(-1)
                    )
                ):
                    value = np.where(
                        np.asarray(condition[0], dtype=bool),
                        when_true[0],
                        when_false[0],
                    )
                    dependencies.update(condition[1])
                    dependencies.update(when_true[1])
                    dependencies.update(when_false[1])
            elif op == "Expand":
                source = operand(0)
                target = operand(1)
                if (
                    source is not None
                    and target is not None
                    and self.dtype_of(node.input[0]) in {"int32", "int64"}
                    and self.dtype_of(node.input[1]) in {"int32", "int64"}
                ):
                    extents = [
                        self._shape_extent(item) for item in target[0].reshape(-1)
                    ]
                    if (
                        all(isinstance(item, int) and item > 0 for item in extents)
                        and math.prod(int(item) for item in extents) <= 4096
                    ):
                        value = np.broadcast_to(
                            source[0], tuple(int(item) for item in extents)
                        )
                        dependencies.update(source[1])
                        dependencies.update(target[1])
            elif op in {"Add", "Sub", "Mul", "Div"}:
                left = operand(0)
                right = operand(1)
                if left is not None and right is not None:
                    value = self._evaluate_symbolic_shape_arithmetic(
                        op, left[0], right[0]
                    )
                    dependencies.update(left[1])
                    dependencies.update(right[1])
        except (IndexError, TypeError, ValueError, OverflowError):
            value = None

        visiting.remove(resolved)
        if value is None:
            memo[resolved] = None
            return None
        if index not in self.folded:
            dependencies.add(index)
        result = (np.asarray(value, dtype=object), frozenset(dependencies))
        memo[resolved] = result
        return result

    def _structural_target_matches(
        self,
        node,
        target: np.ndarray,
        *,
        source_node: str,
    ) -> bool:
        requested = [
            self._shape_extent(value) for value in target.reshape(-1)
        ]
        if not requested or any(value is None for value in requested):
            return False
        requested = [value for value in requested if value is not None]
        output_shape = self.shape_of(node.output[0])
        if node.op_type == "Expand":
            try:
                expected = _broadcast_shape(
                    self.shape_of(node.input[0]), requested, node=source_node
                )
            except ExporterError:
                return False
            return self._replace_opaque_shape(node.output[0], expected)

        if node.op_type != "Reshape" or len(requested) != len(output_shape):
            return False
        allowzero = bool(int(_attribute(node, "allowzero", 0)))
        source_shape = self.shape_of(node.input[0])
        expected: list[int | str] = []
        inferred_axis: int | None = None
        for axis, dimension in enumerate(requested):
            if isinstance(dimension, str) or dimension > 0:
                expected.append(dimension)
            elif dimension == 0 and not allowzero and axis < len(source_shape):
                expected.append(source_shape[axis])
            elif dimension == -1 and inferred_axis is None:
                inferred_axis = axis
                expected.append(-1)
            else:
                return False
        if inferred_axis is not None:
            inferred = self._shape_factor_quotient(
                source_shape,
                [
                    dimension
                    for axis, dimension in enumerate(expected)
                    if axis != inferred_axis
                ],
            )
            if inferred is None:
                return False
            expected[inferred_axis] = inferred
        if not self._shape_products_equal(source_shape, expected):
            return False
        return self._replace_opaque_shape(node.output[0], expected)

    def _fold_invariant_symbolic_shape_programs(self) -> None:
        """Fold a numeric sink selected invariantly from a symbolic shape.

        This is intentionally narrower than ordinary constant folding. A
        bounded symbol may flow through the metadata evaluator, but only a
        Cast whose complete input contains exact integers (no remaining
        symbol) seeds executable constants. The producer-common
        ``Shape -> Slice(fixed suffix) -> Squeeze -> Cast -> Pow`` attention
        scale is admitted; selecting the dynamic batch extent is not.
        """

        producers = self._producer_map()
        consumers: dict[str, list[int]] = defaultdict(list)
        for index, node in enumerate(self.model.graph.node):
            for name in node.input:
                if name:
                    consumers[name].append(index)
        graph_outputs = {value.name for value in self.model.graph.output}
        memo: dict[str, tuple[np.ndarray, frozenset[int]] | None] = {}
        recognized: dict[int, frozenset[int]] = {}
        for index, node in enumerate(self.model.graph.node):
            if (
                index in self.folded
                or node.op_type != "Cast"
                or len(node.input) != 1
                or len(node.output) != 1
                or self.dtype_of(node.input[0]) not in {"int32", "int64"}
                or _onnx_dtype_name(int(_attribute(node, "to"))) != "float32"
            ):
                continue
            evaluated = self._evaluate_symbolic_shape_value(
                node.input[0],
                producers=producers,
                memo=memo,
                visiting=set(),
            )
            if evaluated is None or not evaluated[1]:
                continue
            exact = [self._shape_extent(item) for item in evaluated[0].reshape(-1)]
            if not all(isinstance(item, int) for item in exact):
                continue
            source_dtype = (
                np.int32 if self.dtype_of(node.input[0]) == "int32" else np.int64
            )
            source = np.asarray(exact, dtype=source_dtype).reshape(evaluated[0].shape)
            result = self._evaluate_constant(node, [source])
            if result is None:
                continue
            self.arrays[node.output[0]] = np.asarray(result)
            self.shape_map[node.output[0]] = list(np.asarray(result).shape)
            self.dtype_map[node.output[0]] = _runtime_dtype_for_array(
                np.asarray(result)
            )
            self.folded.add(index)
            recognized[index] = evaluated[1]

        if not recognized:
            return

        # Newly seeded Cast constants can make the downstream Pow and related
        # scalar arithmetic ordinary immutable constants.
        changed = True
        while changed:
            changed = False
            for index, node in enumerate(self.model.graph.node):
                if index in self.folded or not node.output:
                    continue
                inputs = [self.resolve(name) for name in node.input if name]
                if not inputs or not all(name in self.arrays for name in inputs):
                    continue
                result = self._evaluate_constant(
                    node, [self.arrays[name] for name in inputs]
                )
                if result is None:
                    continue
                results = result if isinstance(result, tuple) else (result,)
                if len(results) != len(node.output):
                    continue
                for name, array in zip(node.output, results):
                    self.arrays[name] = np.asarray(array)
                    self.shape_map[name] = list(np.asarray(array).shape)
                    self.dtype_map[name] = _runtime_dtype_for_array(np.asarray(array))
                self.folded.add(index)
                changed = True

        candidates = set().union(*recognized.values())
        changed = True
        while changed:
            changed = False
            for index in tuple(candidates):
                node = self.model.graph.node[index]
                if any(
                    output in graph_outputs
                    or any(
                        consumer not in candidates and consumer not in self.folded
                        for consumer in consumers.get(output, ())
                    )
                    for output in node.output
                    if output
                ):
                    candidates.remove(index)
                    changed = True
        admitted = set().union(*(
            dependencies
            for dependencies in recognized.values()
            if dependencies <= candidates
        )) if recognized else set()
        self.structural_shape_nodes.update(admitted)

    def _replace_opaque_shape(
        self,
        output: str,
        expected: Iterable[int | str],
    ) -> bool:
        """Refine only ONNX shape-inference ``unk__N`` dimensions."""

        candidate = list(expected)
        declared = self.shape_of(output)
        if len(candidate) != len(declared) or any(
            not (
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
            ) and not (
                isinstance(dimension, str)
                and self.shape_environment.get(dimension) is not None
            )
            for dimension in candidate
        ):
            return False
        if any(
            actual != refined
            and not (
                isinstance(actual, str)
                and re.fullmatch(r"unk__[0-9]+", actual) is not None
            )
            for actual, refined in zip(declared, candidate)
        ):
            return False
        self.shape_map[output] = candidate
        return True

    def _shape_factor_quotient(
        self,
        dividend: Iterable[int | str],
        divisor: Iterable[int | str],
    ) -> int | str | None:
        def factors(shape: Iterable[int | str]):
            fixed = 1
            symbols: dict[str, int] = defaultdict(int)
            for dimension in shape:
                if isinstance(dimension, int) and dimension > 0:
                    fixed *= dimension
                elif isinstance(dimension, str):
                    constraint = self.shape_environment.get(dimension)
                    if constraint is None:
                        return None
                    if constraint.min == constraint.max:
                        fixed *= constraint.min
                    else:
                        symbols[dimension] += 1
                else:
                    return None
            return fixed, symbols

        numerator = factors(dividend)
        denominator = factors(divisor)
        if numerator is None or denominator is None:
            return None
        numerator_fixed, numerator_symbols = numerator
        denominator_fixed, denominator_symbols = denominator
        if denominator_fixed <= 0 or numerator_fixed % denominator_fixed:
            return None
        for symbol, count in denominator_symbols.items():
            if numerator_symbols[symbol] < count:
                return None
            numerator_symbols[symbol] -= count
        remaining = [
            symbol
            for symbol, count in numerator_symbols.items()
            for _ in range(count)
        ]
        fixed = numerator_fixed // denominator_fixed
        if not remaining:
            return fixed if fixed > 0 else None
        if fixed == 1 and len(remaining) == 1:
            return remaining[0]
        return None

    def _logical_reshape_candidate(self, node) -> list[int | str] | None:
        if len(node.input) < 2:
            return None
        target = self._array(node.input[1])
        if target is None or np.asarray(target).ndim != 1:
            return None
        requested = [int(value) for value in np.asarray(target).reshape(-1)]
        source_shape = self.shape_of(node.input[0])
        allowzero = bool(int(_attribute(node, "allowzero", 0)))
        expected: list[int | str] = []
        inferred_axis: int | None = None
        for axis, dimension in enumerate(requested):
            if dimension > 0:
                expected.append(dimension)
            elif dimension == 0 and not allowzero and axis < len(source_shape):
                expected.append(source_shape[axis])
            elif dimension == -1 and inferred_axis is None:
                inferred_axis = axis
                expected.append(-1)
            else:
                return None
        if inferred_axis is not None:
            inferred = self._shape_factor_quotient(
                source_shape,
                [
                    dimension
                    for axis, dimension in enumerate(expected)
                    if axis != inferred_axis
                ],
            )
            if inferred is None:
                return None
            expected[inferred_axis] = inferred
        if not self._shape_products_equal(source_shape, expected):
            return None
        return expected

    def _logical_shape_candidate(self, node) -> list[int | str] | None:
        if not node.input or not node.output:
            return None
        input_shape = self.shape_of(node.input[0])
        if not input_shape:
            return None
        shape_preserving = {
            "Identity", "Cast", "Clip", "Not", "Relu", "Sigmoid", "Tanh",
            "Erf", "LayerNormalization", "Softmax", "LogSoftmax",
            "QuantizeLinear", "DequantizeLinear",
        }
        if node.op_type in shape_preserving:
            return input_shape
        if node.op_type == "Reshape":
            return self._logical_reshape_candidate(node)
        if node.op_type == "Transpose":
            permutation = list(
                _attribute(
                    node,
                    "perm",
                    list(reversed(range(len(input_shape)))),
                )
            )
            if sorted(permutation) == list(range(len(input_shape))):
                return [input_shape[axis] for axis in permutation]
            return None
        if node.op_type in {"Add", "Sub", "Mul", "Div", "Equal", "GreaterOrEqual"}:
            if len(node.input) < 2:
                return None
            try:
                return _broadcast_shape(
                    input_shape,
                    self.shape_of(node.input[1]),
                    node=_source_name(node, -1),
                )
            except ExporterError:
                return None
        if node.op_type == "Where" and len(node.input) == 3:
            try:
                values = _broadcast_shape(
                    self.shape_of(node.input[1]),
                    self.shape_of(node.input[2]),
                    node=_source_name(node, -1),
                )
                return _broadcast_shape(
                    input_shape, values, node=_source_name(node, -1)
                )
            except ExporterError:
                return None
        if node.op_type == "MatMul" and len(node.input) == 2:
            right_shape = self.shape_of(node.input[1])
            if (
                len(input_shape) < 2
                or len(right_shape) < 2
                or input_shape[-1] != right_shape[-2]
            ):
                return None
            try:
                batch = _broadcast_shape(
                    input_shape[:-2],
                    right_shape[:-2],
                    node=_source_name(node, -1),
                )
            except ExporterError:
                return None
            return [*batch, input_shape[-2], right_shape[-1]]
        if node.op_type == "Gemm" and len(node.input) >= 2:
            right_shape = self.shape_of(node.input[1])
            if len(input_shape) != 2 or len(right_shape) != 2:
                return None
            trans_a = bool(int(_attribute(node, "transA", 0)))
            trans_b = bool(int(_attribute(node, "transB", 0)))
            rows = input_shape[1] if trans_a else input_shape[0]
            inner_left = input_shape[0] if trans_a else input_shape[1]
            inner_right = right_shape[1] if trans_b else right_shape[0]
            columns = right_shape[0] if trans_b else right_shape[1]
            return [rows, columns] if inner_left == inner_right else None
        return None

    def _refine_opaque_passthrough_shapes(self) -> None:
        """Propagate exact operator shapes through producer-opaque ValueInfo."""

        for _ in range(len(self.model.graph.node) + 1):
            changed = False
            for node in self.model.graph.node:
                if not node.output:
                    continue
                candidate = self._logical_shape_candidate(node)
                if candidate is None:
                    continue
                before = self.shape_of(node.output[0])
                if self._replace_opaque_shape(node.output[0], candidate):
                    changed = changed or before != candidate
            if not changed:
                return
        raise ExporterError(Diagnostic(
            "VXONNX_SHAPE_REFINEMENT",
            "opaque ONNX shape refinement did not reach a fixed point",
            "logical-shapes",
            constraint="acyclic exact operator-shape propagation",
        ))

    def _recognize_structural_shape_programs(self) -> None:
        """Erase proven symbolic shape programs represented by output metadata.

        VolvoxAI Reshape/Expand operators carry their logical output shape in
        the closed runtime descriptor.  PyTorch ONNX commonly computes the same
        descriptor through a Shape/Gather/Concat subgraph.  Retain that program
        in the lossless source IR, but omit it from executable lowering only
        after evaluating it over canonical bounded symbols and proving that it
        matches the authored output shape.
        """

        producers = self._producer_map()
        consumers: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, node in enumerate(self.model.graph.node):
            for position, name in enumerate(node.input):
                if name:
                    consumers[name].append((index, position))
        graph_outputs = {value.name for value in self.model.graph.output}
        previous_state = None
        for _ in range(len(self.model.graph.node) + 2):
            memo: dict[str, tuple[np.ndarray, frozenset[int]] | None] = {}
            recognized: dict[tuple[int, int], frozenset[int]] = {}
            candidates: set[int] = set()
            for index, node in enumerate(self.model.graph.node):
                if node.op_type not in {"Reshape", "Expand"} or len(node.input) < 2:
                    continue
                target_name = self.resolve(node.input[1])
                if self._array(target_name) is not None:
                    continue
                evaluated = self._evaluate_symbolic_shape_value(
                    target_name,
                    producers=producers,
                    memo=memo,
                    visiting=set(),
                )
                if evaluated is None or not evaluated[1]:
                    continue
                if not self._structural_target_matches(
                    node,
                    evaluated[0],
                    source_node=_source_name(node, index),
                ):
                    continue
                recognized[(index, 1)] = evaluated[1]
                candidates.update(evaluated[1])

            # A program is executable metadata only when none of its values
            # escape to a data operand or public output. Prune transitively if
            # one does.
            changed = True
            while changed:
                changed = False
                for index in tuple(candidates):
                    node = self.model.graph.node[index]
                    escapes = any(
                        output in graph_outputs
                        or any(
                            consumer not in candidates
                            and consumer not in self.folded
                            and (consumer, position) not in recognized
                            for consumer, position in consumers.get(output, ())
                        )
                        for output in node.output
                        if output
                    )
                    if escapes:
                        candidates.remove(index)
                        changed = True

            self.structural_shape_targets = {
                target for target, dependencies in recognized.items()
                if dependencies <= candidates
            }
            self.structural_shape_nodes = set().union(
                *(
                    recognized[target]
                    for target in self.structural_shape_targets
                )
            ) if self.structural_shape_targets else set()
            self._refine_opaque_passthrough_shapes()
            state = (
                frozenset(self.structural_shape_targets),
                frozenset(self.structural_shape_nodes),
                tuple(sorted(
                    (name, tuple(shape)) for name, shape in self.shape_map.items()
                )),
            )
            if state == previous_state:
                return
            previous_state = state
        raise ExporterError(Diagnostic(
            "VXONNX_SHAPE_PROGRAM",
            "symbolic shape-program recognition did not reach a fixed point",
            "logical-shapes",
            constraint="acyclic immutable integer shape programs",
        ))

    def _canonical_concat_output_shape(
        self,
        node,
        input_shapes: list[list[int | str]],
        *,
        axis: int,
        dtype: str,
        source_node: str,
    ) -> list[int | str]:
        """Recover one producer-opaque Concat axis from the bounded ABI.

        ONNX shape inference often replaces ``P + 1`` with an anonymous
        ``unk__N`` dim_param on an intermediate Q/DQ edge even when the public
        output is canonically declared as ``R``.  An opaque name is never
        trusted.  It may be replaced only when exactly one output-only symbol
        in the caller's environment satisfies the complete affine Concat
        domain proof.
        """

        declared = self.shape_of(node.output[0])
        original_error: ExporterError | None = None
        try:
            return _concat_shape(
                input_shapes,
                axis,
                node=source_node,
                dtype=dtype,
                environment=self.shape_environment,
                declared_output=declared or None,
            )
        except ExporterError as error:
            original_error = error
        assert original_error is not None

        if not input_shapes:
            raise original_error
        rank = len(input_shapes[0])
        normalized_axis = axis + rank if axis < 0 else axis
        if (
            len(declared) != rank
            or normalized_axis < 0
            or normalized_axis >= rank
            or not isinstance(declared[normalized_axis], str)
            or re.fullmatch(r"unk__[0-9]+", declared[normalized_axis]) is None
            or self.shape_environment.get(declared[normalized_axis]) is not None
            or any(
                not (
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension > 0
                ) and not (
                    isinstance(dimension, str)
                    and self.shape_environment.get(dimension) is not None
                )
                for position, dimension in enumerate(declared)
                if position != normalized_axis
            )
        ):
            raise original_error

        input_symbols = {
            dimension
            for shape in input_shapes
            for dimension in shape
            if isinstance(dimension, str)
        }
        matches: list[list[int | str]] = []
        for constraint in self.shape_environment.dimensions:
            if constraint.name in input_symbols:
                continue
            candidate = list(declared)
            candidate[normalized_axis] = constraint.name
            try:
                inferred = _concat_shape(
                    input_shapes,
                    axis,
                    node=source_node,
                    dtype=dtype,
                    environment=self.shape_environment,
                    declared_output=candidate,
                )
            except ExporterError:
                continue
            matches.append(inferred)
        if len(matches) == 1:
            self.shape_map[node.output[0]] = list(matches[0])
            return list(matches[0])
        if len(matches) > 1:
            symbols = ", ".join(shape[normalized_axis] for shape in matches)
            raise ExporterError(Diagnostic(
                "VXCONCAT_SHAPE",
                f"{source_node} opaque output axis "
                f"{declared[normalized_axis]!r} has multiple exact canonical "
                f"symbol matches: {symbols}",
                "logical-shapes",
                source_node=source_node,
                source_op="Concat",
                constraint="one unique output-only affine symbol",
            ))
        raise original_error
