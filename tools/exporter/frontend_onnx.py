"""Static ONNX transformer/primitives frontend.

The frontend intentionally accepts only source constructs whose complete
semantics can be represented by existing VolvoxAI inference operators.  It
does not read checkpoints, invent calibration, or extract decode loops.
"""

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


_RUNTIME_DTYPES = {"float32", "int32", "int8", "uint8"}


def _attribute(node, name: str, default=None):
    import onnx

    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def _source_name(node, index: int) -> str:
    return node.name or f"{node.op_type}[{index}]"


def _onnx_dtype_name(element_type: int) -> str:
    import onnx

    values = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.INT8: "int8",
        onnx.TensorProto.UINT8: "uint8",
        onnx.TensorProto.BOOL: "bool",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.BFLOAT16: "bfloat16",
        onnx.TensorProto.DOUBLE: "float64",
    }
    return values.get(element_type, f"onnx:{element_type}")


def _runtime_dtype_for_array(array: np.ndarray) -> str:
    dtype = np.asarray(array).dtype
    if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
        return "float32"
    if dtype == np.int32:
        return "int32"
    if dtype == np.int64:
        return "int64"
    if dtype == np.int8:
        return "int8"
    if dtype == np.uint8:
        return "uint8"
    if dtype == np.bool_:
        return "bool"
    return str(dtype)


def _shape(value_info) -> list[int | str]:
    dimensions: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param") and dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(0)
    return dimensions


def _set_value_info_shape(value_info, shape: Iterable[int | str]) -> None:
    tensor_shape = value_info.type.tensor_type.shape
    del tensor_shape.dim[:]
    for extent in shape:
        dimension = tensor_shape.dim.add()
        if isinstance(extent, str):
            dimension.dim_param = extent
        else:
            dimension.dim_value = int(extent)


def _product(shape: Iterable[int | str], *, node: str = "shape algebra") -> int:
    dimensions = tuple(shape)
    result = 1
    for dimension in dimensions:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise ExporterError(Diagnostic(
                "VXONNX_SYMBOLIC_PRODUCT",
                f"{node} cannot collapse symbolic shape {list(dimensions)!r} into "
                "one v1 dimension",
                "logical-shapes",
                source_node=node,
                constraint="derived shape products require explicit canonical support",
            ))
        result *= dimension
    return result


def _normalize_axis(axis: int, rank: int, *, node: str) -> int:
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ExporterError(Diagnostic(
            "VXONNX_AXIS",
            f"{node} axis {axis} is outside rank {rank}",
            "canonicalize",
            source_node=node,
            constraint="axis within source rank",
        ))
    return normalized


def _broadcast_shape(
    left: list[int | str],
    right: list[int | str],
    *,
    node: str,
) -> list[int | str]:
    result: list[int | str] = []
    for offset in range(1, max(len(left), len(right)) + 1):
        a = left[-offset] if offset <= len(left) else 1
        b = right[-offset] if offset <= len(right) else 1
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ExporterError(Diagnostic(
                "VXONNX_BROADCAST",
                f"{node} cannot broadcast shapes {left} and {right}",
                "canonicalize",
                source_node=node,
                constraint="ONNX multidirectional broadcasting",
            ))
    return list(reversed(result))


def _concat_shape(
    shapes: Iterable[list[int | str]],
    axis: int,
    *,
    node: str,
    dtype: str = "float32",
    environment: ShapeEnvironment | None = None,
    declared_output: list[int | str] | None = None,
) -> list[int | str]:
    operands = [list(shape) for shape in shapes]
    if not operands:
        raise ExporterError(Diagnostic(
            "VXCONCAT_SHAPE", f"{node} has no concat operands", "logical-shapes",
            source_node=node,
        ))
    request: dict[str, Any] = {
        "environment": environment or ShapeEnvironment(()),
        "inputs": {
            f"input{position}": {"shape": shape, "dtype": dtype}
            for position, shape in enumerate(operands)
        },
        "params": {"axis": axis},
    }
    if declared_output:
        request["declaredOutputs"] = {
            "out": {"shape": list(declared_output), "dtype": dtype}
        }
    proof = prove_operator_shape_domain("Concat", request)
    if not proof.supported:
        diagnostic_code = (
            "VXDYNAMIC_CONCAT_AXIS"
            if proof.code == "UNPROVABLE_DYNAMIC_SHAPE_FORMULA"
            else "VXCONCAT_SHAPE"
        )
        raise ExporterError(Diagnostic(
            diagnostic_code,
            f"{node} failed canonical Concat whole-domain inference "
            f"({proof.code}): {proof.reason}",
            "logical-shapes",
            source_node=node,
            source_op="Concat",
            constraint=(
                "one dynamic Concat-axis symbol plus fixed extents requires "
                "one exact declared output-only affine symbol"
            ),
        ))
    return list(proof.outputs["out"].shape)


def _quantization(scale: np.ndarray, zero_point: np.ndarray, axis: int) -> dict[str, Any]:
    scales = np.asarray(scale, dtype=np.float32).reshape(-1)
    zero_points = np.asarray(zero_point).reshape(-1)
    if scales.size == 1:
        return {
            "scheme": "per_tensor",
            "scale": float(scales[0]),
            "zero_point": int(zero_points[0] if zero_points.size else 0),
        }
    if zero_points.size == 1:
        zero_points = np.repeat(zero_points, scales.size)
    return {
        "scheme": "per_axis",
        "axis": int(axis),
        "scales": [float(value) for value in scales],
        "zero_points": [int(value) for value in zero_points],
    }


def _dequantize(array: np.ndarray, scale: np.ndarray, zero_point: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    scales = np.asarray(scale, dtype=np.float32).reshape(-1)
    zero_points = np.asarray(zero_point, dtype=np.float32).reshape(-1)
    if scales.size == 1:
        zero = float(zero_points[0]) if zero_points.size else 0.0
        return (values - zero) * float(scales[0])
    if zero_points.size == 1:
        zero_points = np.repeat(zero_points, scales.size)
    broadcast = [1] * values.ndim
    broadcast[axis] = scales.size
    return (values - zero_points.reshape(broadcast)) * scales.reshape(broadcast)


def _to_i32_checked(array: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(array)
    if values.size:
        minimum = int(values.min())
        maximum = int(values.max())
        if minimum < -(2**31) or maximum > 2**31 - 1:
            raise ExporterError(Diagnostic(
                "VXDTYPE_RANGE",
                f"{label} contains an integer outside int32 range",
                "dtype-legalize",
                source_node=label,
                constraint="lossless INT64 to INT32 legalization",
            ))
    return values.astype(np.int32)


class _Names:
    def __init__(self):
        self.mapping: dict[str, str] = {}
        self.used: set[str] = set()
        self.next_value = 0
        self.next_weight = 0

    def bind(self, source: str, exported: str) -> str:
        if source in self.mapping and self.mapping[source] != exported:
            raise ExporterError(f"source tensor {source!r} has conflicting exported names")
        if exported in self.used and self.mapping.get(source) != exported:
            raise ExporterError(f"exported tensor name {exported!r} is not unique")
        self.mapping[source] = exported
        self.used.add(exported)
        return exported

    def get(self, source: str, *, weight: bool = False, preferred: Optional[str] = None) -> str:
        if source in self.mapping:
            return self.mapping[source]
        if preferred:
            base = preferred
        elif weight:
            base = f"w{self.next_weight}"
            self.next_weight += 1
        else:
            base = f"v{self.next_value}"
            self.next_value += 1
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)[:96].strip("._-") or "tensor"
        candidate = base
        suffix = 1
        while candidate in self.used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        return self.bind(source, candidate)


@dataclass
class _DQ:
    raw: str
    scale: str
    zero_point: str
    axis: int


@dataclass
class _Attention:
    source_node: str
    q: str
    k: str
    v: str
    output: str
    heads: int
    scale: float
    causal: bool
    mask: Optional[str]
    skip: frozenset[int]


@dataclass
class _GeluRegion:
    source_node: str
    input: str
    output: str
    skip: frozenset[int]


@dataclass
class _GroupNormRegion:
    source_node: str
    input: str
    output: str
    weight: str
    bias: str
    groups: int
    epsilon: float
    skip: frozenset[int]


@dataclass
class _QArgMaxRegion:
    source_node: str
    raw_input: str
    output: str
    axis: int
    input_quantization: dict[str, Any]
    skip: frozenset[int]


@dataclass
class _QLinearRegion:
    source_node: str
    op_type: str
    source_op: str
    raw_input: str
    raw_weight: str
    output: str
    weight_out_in: np.ndarray
    bias_i32: np.ndarray
    input_quantization: dict[str, Any]
    weight_quantization: dict[str, Any]
    output_quantization: dict[str, Any]
    output_dtype: str
    skip: frozenset[int]


@dataclass
class _QBatchMatMulRegion:
    source_node: str
    raw_a: str
    raw_b: str
    output: str
    a_quantization: dict[str, Any]
    b_quantization: dict[str, Any]
    output_quantization: dict[str, Any]
    output_dtype: str
    output_shape: list[int]
    skip: frozenset[int]


@dataclass
class _QConvRegion:
    source_node: str
    raw_input: str
    raw_weight: str
    output: str
    weight_ohwi: np.ndarray
    bias_i32: np.ndarray
    input_quantization: dict[str, Any]
    weight_quantization: dict[str, Any]
    output_quantization: dict[str, Any]
    output_dtype: str
    input_nhwc_shape: list[int]
    output_nchw_shape: list[int]
    output_nhwc_shape: list[int]
    params: dict[str, Any]
    skip: frozenset[int]


class OnnxCompiler:
    def __init__(
        self,
        model_path: str,
        *,
        weight_dtype: str = "auto",
        output_names: Optional[Iterable[str]] = None,
        image_normalizations: Optional[Iterable[str]] = None,
        input_shapes: Optional[Mapping[str, Iterable[int]]] = None,
        dimension_bounds: Optional[Mapping[str, Any]] = None,
        anonymous_dimension_bounds: Optional[Mapping[tuple[str, int], Any]] = None,
        input_dtypes: Optional[Mapping[str, str]] = None,
        output_dtypes: Optional[Mapping[str, str]] = None,
        specialize_inputs: Optional[Mapping[str, int]] = None,
        registered_domains: Optional[Iterable[str]] = None,
    ):
        import onnx
        from onnx import numpy_helper

        self.onnx = onnx
        self.numpy_helper = numpy_helper
        self.model_path = str(model_path)
        # Parsing is a lossless, checked stage of its own.  The target-aware
        # lowerer below may recognize regions, but it can no longer be the
        # first representation of the source graph or erase the 1:1 oracle.
        self.source_ir = import_onnx_source(self.model_path)
        self.model = onnx.load(self.model_path)
        # A domain name is not a semantic registration.  Until a custom-domain
        # lowering registry carries an exact schema/opset contract, accepting a
        # familiar op spelling would silently reinterpret producer-defined
        # semantics as ai.onnx semantics.
        self.registered_domains = set(registered_domains or ())
        unsupported_domains = sorted({
            node.domain
            for node in self.model.graph.node
            if node.domain not in {"", "ai.onnx"}
        })
        if unsupported_domains:
            raise ExporterError(Diagnostic(
                "VXONNX_DOMAIN",
                f"unsupported ONNX operator domain(s): {', '.join(unsupported_domains)}",
                "frontend",
                constraint="standard ai.onnx operators with registered source semantics",
            ))
        self.weight_dtype = weight_dtype
        self.output_names_requested = list(output_names) if output_names is not None else None
        requested_normalizations = list(image_normalizations or ())
        if requested_normalizations:
            raise ExporterError(Diagnostic(
                "VXIMAGE_NORMALIZATION",
                "application image preprocessing cannot be embedded in the "
                "closed volvox-graph/v1 input schema",
                "usage",
                constraint="configure preprocessing in the consuming application",
            ))
        self.input_shape_bindings = {key: [int(value) for value in values] for key, values in (input_shapes or {}).items()}
        # Explicit static input views can resolve a source dim_param for every
        # public value that carries the same ONNX symbol.  Keep that proof
        # separate from dimension_bounds: a caller binding specializes the
        # package to one extent, while bounds describe a retained dynamic ABI.
        self.static_dimension_bindings: dict[str, int] = {}
        self.dimension_bounds = dict(dimension_bounds or {})
        self.anonymous_dimension_bounds = dict(anonymous_dimension_bounds or {})
        self.input_dtype_bindings = dict(input_dtypes or {})
        self.output_dtype_bindings = dict(output_dtypes or {})
        self.specializations = {key: int(value) for key, value in (specialize_inputs or {}).items()}
        self.arrays = {initializer.name: numpy_helper.to_array(initializer) for initializer in self.model.graph.initializer}
        self.initializer_names = set(self.arrays)
        self.alias: dict[str, str] = {}
        self.dq: dict[str, _DQ] = {}
        self.folded: set[int] = set()
        self.structural_shape_nodes: set[int] = set()
        self.structural_shape_targets: set[tuple[int, int]] = set()
        self.identity_bool_casts: set[int] = set()
        self.identity_int64_casts: set[int] = set()
        self.shape_map: dict[str, list[int | str]] = {}
        self.dtype_map: dict[str, str] = {}
        self.names = _Names()
        self.weights: dict[str, np.ndarray] = {}
        self.tensor_quantization: dict[str, dict[str, Any]] = {}
        self.nodes: list[dict[str, Any]] = []
        self.features: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.abi_changes: list[dict[str, Any]] = []
        self.node_sources: list[dict[str, str]] = []
        self.publication_report: dict[str, Any] = {}
        self.skipped_nodes = 0
        self.opset = 0
        self.attention_replacements: dict[int, _Attention] = {}
        self.attention_skip: set[int] = set()
        self.gelu_replacements: dict[int, _GeluRegion] = {}
        self.gelu_skip: set[int] = set()
        self.groupnorm_replacements: dict[int, _GroupNormRegion] = {}
        self.groupnorm_skip: set[int] = set()
        self.qargmax_replacements: dict[int, _QArgMaxRegion] = {}
        self.qlinear_replacements: dict[int, _QLinearRegion] = {}
        self.qbatch_matmul_replacements: dict[int, _QBatchMatMulRegion] = {}
        self.qconv_replacements: dict[int, _QConvRegion] = {}
        self.w8a8_skip: set[int] = set()
        self.activation_quantization: dict[str, dict[str, Any]] = {}
        for value in self.model.opset_import:
            if value.domain in ("", "ai.onnx"):
                self.opset = int(value.version)
        self.graph_output_names = {value.name for value in self.model.graph.output}
        self._apply_shape_bindings()
        self._infer_shapes()
        self.publication_ir = import_onnx_source(
            self.model,
            dimension_bounds=self.dimension_bounds,
            anonymous_dimension_bounds=self.anonymous_dimension_bounds,
        )
        self.shape_environment = self.publication_ir.shape_environment
        for name, tensor in self.publication_ir.tensors.items():
            if tensor.shape and all(
                isinstance(dimension, (int, str)) and dimension != 0
                for dimension in tensor.shape
            ):
                self.shape_map[name] = list(tensor.shape)
        self._load_constants()
        self._prepare_public_names()
        self._validate_dtype_binding_names()
        self._apply_specializations_and_fold()
        self._recognize_structural_shape_programs()
        self._recognize_identity_abi_casts()
        self._recognize_w8a8()
        self._recognize_exact_gelu()
        self._recognize_group_norm()
        self._recognize_features()
        self._recognize_attention()

    def resolve(self, name: str) -> str:
        seen = set()
        while name in self.alias:
            if name in seen:
                raise ExporterError(f"alias cycle involving {name!r}")
            seen.add(name)
            name = self.alias[name]
        return name

    def shape_of(self, name: str) -> list[int | str]:
        resolved = self.resolve(name)
        if resolved in self.arrays:
            return list(np.asarray(self.arrays[resolved]).shape)
        return list(self.shape_map.get(resolved) or self.shape_map.get(name) or ())

    def dtype_of(self, name: str) -> str:
        resolved = self.resolve(name)
        if resolved in self.arrays:
            return _runtime_dtype_for_array(np.asarray(self.arrays[resolved]))
        return self.dtype_map.get(resolved) or self.dtype_map.get(name) or "float32"

    def execution_dtype_of(self, name: str) -> str:
        """Return the runtime dtype after an explicit public-input ABI binding."""

        resolved = self.resolve(name)
        graph_inputs = [
            value for value in self.model.graph.input
            if value.name not in self.initializer_names
        ]
        for index, value in enumerate(graph_inputs):
            if resolved != value.name:
                continue
            canonical = self.names.get(value.name) if value.name in self.names.mapping else f"input{index}"
            binding = self._binding(self.input_dtype_bindings, value.name, canonical)
            return str(binding or self.dtype_of(resolved))
        return self.dtype_of(resolved)

    def _require_execution_dtype(
        self,
        name: str,
        allowed: Iterable[str],
        *,
        source_node: str,
        source_op: str,
        role: str,
    ) -> str:
        dtype = self.execution_dtype_of(name)
        accepted = set(allowed)
        if dtype not in accepted:
            raise ExporterError(Diagnostic(
                "VXOPERAND_DTYPE",
                f"{source_node} {role} has {dtype} execution dtype; expected "
                f"{', '.join(sorted(accepted))}",
                "dtype-legalize",
                source_node=source_node,
                source_op=source_op,
                constraint="no implicit execution-dtype reinterpretation",
            ))
        return dtype

    def _binding(self, bindings: Mapping[str, Any], source_name: str, canonical_name: str):
        has_source = source_name in bindings
        has_canonical = canonical_name in bindings
        if has_source and has_canonical and bindings[source_name] != bindings[canonical_name]:
            raise ExporterError(Diagnostic(
                "VXCLI_BINDING",
                f"conflicting bindings for {source_name!r} and {canonical_name!r}",
                "usage",
                source_node=source_name,
            ))
        if has_source:
            return bindings[source_name]
        if has_canonical:
            return bindings[canonical_name]
        return None

    def _apply_shape_bindings(self) -> None:
        initializer_names = set(self.arrays)
        inputs = [value for value in self.model.graph.input if value.name not in initializer_names]
        known = {value.name for value in inputs}
        explicit_shapes: dict[str, list[int]] = {}
        for index, value in enumerate(inputs):
            canonical = f"input{index}"
            binding = self._binding(self.input_shape_bindings, value.name, canonical)
            if binding is None:
                continue
            shape = list(binding)
            if not shape or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape):
                raise ExporterError(Diagnostic(
                    "VXSHAPE_BINDING",
                    f"input shape binding for {value.name!r} must contain positive dimensions",
                    "bind-shapes",
                    source_node=value.name,
                ))
            declared_shape = _shape(value)
            if len(shape) != len(declared_shape):
                raise ExporterError(Diagnostic(
                    "VXSHAPE_BINDING",
                    f"input shape binding for {value.name!r} has rank {len(shape)}, "
                    f"but the source input declares rank {len(declared_shape)}",
                    "bind-shapes",
                    source_node=value.name,
                    constraint="explicit static bindings preserve the source input rank",
                ))
            for axis, (declared, extent) in enumerate(zip(declared_shape, shape)):
                if isinstance(declared, int) and declared > 0 and declared != extent:
                    raise ExporterError(Diagnostic(
                        "VXSHAPE_BINDING",
                        f"input shape binding for {value.name!r} axis {axis} has "
                        f"extent {extent}, but the source input declares {declared}",
                        "bind-shapes",
                        source_node=value.name,
                        constraint="explicit static bindings preserve source concrete extents",
                    ))
                if not isinstance(declared, str):
                    continue
                previous = self.static_dimension_bindings.get(declared)
                if previous is not None and previous != extent:
                    raise ExporterError(Diagnostic(
                        "VXSHAPE_BINDING",
                        f"input shape bindings assign conflicting extents {previous} "
                        f"and {extent} to source symbol {declared!r}",
                        "bind-shapes",
                        source_node=value.name,
                        constraint="one exact extent per source dimension symbol",
                    ))
                self.static_dimension_bindings[declared] = extent
            explicit_shapes[value.name] = shape

        # A repeated dim_param is one source-level equality constraint.  Once
        # an explicit binding proves its exact extent, specialize every public
        # input occurrence before ONNX shape inference so downstream values
        # receive the same concrete ABI.
        for value in inputs:
            declared_shape = _shape(value)
            shape = explicit_shapes.get(value.name)
            if shape is None:
                shape = [
                    self.static_dimension_bindings.get(dimension, dimension)
                    if isinstance(dimension, str) else dimension
                    for dimension in declared_shape
                ]
            if shape != declared_shape:
                _set_value_info_shape(value, shape)
        unknown = sorted(set(self.input_shape_bindings) - known - {f"input{i}" for i in range(len(inputs))})
        if unknown:
            raise ExporterError(Diagnostic(
                "VXSHAPE_UNKNOWN",
                f"shape binding names unknown input(s): {', '.join(unknown)}",
                "bind-shapes",
            ))

    def _infer_shapes(self) -> None:
        from onnx import shape_inference

        source_public_shapes = {
            value.name: _shape(value)
            for value in [*self.model.graph.input, *self.model.graph.output]
            if value.type.HasField("tensor_type")
        }
        try:
            self.model = shape_inference.infer_shapes(self.model, strict_mode=True, data_prop=True)
        except Exception as error:
            raise ExporterError(Diagnostic(
                "VXONNX_SHAPE_INFERENCE",
                f"ONNX shape inference failed: {error}",
                "infer-shapes",
            )) from error
        environment: ShapeEnvironment | None = None
        for value in [*self.model.graph.input, *self.model.graph.output]:
            source_shape = source_public_shapes.get(value.name)
            if source_shape is None or not value.type.HasField("tensor_type"):
                continue
            inferred_shape = _shape(value)
            if len(source_shape) != len(inferred_shape):
                raise ExporterError(Diagnostic(
                    "VXONNX_PUBLIC_SHAPE_CONFLICT",
                    f"public value {value.name!r} declared rank {len(source_shape)}, "
                    f"but ONNX shape inference produced rank {len(inferred_shape)}",
                    "infer-shapes",
                    source_node=value.name,
                    constraint="source-declared public rank must agree with ONNX inference",
                ))
            reconciled = list(inferred_shape)
            for axis, (declared, inferred) in enumerate(
                zip(source_shape, inferred_shape)
            ):
                if declared == inferred or declared == 0:
                    continue
                if isinstance(declared, int):
                    raise ExporterError(Diagnostic(
                        "VXONNX_PUBLIC_SHAPE_CONFLICT",
                        f"public value {value.name!r} axis {axis} declared concrete "
                        f"extent {declared}, but ONNX shape inference produced {inferred!r}",
                        "infer-shapes",
                        source_node=value.name,
                        constraint="source-declared public concrete extents must agree with ONNX inference",
                    ))
                if not isinstance(inferred, int) or inferred <= 0:
                    raise ExporterError(Diagnostic(
                        "VXONNX_PUBLIC_SHAPE_CONFLICT",
                        f"public value {value.name!r} axis {axis} declared symbol "
                        f"{declared!r}, but ONNX shape inference produced {inferred!r}",
                        "infer-shapes",
                        source_node=value.name,
                        constraint="a changed public symbol requires an exact singleton domain proof",
                    ))
                bound_extent = self.static_dimension_bindings.get(declared)
                if bound_extent is not None:
                    if bound_extent != inferred:
                        raise ExporterError(Diagnostic(
                            "VXONNX_PUBLIC_SHAPE_CONFLICT",
                            f"public value {value.name!r} axis {axis} declared symbol "
                            f"{declared!r}, but explicit input binding proves "
                            f"extent {bound_extent} and ONNX inferred {inferred}",
                            "infer-shapes",
                            source_node=value.name,
                            constraint="public symbol inference must equal its explicit static binding",
                        ))
                    # Keep the inferred concrete extent.  Unlike a singleton
                    # retained domain, an input_shapes binding deliberately
                    # authors a static package ABI.
                    continue
                if environment is None:
                    try:
                        environment = ShapeEnvironment(tuple(
                            {"name": name, **dict(bounds)}
                            for name, bounds in self.dimension_bounds.items()
                        ))
                    except Exception as error:
                        raise ExporterError(Diagnostic(
                            "VXONNX_PUBLIC_SHAPE_CONFLICT",
                            "public symbolic shape reconciliation has an invalid "
                            f"dimension environment: {error}",
                            "infer-shapes",
                            source_node=value.name,
                            constraint="canonical bounded-shape environment",
                        )) from error
                constraint = environment.get(declared)
                if (
                    constraint is None
                    or constraint.min != inferred
                    or constraint.max != inferred
                ):
                    raise ExporterError(Diagnostic(
                        "VXONNX_PUBLIC_SHAPE_CONFLICT",
                        f"public value {value.name!r} axis {axis} declared symbol "
                        f"{declared!r}, but inferred extent {inferred} is not exactly "
                        "equivalent over its complete bounded domain",
                        "infer-shapes",
                        source_node=value.name,
                        constraint="changed public symbols require min=max=inferred extent",
                    ))
                reconciled[axis] = declared
            if reconciled != inferred_shape:
                _set_value_info_shape(value, reconciled)
        values = [*self.model.graph.input, *self.model.graph.value_info, *self.model.graph.output]
        for value in values:
            if not value.type.HasField("tensor_type"):
                continue
            self.shape_map[value.name] = _shape(value)
            self.dtype_map[value.name] = _onnx_dtype_name(value.type.tensor_type.elem_type)
        for name, array in self.arrays.items():
            self.shape_map[name] = list(np.asarray(array).shape)
            self.dtype_map[name] = _runtime_dtype_for_array(np.asarray(array))

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
                expected.append(output_shape[axis])
            else:
                return False
        if not self._shape_products_equal(source_shape, expected):
            return False
        return self._replace_opaque_shape(node.output[0], expected)

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

        # A program is executable metadata only when none of its values escape
        # to a data operand or public output.  Prune transitively if one does.
        changed = True
        while changed:
            changed = False
            for index in tuple(candidates):
                node = self.model.graph.node[index]
                escapes = any(
                    output in graph_outputs
                    or any(
                        consumer not in candidates
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
            *(recognized[target] for target in self.structural_shape_targets)
        ) if self.structural_shape_targets else set()
        self._refine_opaque_passthrough_shapes()

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

    def _execution_storage_dtype(self, name: str) -> str:
        """Return the canonical storage dtype for one live ONNX value."""

        resolved = self.resolve(name)
        value = self._array(resolved)
        if value is not None:
            dtype = _runtime_dtype_for_array(np.asarray(value))
            if dtype == "bool":
                return "int32"
            if dtype == "int64":
                _to_i32_checked(np.asarray(value), label=resolved)
                return "int32"
            return dtype
        return self.execution_dtype_of(resolved)

    def _execution_ref(
        self,
        name: str,
        *,
        allowed: Iterable[str],
        source_node: str,
        source_op: str,
        role: str,
    ) -> tuple[str, str]:
        resolved = self.resolve(name)
        dtype = self._execution_storage_dtype(resolved)
        accepted = set(allowed)
        if dtype not in accepted:
            raise ExporterError(Diagnostic(
                "VXOPERAND_DTYPE",
                f"{source_node} {role} has {dtype} execution dtype; expected "
                f"{', '.join(sorted(accepted))}",
                "dtype-legalize",
                source_node=source_node,
                source_op=source_op,
                constraint="no implicit execution-dtype reinterpretation",
            ))
        reference = (
            self.ensure_weight(resolved)
            if self._array(resolved) is not None
            else self.names.get(resolved)
        )
        return reference, dtype

    def _broadcast_execution_ref(
        self,
        name: str,
        *,
        output_shape: list[int],
        dtype: str,
        stem: str,
        role: str,
        source_node: str,
        source_op: str,
    ) -> str:
        """Make an ONNX broadcast operand exact-shape for canonical runtimes."""

        resolved = self.resolve(name)
        input_shape = self.shape_of(resolved)
        if input_shape == output_shape:
            return (
                self.ensure_weight(resolved)
                if self._array(resolved) is not None
                else self.names.get(resolved)
            )
        if _broadcast_shape(input_shape, output_shape, node=source_node) != output_shape:
            raise ExporterError(Diagnostic(
                "VXBROADCAST_SHAPE",
                f"{source_node} {role} shape {input_shape} cannot broadcast to {output_shape}",
                "canonicalize",
                source_node=source_node,
                source_op=source_op,
            ))
        value = self._array(resolved)
        derived = f"{stem}__{role}_expanded"
        if value is not None and all(isinstance(axis, int) for axis in output_shape):
            try:
                expanded = np.broadcast_to(np.asarray(value), output_shape)
            except (TypeError, ValueError) as error:
                raise ExporterError(Diagnostic(
                    "VXBROADCAST_SHAPE",
                    f"{source_node} {role} cannot broadcast to {output_shape}: {error}",
                    "canonicalize",
                    source_node=source_node,
                    source_op=source_op,
                )) from error
            return self.ensure_weight(
                resolved,
                array=np.ascontiguousarray(expanded),
                preferred=self.names.get(derived, weight=True),
            )
        self._add_node(
            "Expand",
            {
                "input": (
                    self.ensure_weight(resolved)
                    if value is not None
                    else self.names.get(resolved)
                )
            },
            derived,
            shape=output_shape,
            dtype=dtype,
            source_node=source_node,
            source_op=source_op,
        )
        return self.names.get(derived)

    def _output_dtype(self, source_name: str, default: Optional[str] = None) -> str:
        dtype = default or self.dtype_of(source_name)
        if source_name in self.graph_output_names:
            canonical = self.names.get(source_name)
            binding = self._binding(self.output_dtype_bindings, source_name, canonical)
            source_dtype = self.dtype_map.get(source_name, dtype)
            if binding is not None:
                if binding not in _RUNTIME_DTYPES:
                    raise ExporterError(Diagnostic(
                        "VXOUTPUT_DTYPE", f"unsupported output dtype binding {binding!r}", "usage", source_node=source_name
                    ))
                if source_dtype != binding and not (
                    source_dtype in {"bool", "int64"} and binding == "int32"
                ):
                    raise ExporterError(Diagnostic(
                        "VXOUTPUT_DTYPE_REWRITE",
                        f"cannot rewrite public output {source_name!r} from {source_dtype} to {binding}",
                        "dtype-legalize",
                        source_node=source_name,
                        constraint=(
                            "only BOOL masks or range-bounded ONNX integers may "
                            "be represented as int32"
                        ),
                    ))
                if source_dtype != binding:
                    self.abi_changes.append({
                        "kind": "output-dtype",
                        "name": source_name,
                        "source": source_dtype,
                        "exported": binding,
                    })
                dtype = binding
            elif source_dtype in {"bool", "int64"} and dtype == "int32":
                raise ExporterError(Diagnostic(
                    "VXOUTPUT_DTYPE_BINDING",
                    f"public output {source_name!r} is {source_dtype}; bind it explicitly to int32",
                    "dtype-legalize",
                    source_node=source_name,
                    constraint=f"--output-dtype {source_name}=int32",
                ))
        return dtype

    def _add_node(
        self,
        op_type: str,
        inputs: Mapping[str, str],
        output: str,
        *,
        shape: Optional[Iterable[int | str]] = None,
        dtype: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        source_node: str,
        source_op: str,
        quantization: Optional[Mapping[str, Any]] = None,
    ) -> str:
        output_shape = list(shape if shape is not None else self.shape_of(output))
        if not output_shape or any(
            not (
                isinstance(value, int) and not isinstance(value, bool) and value > 0
            ) and not (
                isinstance(value, str)
                and self.shape_environment.get(value) is not None
            )
            for value in output_shape
        ):
            raise ExporterError(Diagnostic(
                "VXBOUNDED_SHAPE",
                f"{source_node} output {output!r} has unresolved shape {output_shape}",
                "logical-shapes",
                source_node=source_node,
                source_op=source_op,
                constraint="positive constants or caller-bounded dimension symbols",
            ))
        output_dtype = self._output_dtype(output, dtype)
        if output_dtype not in _RUNTIME_DTYPES:
            raise ExporterError(Diagnostic(
                "VXEXEC_DTYPE",
                f"{source_node} output {output!r} has unsupported live dtype {output_dtype!r}",
                "dtype-legalize",
                source_node=source_node,
                source_op=source_op,
            ))
        exported_output = self.names.get(output)
        canonical_params = dict(params or {})
        if op_type in {"Reshape", "Expand"}:
            canonical_params.setdefault("shape", list(output_shape))
        node_id = f"node_{len(self.nodes)}"
        node = {
            "id": node_id,
            "opType": op_type,
            "inputs": dict(inputs),
            "outputs": {"out": {
                "tensor": exported_output,
                "dtype": output_dtype,
                "shape": output_shape,
            }},
            "params": canonical_params,
        }
        if quantization is not None:
            self.tensor_quantization[exported_output] = dict(quantization)
        self.nodes.append(node)
        self.node_sources.append({
            "id": node_id,
            "source_node": source_node,
            "source_op": source_op,
        })
        self.shape_map[output] = output_shape
        self.dtype_map[output] = output_dtype
        return exported_output

    def _linear_weight(self, node, weight_name: str, source_node: str):
        resolved = self.resolve(weight_name)
        dq = self.dq.get(weight_name) or self.dq.get(resolved)
        if dq and self._array(dq.raw) is not None:
            raw = np.asarray(self._array(dq.raw))
            scale = self._array(dq.scale)
            zero = self._array(dq.zero_point) if dq.zero_point else None
            if raw.ndim != 2 or scale is None:
                raise ExporterError(Diagnostic(
                    "VXWEIGHT_QDQ",
                    f"{source_node} weight-only QDQ requires immutable rank-2 "
                    "weight and scale tensors",
                    "quant-fold",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            zero_value = (
                np.asarray(zero)
                if zero is not None
                else np.asarray(0, dtype=raw.dtype)
            )
            decoded = _dequantize(
                raw, np.asarray(scale), zero_value, dq.axis,
            ).astype(np.float32)
            exported = self.ensure_weight(
                resolved,
                array=decoded,
                preferred=f"{self.names.get(resolved, weight=True)}_decoded_f32",
            )
            self.features["dequantized_weight_linear"].append({
                "source_node": source_node,
                "reason": "closed-v1 Linear has no runtime affine weight ports",
            })
            return {"weight": exported}, "din_dout", "fp32"
        value = self._array(weight_name)
        if value is None or np.asarray(value).ndim != 2:
            raise ExporterError(Diagnostic(
                "VXLINEAR_RHS",
                f"{source_node} requires an immutable rank-2 right-hand matrix",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
                constraint="constant linear-compatible RHS",
            ))
        return {"weight": self.ensure_weight(weight_name)}, "din_dout", "fp32"

    def _emit_matmul(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) != 2:
            raise ExporterError(Diagnostic("VXMATMUL_ARITY", f"{source_node} requires two inputs", "lower", source_node=source_node))
        input_name = self.resolve(node.input[0])
        right_name = self.resolve(node.input[1])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        right_value = self._array(right_name)
        if right_value is None or np.asarray(right_value).ndim != 2:
            self._require_execution_dtype(
                right_name, {"float32"}, source_node=source_node,
                source_op=node.op_type, role="right operand",
            )
            left_shape = self.shape_of(input_name)
            right_shape = self.shape_of(right_name)
            if len(left_shape) < 2 or len(right_shape) < 2:
                raise ExporterError(Diagnostic(
                    "VXBATCH_MATMUL_RANK",
                    f"{source_node} dynamic MatMul requires rank >= 2 operands",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            if left_shape[-1] != right_shape[-2]:
                raise ExporterError(Diagnostic(
                    "VXBATCH_MATMUL_INNER",
                    f"{source_node} has incompatible inner dimensions "
                    f"{left_shape[-1]} and {right_shape[-2]}",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            batch_shape = _broadcast_shape(
                left_shape[:-2], right_shape[:-2], node=source_node
            )
            expected_shape = [*batch_shape, left_shape[-2], right_shape[-1]]
            output_shape = self.shape_of(node.output[0])
            if output_shape != expected_shape:
                raise ExporterError(Diagnostic(
                    "VXBATCH_MATMUL_SHAPE",
                    f"{source_node} inferred output {output_shape} does not match "
                    f"canonical batch MatMul shape {expected_shape}",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            left_ref = (
                self.ensure_weight(input_name)
                if self._array(input_name) is not None
                else self.names.get(input_name)
            )
            right_ref = (
                self.ensure_weight(right_name)
                if right_value is not None
                else self.names.get(right_name)
            )
            self._add_node(
                "BatchMatMul",
                {"a": left_ref, "b": right_ref},
                node.output[0],
                shape=expected_shape,
                dtype="float32",
                source_node=source_node,
                source_op=node.op_type,
            )
            return
        inputs, layout, mode = self._linear_weight(node, node.input[1], source_node)
        inputs = {"input": self.names.get(input_name), **inputs}
        self._add_node(
            "Linear", inputs, node.output[0], params={"weight_layout": layout},
            source_node=source_node, source_op=node.op_type,
        )
        if mode == "w8a32":
            self.features["w8a32_linear"].append({"source_node": source_node})

    def _emit_gemm(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) < 2:
            raise ExporterError(Diagnostic("VXGEMM_ARITY", f"{source_node} has too few inputs", "lower", source_node=source_node))
        self._require_execution_dtype(
            self.resolve(node.input[0]), {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        if int(_attribute(node, "transA", 0)) != 0:
            raise ExporterError(Diagnostic(
                "VXGEMM_TRANSA", f"{source_node} transA requires an explicit supported transpose", "lower", source_node=source_node
            ))
        trans_b = int(_attribute(node, "transB", 0))
        alpha = float(_attribute(node, "alpha", 1.0))
        beta = float(_attribute(node, "beta", 1.0))
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise ExporterError(Diagnostic("VXGEMM_SCALE", f"{source_node} alpha/beta must be finite", "canonicalize"))
        weight_name = node.input[1]
        dq = self.dq.get(weight_name) or self.dq.get(self.resolve(weight_name))
        if dq:
            raw = np.asarray(self._array(dq.raw))
            # Normalize source B to [d_in,d_out] before the W8A32 helper transposes it.
            normalized = raw.T if trans_b else raw
            synthetic = f"{node.output[0]}__gemm_weight_dq"
            self.arrays[synthetic] = normalized
            if alpha <= 0:
                raise ExporterError(Diagnostic(
                    "VXGEMM_ALPHA_QUANT",
                    f"{source_node} quantized Gemm requires positive alpha, got {alpha}",
                    "quant-fold",
                    source_node=source_node,
                    constraint="positive weight scales",
                ))
            scale = np.asarray(self._array(dq.scale), dtype=np.float32) * alpha
            scale_name = f"{synthetic}__scale"
            zero_name = f"{synthetic}__zero"
            self.arrays[scale_name] = scale
            zero_value = self._array(dq.zero_point) if dq.zero_point else None
            self.arrays[zero_name] = np.asarray(
                zero_value if zero_value is not None else np.zeros(scale.shape, dtype=raw.dtype)
            )
            self.dq[synthetic] = _DQ(dq.raw, scale_name, zero_name, 1 if np.asarray(scale).size > 1 else 0)
            # Store normalized raw under the descriptor's raw name for this derived path.
            raw_name = f"{synthetic}__raw"
            self.arrays[raw_name] = normalized
            self.dq[synthetic] = _DQ(raw_name, scale_name, zero_name, 1 if np.asarray(scale).size > 1 else 0)
            inputs, layout, mode = self._linear_weight(node, synthetic, source_node)
        else:
            weight = self._array(weight_name)
            if weight is None or np.asarray(weight).ndim != 2:
                raise ExporterError(Diagnostic("VXGEMM_WEIGHT", f"{source_node} B must be immutable rank-2", "lower"))
            normalized = np.asarray(weight).T if trans_b else np.asarray(weight)
            normalized = np.asarray(normalized, dtype=np.float32) * alpha
            exported = self.ensure_weight(weight_name, array=normalized, preferred=f"{self.names.get(self.resolve(weight_name), weight=True)}_gemm")
            inputs, layout, mode = {"weight": exported}, "din_dout", "fp32"
        inputs = {"input": self.names.get(self.resolve(node.input[0])), **inputs}
        if len(node.input) > 2 and node.input[2]:
            bias = self._array(node.input[2])
            output_shape = self.shape_of(node.output[0])
            d_out = output_shape[-1] if output_shape else 0
            bias_shape = tuple(np.asarray(bias).shape) if bias is not None else None
            scalar_bias = bias_shape in {(), (1,)}
            trailing_bias = bias_shape in {(d_out,), (1, d_out)}
            if bias is None or not (scalar_bias or trailing_bias):
                raise ExporterError(Diagnostic(
                    "VXGEMM_BIAS",
                    f"{source_node} C is not a representable scalar or trailing [d_out] bias",
                    "canonicalize",
                    source_node=source_node,
                    constraint="scalar, [d_out], or [1,d_out] C broadcast",
                ))
            bias = np.asarray(bias, dtype=np.float32).reshape(-1) * beta
            if bias.size == 1 and d_out > 1:
                bias = np.repeat(bias, d_out)
            inputs["bias"] = self.ensure_weight(node.input[2], array=bias, preferred=f"{self.names.get(self.resolve(node.input[2]), weight=True)}_gemm")
        self._add_node(
            "Linear", inputs, node.output[0], params={"weight_layout": layout},
            source_node=source_node, source_op=node.op_type,
        )
        if mode == "w8a32":
            self.features["w8a32_linear"].append({"source_node": source_node})

    def _emit_softmax(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="input",
        )
        input_shape = self.shape_of(input_name)
        if not input_shape:
            raise ExporterError(Diagnostic("VXSOFTMAX_SHAPE", f"{source_node} input shape is unresolved", "logical-shapes"))
        axis_default = 1 if self.opset <= 12 else -1
        axis = _normalize_axis(int(_attribute(node, "axis", axis_default)), len(input_shape), node=source_node)
        current = input_name
        op_type = "LogSoftmax" if node.op_type == "LogSoftmax" else "Softmax"
        if self.opset <= 12:
            rows = _product(input_shape[:axis]) if axis else 1
            width = _product(input_shape[axis:])
            reshaped = f"{node.output[0]}__legacy_rows"
            self._add_node(
                "Reshape", {"input": self.names.get(current)}, reshaped, shape=[rows, width], dtype="float32",
                source_node=source_node, source_op=node.op_type,
            )
            normalized = f"{node.output[0]}__normalized"
            self._add_node(
                op_type, {"input": self.names.get(reshaped)}, normalized, shape=[rows, width], dtype="float32",
                params={"axis": -1}, source_node=source_node, source_op=node.op_type,
            )
            self._add_node(
                "Reshape", {"input": self.names.get(normalized)}, node.output[0], shape=input_shape, dtype="float32",
                source_node=source_node, source_op=node.op_type,
            )
            return
        if axis != len(input_shape) - 1:
            permutation = [value for value in range(len(input_shape)) if value != axis] + [axis]
            transposed_shape = [input_shape[value] for value in permutation]
            transposed = f"{node.output[0]}__axis_last"
            self._add_node(
                "Transpose", {"input": self.names.get(current)}, transposed, shape=transposed_shape, dtype="float32",
                params={"perm": permutation}, source_node=source_node, source_op=node.op_type,
            )
            normalized = f"{node.output[0]}__normalized"
            self._add_node(
                op_type, {"input": self.names.get(transposed)}, normalized, shape=transposed_shape, dtype="float32",
                params={"axis": -1}, source_node=source_node, source_op=node.op_type,
            )
            inverse = [permutation.index(value) for value in range(len(permutation))]
            self._add_node(
                "Transpose", {"input": self.names.get(normalized)}, node.output[0], shape=input_shape, dtype="float32",
                params={"perm": inverse}, source_node=source_node, source_op=node.op_type,
            )
            return
        self._add_node(
            op_type, {"input": self.names.get(current)}, node.output[0], shape=input_shape, dtype="float32",
            params={"axis": -1}, source_node=source_node, source_op=node.op_type,
        )

    def _reduction_axes(self, node) -> list[int]:
        axes = _attribute(node, "axes")
        if axes is None and len(node.input) > 1 and node.input[1]:
            value = self._array(node.input[1])
            axes = list(np.asarray(value).reshape(-1)) if value is not None else None
        return [int(value) for value in axes] if axes is not None else list(range(len(self.shape_of(node.input[0]))))

    def _emit_reduction(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="input",
        )
        input_shape = self.shape_of(input_name)
        axes = self._reduction_axes(node)
        if len(axes) != 1:
            raise ExporterError(Diagnostic(
                "VXREDUCE_AXES", f"{source_node} requires exactly one reduction axis", "lower", source_node=source_node
            ))
        axis = _normalize_axis(axes[0], len(input_shape), node=source_node)
        keepdims = bool(int(_attribute(node, "keepdims", 1)))
        current = input_name
        current_shape = input_shape
        permutation = list(range(len(input_shape)))
        if axis != len(input_shape) - 1:
            permutation = [value for value in range(len(input_shape)) if value != axis] + [axis]
            current_shape = [input_shape[value] for value in permutation]
            transposed = f"{node.output[0]}__axis_last"
            self._add_node(
                "Transpose", {"input": self.names.get(current)}, transposed, shape=current_shape, dtype="float32",
                params={"perm": permutation}, source_node=source_node, source_op=node.op_type,
            )
            current = transposed
        reduced_shape = current_shape[:-1] + ([1] if keepdims else [])
        reduced = node.output[0] if axis == len(input_shape) - 1 or not keepdims else f"{node.output[0]}__reduced"
        self._add_node(
            node.op_type, {"input": self.names.get(current)}, reduced, shape=reduced_shape, dtype="float32",
            params={"axis": -1, "keepdims": keepdims}, source_node=source_node, source_op=node.op_type,
        )
        if reduced != node.output[0]:
            inverse = [permutation.index(value) for value in range(len(permutation))]
            self._add_node(
                "Transpose", {"input": self.names.get(reduced)}, node.output[0], shape=self.shape_of(node.output[0]), dtype="float32",
                params={"perm": inverse}, source_node=source_node, source_op=node.op_type,
            )

    def _emit_gather(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        data_name = self.resolve(node.input[0])
        indices_name = self.resolve(node.input[1])
        axis = _normalize_axis(int(_attribute(node, "axis", 0)), len(self.shape_of(data_name)), node=source_node)
        data = self._array(data_name)
        indices = self._array(indices_name)
        self._require_execution_dtype(
            data_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="data",
        )
        if indices is None:
            self._require_execution_dtype(
                indices_name, {"int32"}, source_node=source_node,
                source_op=node.op_type, role="indices",
            )
        if indices is not None:
            indices_name = f"{indices_name}__i32"
            self.arrays[indices_name] = _to_i32_checked(indices, label=source_node)
        if indices is not None and np.asarray(indices).ndim == 0:
            data_shape = self.shape_of(data_name)
            if not isinstance(data_shape[axis], int):
                raise ExporterError(Diagnostic(
                    "VXDYNAMIC_GATHER_INDEX",
                    f"{source_node} cannot normalize a scalar index against "
                    f"symbolic axis {data_shape[axis]!r}",
                    "logical-shapes",
                    source_node=source_node,
                    source_op=node.op_type,
                    constraint="bounded-domain scalar-index proof is not implemented",
                ))
            selected = int(np.asarray(indices).item())
            if selected < 0:
                selected += data_shape[axis]
            if selected < 0 or selected >= data_shape[axis]:
                raise ExporterError(Diagnostic(
                    "VXGATHER_INDEX",
                    f"{source_node} scalar index is out of range for axis {axis}",
                    "canonicalize",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            data_ref = (
                self.ensure_weight(data_name)
                if data is not None
                else self.names.get(data_name)
            )
            slice_output = f"{node.output[0]}__scalar_slice"
            slice_shape = list(data_shape)
            slice_shape[axis] = 1
            self._add_node(
                "Slice",
                {"input": data_ref},
                slice_output,
                shape=slice_shape,
                dtype="float32",
                params={
                    "starts": [selected], "ends": [selected + 1],
                    "axes": [axis], "steps": [1],
                },
                source_node=source_node,
                source_op=node.op_type,
            )
            self._add_node(
                "Squeeze",
                {"input": self.names.get(slice_output)},
                node.output[0],
                shape=self.shape_of(node.output[0]),
                dtype="float32",
                params={"axes": [axis]},
                source_node=source_node,
                source_op=node.op_type,
            )
            return
        if data is not None and np.asarray(data).ndim == 2 and axis == 0:
            self._add_node(
                "Embedding",
                {"input": self.ensure_weight(indices_name) if indices is not None else self.names.get(indices_name),
                 "weight": self.ensure_weight(data_name)},
                node.output[0],
                dtype="float32",
                source_node=source_node,
                source_op=node.op_type,
            )
            return
        data_ref = self.ensure_weight(data_name) if data is not None else self.names.get(data_name)
        indices_ref = self.ensure_weight(indices_name) if indices is not None else self.names.get(indices_name)
        self._add_node(
            "Gather", {"input": data_ref, "indices": indices_ref}, node.output[0],
            dtype=self.dtype_of(data_name), params={"axis": axis}, source_node=source_node, source_op=node.op_type,
        )

    def _emit_i32_comparison(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) != 2:
            raise ExporterError(Diagnostic(
                "VXCOMPARE_ARITY",
                f"{source_node} requires exactly two operands",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        left_ref, _ = self._execution_ref(
            node.input[0],
            allowed={"int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="left operand",
        )
        right_ref, _ = self._execution_ref(
            node.input[1],
            allowed={"int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="right operand",
        )
        inferred = _broadcast_shape(
            self.shape_of(node.input[0]),
            self.shape_of(node.input[1]),
            node=source_node,
        )
        output_shape = self.shape_of(node.output[0]) or inferred
        if output_shape != inferred:
            raise ExporterError(Diagnostic(
                "VXCOMPARE_SHAPE",
                f"{source_node} output shape {output_shape} does not match "
                f"broadcast result {inferred}",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        self._add_node(
            node.op_type,
            {"a": left_ref, "b": right_ref},
            node.output[0],
            shape=output_shape,
            dtype="int32",
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_not(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_ref, _ = self._execution_ref(
            node.input[0],
            allowed={"int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="input",
        )
        self._add_node(
            "Not",
            {"input": input_ref},
            node.output[0],
            dtype="int32",
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_where(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) != 3:
            raise ExporterError(Diagnostic(
                "VXWHERE_ARITY",
                f"{source_node} requires condition, x, and y",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        condition_ref, _ = self._execution_ref(
            node.input[0],
            allowed={"int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="condition",
        )
        x_ref, x_dtype = self._execution_ref(
            node.input[1],
            allowed={"float32", "int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="x",
        )
        y_ref, y_dtype = self._execution_ref(
            node.input[2],
            allowed={"float32", "int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="y",
        )
        if x_dtype != y_dtype:
            raise ExporterError(Diagnostic(
                "VXWHERE_DTYPE",
                f"{source_node} x/y execution dtypes differ: {x_dtype} and {y_dtype}",
                "dtype-legalize",
                source_node=source_node,
                source_op=node.op_type,
            ))
        condition_shape = self.shape_of(node.input[0])
        value_shape = _broadcast_shape(
            self.shape_of(node.input[1]),
            self.shape_of(node.input[2]),
            node=source_node,
        )
        output_shape = _broadcast_shape(condition_shape, value_shape, node=source_node)
        inferred_output = self.shape_of(node.output[0])
        if inferred_output and inferred_output != output_shape:
            raise ExporterError(Diagnostic(
                "VXWHERE_SHAPE",
                f"{source_node} output shape {inferred_output} does not match "
                f"broadcast result {output_shape}",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        stem = node.output[0]
        condition_ref = self._broadcast_execution_ref(
            node.input[0],
            output_shape=output_shape,
            dtype="int32",
            stem=stem,
            role="condition",
            source_node=source_node,
            source_op=node.op_type,
        )
        x_ref = self._broadcast_execution_ref(
            node.input[1],
            output_shape=output_shape,
            dtype=x_dtype,
            stem=stem,
            role="x",
            source_node=source_node,
            source_op=node.op_type,
        )
        y_ref = self._broadcast_execution_ref(
            node.input[2],
            output_shape=output_shape,
            dtype=x_dtype,
            stem=stem,
            role="y",
            source_node=source_node,
            source_op=node.op_type,
        )
        self._add_node(
            "Where",
            {"condition": condition_ref, "a": x_ref, "b": y_ref},
            node.output[0],
            shape=output_shape,
            dtype=x_dtype,
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_expand(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_ref, dtype = self._execution_ref(
            node.input[0],
            allowed={"float32", "int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="input",
        )
        target = self._array(node.input[1]) if len(node.input) > 1 else None
        output_shape = self.shape_of(node.output[0])
        target_shape = (
            [int(value) for value in np.asarray(target).reshape(-1)]
            if target is not None
            else []
        )
        if target is None and (index, 1) in self.structural_shape_targets:
            target_shape = list(output_shape)
        if (
            not target_shape
            or any(
                not (
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension > 0
                ) and not (
                    isinstance(dimension, str)
                    and self.shape_environment.get(dimension) is not None
                )
                for dimension in target_shape
            )
            or _broadcast_shape(
                self.shape_of(node.input[0]), target_shape, node=source_node
            ) != output_shape
        ):
            raise ExporterError(Diagnostic(
                "VXEXPAND_SHAPE",
                f"{source_node} requires an immutable target whose broadcast "
                f"result equals its declared bounded output shape",
                "logical-shapes",
                source_node=source_node,
                source_op=node.op_type,
            ))
        if _broadcast_shape(self.shape_of(node.input[0]), output_shape, node=source_node) != output_shape:
            raise ExporterError(Diagnostic(
                "VXEXPAND_BROADCAST",
                f"{source_node} input cannot broadcast to {output_shape}",
                "canonicalize",
                source_node=source_node,
                source_op=node.op_type,
            ))
        self._add_node(
            "Expand",
            {"input": input_ref},
            node.output[0],
            shape=output_shape,
            dtype=dtype,
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_slice(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_ref, dtype = self._execution_ref(
            node.input[0],
            allowed={"float32", "int32"},
            source_node=source_node,
            source_op=node.op_type,
            role="input",
        )
        starts_value = self._array(node.input[1]) if len(node.input) > 1 else None
        ends_value = self._array(node.input[2]) if len(node.input) > 2 else None
        axes_value = self._array(node.input[3]) if len(node.input) > 3 and node.input[3] else None
        steps_value = self._array(node.input[4]) if len(node.input) > 4 and node.input[4] else None
        if starts_value is None or ends_value is None:
            raise ExporterError(Diagnostic(
                "VXSLICE_STATIC",
                f"{source_node} starts/ends must be immutable",
                "staticize",
                source_node=source_node,
                source_op=node.op_type,
            ))
        starts_raw = [int(value) for value in np.asarray(starts_value).reshape(-1)]
        ends_raw = [int(value) for value in np.asarray(ends_value).reshape(-1)]
        axes_raw = (
            [int(value) for value in np.asarray(axes_value).reshape(-1)]
            if axes_value is not None
            else list(range(len(starts_raw)))
        )
        steps_raw = (
            [int(value) for value in np.asarray(steps_value).reshape(-1)]
            if steps_value is not None
            else [1] * len(starts_raw)
        )
        if not (
            len(starts_raw) == len(ends_raw) == len(axes_raw) == len(steps_raw)
        ):
            raise ExporterError(Diagnostic(
                "VXSLICE_PARAMS",
                f"{source_node} slice parameter lengths differ",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        input_shape = self.shape_of(node.input[0])
        output_shape = self.shape_of(node.output[0])
        if not starts_raw:
            if input_shape != output_shape:
                raise ExporterError(Diagnostic(
                    "VXSLICE_SHAPE",
                    f"{source_node} empty slice parameters must preserve shape",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            self._add_node(
                "Identity",
                {"input": input_ref},
                node.output[0],
                shape=output_shape,
                dtype=dtype,
                source_node=source_node,
                source_op=node.op_type,
            )
            return
        normalized_axes: list[int] = []
        normalized_starts: list[int] = []
        normalized_ends: list[int] = []
        normalized_steps: list[int] = []
        expected_shape = list(input_shape)
        seen: set[int] = set()
        for raw_start, raw_end, raw_axis, raw_step in zip(
            starts_raw, ends_raw, axes_raw, steps_raw
        ):
            axis = _normalize_axis(raw_axis, len(input_shape), node=source_node)
            if not isinstance(input_shape[axis], int):
                raise ExporterError(Diagnostic(
                    "VXDYNAMIC_SLICE_AXIS",
                    f"{source_node} cannot normalize static slice bounds against "
                    f"symbolic axis {input_shape[axis]!r}",
                    "logical-shapes",
                    source_node=source_node,
                    source_op=node.op_type,
                    constraint="bounded-domain slice proof is not implemented",
                ))
            if axis in seen or raw_step <= 0:
                raise ExporterError(Diagnostic(
                    "VXSLICE_PARAMS",
                    f"{source_node} requires unique axes and positive steps",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            start, stop, step = slice(
                raw_start, raw_end, raw_step
            ).indices(input_shape[axis])
            length = len(range(start, stop, step))
            if length <= 0:
                raise ExporterError(Diagnostic(
                    "VXSLICE_EMPTY",
                    f"{source_node} produces an empty axis, which the runtime cannot represent",
                    "lower",
                    source_node=source_node,
                    source_op=node.op_type,
                ))
            expected_shape[axis] = length
            normalized_axes.append(axis)
            normalized_starts.append(start)
            normalized_ends.append(stop)
            normalized_steps.append(step)
            seen.add(axis)
        if output_shape != expected_shape:
            raise ExporterError(Diagnostic(
                "VXSLICE_SHAPE",
                f"{source_node} output shape {output_shape} does not match {expected_shape}",
                "lower",
                source_node=source_node,
                source_op=node.op_type,
            ))
        self._add_node(
            "Slice",
            {"input": input_ref},
            node.output[0],
            shape=output_shape,
            dtype=dtype,
            params={
                "starts": normalized_starts,
                "ends": normalized_ends,
                "axes": normalized_axes,
                "steps": normalized_steps,
            },
            source_node=source_node,
            source_op=node.op_type,
        )

    def _emit_argmax(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="input",
        )
        input_shape = self.shape_of(input_name)
        axis = _normalize_axis(int(_attribute(node, "axis", 0)), len(input_shape), node=source_node)
        select_last = int(_attribute(node, "select_last_index", 0))
        if select_last != 0:
            raise ExporterError(Diagnostic(
                "VXARGMAX_TIE", f"{source_node} select_last_index=1 is unsupported", "canonicalize",
                source_node=source_node, constraint="first-index ties",
            ))
        if node.output[0] in self.graph_output_names:
            canonical = self.names.get(node.output[0])
            binding = self._binding(self.output_dtype_bindings, node.output[0], canonical)
            if self.dtype_map.get(node.output[0]) == "int64" and binding != "int32":
                raise ExporterError(Diagnostic(
                    "VXARGMAX_PUBLIC_I64",
                    f"public ArgMax output {node.output[0]!r} is INT64; bind it explicitly to int32",
                    "dtype-legalize",
                    source_node=source_node,
                    source_op="ArgMax",
                    constraint="--output-dtype NAME=int32",
                ))
        self._add_node(
            "ArgMax", {"input": self.names.get(input_name)}, node.output[0], dtype="int32",
            params={"axis": axis, "keepdims": bool(int(_attribute(node, "keepdims", 1))), "select_last_index": 0},
            source_node=source_node, source_op=node.op_type,
        )

    def _require_public_i32_index(self, output: str, *, source_node: str, source_op: str) -> None:
        if output not in self.graph_output_names:
            return
        canonical = self.names.get(output)
        binding = self._binding(self.output_dtype_bindings, output, canonical)
        if self.dtype_map.get(output) == "int64" and binding != "int32":
            raise ExporterError(Diagnostic(
                "VXARGMAX_PUBLIC_I64",
                f"public ArgMax output {output!r} is INT64; bind it explicitly to int32",
                "dtype-legalize",
                source_node=source_node,
                source_op=source_op,
                constraint="--output-dtype NAME=int32",
            ))

    def _emit_qargmax_region(self, region: _QArgMaxRegion) -> None:
        self._require_public_i32_index(
            region.output, source_node=region.source_node, source_op="ArgMax",
        )
        self._add_node(
            "QArgMax",
            {"input": self.names.get(region.raw_input)},
            region.output,
            dtype="int32",
            params={"axis": region.axis},
            source_node=region.source_node,
            source_op="DequantizeLinear-ArgMax",
        )

    def _emit_qlinear_region(self, region: _QLinearRegion) -> None:
        weight = self.ensure_weight(
            region.raw_weight,
            array=region.weight_out_in,
            preferred=f"{self.names.get(region.raw_weight, weight=True)}_out_in",
        )
        bias = self.ensure_weight(
            f"{region.source_node}__bias_i32",
            array=region.bias_i32,
            preferred=f"{weight}_bias_i32",
        )
        self.tensor_quantization[weight] = dict(region.weight_quantization)
        self._add_node(
            region.op_type,
            {
                "input": self.names.get(region.raw_input),
                "weight": weight,
                "bias": bias,
            },
            region.output,
            dtype=region.output_dtype,
            quantization=region.output_quantization,
            source_node=region.source_node,
            source_op=region.source_op,
        )

    def _emit_qbatch_matmul_region(
        self, region: _QBatchMatMulRegion
    ) -> None:
        self._add_node(
            "QBatchMatMul",
            {
                "a": self.names.get(region.raw_a),
                "b": self.names.get(region.raw_b),
            },
            region.output,
            shape=region.output_shape,
            dtype=region.output_dtype,
            quantization=region.output_quantization,
            source_node=region.source_node,
            source_op="QDQ-MatMul-QuantizeLinear",
        )

    def _emit_qconv_region(self, region: _QConvRegion) -> None:
        input_nhwc = f"{region.output}__qconv_nhwc_input"
        self._add_node(
            "Transpose",
            {"input": self.names.get(region.raw_input)},
            input_nhwc,
            shape=region.input_nhwc_shape,
            dtype="uint8",
            quantization=region.input_quantization,
            params={"perm": [0, 2, 3, 1]},
            source_node=region.source_node,
            source_op="QDQ-Conv-QuantizeLinear",
        )
        weight = self.ensure_weight(
            region.raw_weight,
            array=region.weight_ohwi,
            preferred=f"{self.names.get(region.raw_weight, weight=True)}_ohwi",
        )
        bias = self.ensure_weight(
            f"{region.source_node}__bias_i32",
            array=region.bias_i32,
            preferred=f"{weight}_bias_i32",
        )
        self.tensor_quantization[weight] = dict(region.weight_quantization)
        output_nhwc = f"{region.output}__qconv_nhwc_output"
        self._add_node(
            "QConv2D",
            {
                "input": self.names.get(input_nhwc),
                "weight": weight,
                "bias": bias,
            },
            output_nhwc,
            shape=region.output_nhwc_shape,
            dtype=region.output_dtype,
            quantization=region.output_quantization,
            params=region.params,
            source_node=region.source_node,
            source_op="QDQ-Conv-QuantizeLinear",
        )
        self._add_node(
            "Transpose",
            {"input": self.names.get(output_nhwc)},
            region.output,
            shape=region.output_nchw_shape,
            dtype=region.output_dtype,
            quantization=region.output_quantization,
            params={"perm": [0, 3, 1, 2]},
            source_node=region.source_node,
            source_op="QDQ-Conv-QuantizeLinear",
        )

    def _attention_is_self(self, attention: _Attention) -> bool:
        if self.shape_of(attention.q) != self.shape_of(attention.k) or self.shape_of(attention.k) != self.shape_of(attention.v):
            return False
        producers = self._producer_map()
        projection_inputs = []
        for value in (attention.q, attention.k, attention.v):
            entry = producers.get(self.resolve(value))
            if not entry or entry[1].op_type not in {"MatMul", "Gemm"} or not entry[1].input:
                return False
            projection_inputs.append(self.resolve(entry[1].input[0]))
        return len(set(projection_inputs)) == 1

    def _emit_attention(self, attention: _Attention) -> None:
        for role, value in (("query", attention.q), ("key", attention.k), ("value", attention.v)):
            self._require_execution_dtype(
                value, {"float32"}, source_node=attention.source_node,
                source_op="MatMul-Softmax-MatMul", role=role,
            )
        mask_ref = None
        if attention.mask is not None:
            if self.dtype_of(attention.mask) != "int32":
                raise ExporterError(Diagnostic(
                    "VXATTENTION_MASK_DTYPE",
                    f"{attention.source_node} keep mask must be int32",
                    "dtype-legalize",
                    source_node=attention.source_node,
                ))
            mask_ref = self.ensure_weight(attention.mask) if self._array(attention.mask) is not None else self.names.get(attention.mask)
        params = {
            "heads": attention.heads,
            "causal": attention.causal,
            "scale": attention.scale,
        }
        if self._attention_is_self(attention):
            q_shape = self.shape_of(attention.q)
            if not q_shape or not isinstance(q_shape[-1], int):
                raise ExporterError(Diagnostic(
                    "VXDYNAMIC_ATTENTION_PACK",
                    f"{attention.source_node} cannot encode three times symbolic "
                    "attention width in v1",
                    "logical-shapes",
                    source_node=attention.source_node,
                ))
            packed = f"{attention.output}__packed_qkv"
            concat_inputs = {
                "input0": self.names.get(attention.q),
                "input1": self.names.get(attention.k),
                "input2": self.names.get(attention.v),
            }
            self._add_node(
                "Concat", concat_inputs, packed, shape=[*q_shape[:-1], q_shape[-1] * 3], dtype="float32",
                params={"axis": len(q_shape) - 1},
                source_node=attention.source_node, source_op="attention-pack",
            )
            inputs = {"qkv": self.names.get(packed)}
            if mask_ref is not None:
                inputs["mask"] = mask_ref
            self._add_node(
                "SDPA", inputs, attention.output, dtype="float32", params=params,
                source_node=attention.source_node, source_op="MatMul-Softmax-MatMul",
            )
            return
        inputs = {
            "q": self.names.get(attention.q),
            "k": self.names.get(attention.k),
            "v": self.names.get(attention.v),
        }
        if mask_ref is not None:
            inputs["mask"] = mask_ref
        self._add_node(
            "CrossSDPA", inputs, attention.output, dtype="float32", params=params,
            source_node=attention.source_node, source_op="MatMul-Softmax-MatMul",
        )

    def _nchw_to_nhwc(self, input_name: str, *, stem: str, source_node: str, source_op: str) -> tuple[str, list[int]]:
        shape = self.shape_of(input_name)
        if len(shape) != 4:
            raise ExporterError(Diagnostic(
                "VXLAYOUT_RANK", f"{source_node} requires a rank-4 NCHW input", "layout-lower", source_node=source_node
            ))
        output = f"{stem}__nhwc_input"
        output_shape = [shape[0], shape[2], shape[3], shape[1]]
        self._add_node(
            "Transpose", {"input": self.names.get(input_name)}, output, shape=output_shape, dtype="float32",
            params={"perm": [0, 2, 3, 1]}, source_node=source_node, source_op=source_op,
        )
        return output, output_shape

    def _nhwc_to_nchw(
        self,
        input_name: str,
        output_name: str,
        *,
        output_shape: list[int],
        source_node: str,
        source_op: str,
    ) -> None:
        self._add_node(
            "Transpose", {"input": self.names.get(input_name)}, output_name, shape=output_shape, dtype="float32",
            params={"perm": [0, 3, 1, 2]}, source_node=source_node, source_op=source_op,
        )

    def _emit_conv(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        if len(node.input) < 2:
            raise ExporterError(Diagnostic("VXCONV_ARITY", f"{source_node} has too few inputs", "lower"))
        self._require_execution_dtype(
            self.resolve(node.input[0]), {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        auto_pad = _attribute(node, "auto_pad", b"NOTSET")
        if isinstance(auto_pad, bytes):
            auto_pad = auto_pad.decode("utf-8")
        if auto_pad not in {"", "NOTSET"}:
            raise ExporterError(Diagnostic(
                "VXCONV_AUTOPAD", f"{source_node} auto_pad={auto_pad!r} must be resolved by the producer", "layout-lower"
            ))
        weight = self._array(node.input[1])
        if weight is None or np.asarray(weight).ndim != 4:
            raise ExporterError(Diagnostic(
                "VXCONV_WEIGHT", f"{source_node} requires an immutable rank-4 OIHW weight", "lower", source_node=source_node
            ))
        weight = np.asarray(weight)
        output_shape = self.shape_of(node.output[0])
        if len(output_shape) != 4:
            raise ExporterError(Diagnostic("VXCONV_OUTPUT", f"{source_node} output shape is not rank-4 NCHW", "logical-shapes"))
        pads = [int(value) for value in _attribute(node, "pads", [0, 0, 0, 0])]
        if len(pads) != 4 or pads[0] != pads[2] or pads[1] != pads[3]:
            raise ExporterError(Diagnostic(
                "VXCONV_PADS",
                f"{source_node} asymmetric padding {pads} is not admitted by the current portable Conv descriptor",
                "layout-lower",
                source_node=source_node,
                constraint="symmetric explicit spatial padding",
            ))
        input_name = self.resolve(node.input[0])
        nhwc_input, _ = self._nchw_to_nhwc(
            input_name, stem=node.output[0], source_node=source_node, source_op=node.op_type
        )
        # Emit the image-layout form the Conv2D microkernels index directly. OHWI
        # would force every runtime to transpose into HWIO/HWCM at load and hold a
        # second copy of the weight; the byte order is settled here instead.
        groups = int(_attribute(node, "group", 1))
        stem = self.names.get(self.resolve(node.input[1]), weight=True)
        if groups > 1 and weight.shape[1] == 1:
            # Depthwise: OIHW [C*M,1,kh,kw] -> HWCM [kh,kw,C,M]. Only this shape
            # reaches the specialised depthwise kernels; HWIO would fall back to
            # the generic grouped loop.
            multiplier, remainder = divmod(weight.shape[0], groups)
            if remainder:
                raise ExporterError(Diagnostic(
                    "VXCONV_WEIGHT",
                    f"{source_node} depthwise output channels {weight.shape[0]} are not a multiple of group {groups}",
                    "lower", source_node=source_node,
                ))
            image = np.ascontiguousarray(np.transpose(
                weight.reshape(groups, multiplier, weight.shape[2], weight.shape[3]),
                (2, 3, 0, 1),
            ))
            weight_layout = "HWCM"
        else:
            # OIHW [O,I/g,kh,kw] -> HWIO [kh,kw,I/g,O].
            image = np.ascontiguousarray(np.transpose(weight, (2, 3, 1, 0)))
            weight_layout = "HWIO"
        weight_ref = self.ensure_weight(
            node.input[1], array=image, preferred=f"{stem}_{weight_layout.lower()}"
        )
        inputs = {"input": self.names.get(nhwc_input), "weight": weight_ref}
        if len(node.input) > 2 and node.input[2]:
            bias = self._array(node.input[2])
            if bias is None or np.asarray(bias).ndim != 1 or np.asarray(bias).shape[0] != output_shape[1]:
                raise ExporterError(Diagnostic("VXCONV_BIAS", f"{source_node} bias must be immutable [C_out]", "lower"))
            inputs["bias"] = self.ensure_weight(node.input[2])
        nhwc_output = f"{node.output[0]}__nhwc_output"
        self._add_node(
            "Conv2D", inputs, nhwc_output,
            shape=[output_shape[0], output_shape[2], output_shape[3], output_shape[1]],
            dtype="float32",
            params={
                "stride": [int(value) for value in _attribute(node, "strides", [1, 1])],
                "dilation": [int(value) for value in _attribute(node, "dilations", [1, 1])],
                "groups": groups,
                "pads": pads,
                "padding": pads[:2],
                "data_layout": "NHWC",
                "weight_layout": weight_layout,
            },
            source_node=source_node,
            source_op=node.op_type,
        )
        self._nhwc_to_nchw(
            nhwc_output, node.output[0], output_shape=output_shape,
            source_node=source_node, source_op=node.op_type,
        )

    def _emit_group_norm(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        input_shape = self.shape_of(input_name)
        if len(input_shape) != 4:
            raise ExporterError(Diagnostic("VXGROUPNORM_RANK", f"{source_node} requires NCHW rank 4", "layout-lower"))
        channels = input_shape[1]
        if len(node.input) < 3:
            raise ExporterError(Diagnostic("VXGROUPNORM_AFFINE", f"{source_node} requires scale and bias", "lower"))
        weight = self._array(node.input[1])
        bias = self._array(node.input[2])
        if weight is None or bias is None or np.asarray(weight).shape != (channels,) or np.asarray(bias).shape != (channels,):
            raise ExporterError(Diagnostic(
                "VXGROUPNORM_AFFINE",
                f"{source_node} requires immutable per-channel [{channels}] scale and bias",
                "lower",
                source_node=source_node,
                constraint="runtime per-channel affine GroupNorm",
            ))
        groups = int(_attribute(node, "num_groups", _attribute(node, "groups", 0)))
        if groups <= 0 or channels % groups:
            raise ExporterError(Diagnostic("VXGROUPNORM_GROUPS", f"{source_node} has invalid group count {groups}", "lower"))
        nhwc_input, nhwc_shape = self._nchw_to_nhwc(
            input_name, stem=node.output[0], source_node=source_node, source_op=node.op_type
        )
        nhwc_output = f"{node.output[0]}__nhwc_output"
        self._add_node(
            "GroupNorm",
            {
                "input": self.names.get(nhwc_input),
                "weight": self.ensure_weight(node.input[1]),
                "bias": self.ensure_weight(node.input[2]),
            },
            nhwc_output,
            shape=nhwc_shape,
            dtype="float32",
            params={
                "num_groups": groups,
                "eps": float(_attribute(node, "epsilon", _attribute(node, "eps", 1e-5))),
            },
            source_node=source_node,
            source_op=node.op_type,
        )
        self._nhwc_to_nchw(
            nhwc_output, node.output[0], output_shape=self.shape_of(node.output[0]),
            source_node=source_node, source_op=node.op_type,
        )

    def _emit_group_norm_region(self, region: _GroupNormRegion) -> None:
        self._require_execution_dtype(
            region.input, {"float32"}, source_node=region.source_node,
            source_op="GroupNorm-pattern", role="activation",
        )
        nhwc_input, nhwc_shape = self._nchw_to_nhwc(
            region.input,
            stem=region.output,
            source_node=region.source_node,
            source_op="GroupNorm-pattern",
        )
        nhwc_output = f"{region.output}__nhwc_output"
        self._add_node(
            "GroupNorm",
            {
                "input": self.names.get(nhwc_input),
                "weight": self.ensure_weight(region.weight),
                "bias": self.ensure_weight(region.bias),
            },
            nhwc_output,
            shape=nhwc_shape,
            dtype="float32",
            params={
                "num_groups": region.groups,
                "eps": region.epsilon,
            },
            source_node=region.source_node,
            source_op="GroupNorm-pattern",
        )
        self._nhwc_to_nchw(
            nhwc_output,
            region.output,
            output_shape=self.shape_of(region.output),
            source_node=region.source_node,
            source_op="GroupNorm-pattern",
        )

    def _emit_pool(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        output_shape = self.shape_of(node.output[0])
        if len(node.output) > 1 and node.output[1]:
            raise ExporterError(Diagnostic("VXPOOL_INDICES", f"{source_node} observable pool indices are unsupported", "lower"))
        if int(_attribute(node, "ceil_mode", 0)) != 0:
            raise ExporterError(Diagnostic("VXPOOL_CEIL", f"{source_node} ceil_mode is unsupported", "lower"))
        dilations = [int(value) for value in _attribute(node, "dilations", [1, 1])]
        if dilations != [1, 1]:
            raise ExporterError(Diagnostic("VXPOOL_DILATION", f"{source_node} dilated pooling is unsupported", "lower"))
        pads = [int(value) for value in _attribute(node, "pads", [0, 0, 0, 0])]
        if len(pads) != 4 or pads[0] != pads[2] or pads[1] != pads[3]:
            raise ExporterError(Diagnostic("VXPOOL_PADS", f"{source_node} requires symmetric padding", "layout-lower"))
        if node.op_type == "AveragePool" and (
            int(_attribute(node, "count_include_pad", 0)) != 0 or _attribute(node, "auto_pad", b"NOTSET") not in {b"NOTSET", "NOTSET", b"", ""}
        ):
            raise ExporterError(Diagnostic("VXPOOL_AVERAGE", f"{source_node} average-pool padding semantics are unsupported", "lower"))
        nhwc_input, _ = self._nchw_to_nhwc(
            input_name, stem=node.output[0], source_node=source_node, source_op=node.op_type
        )
        nhwc_output = f"{node.output[0]}__nhwc_output"
        op_type = "MaxPool2D" if node.op_type == "MaxPool" else "AveragePool2D"
        self._add_node(
            op_type, {"input": self.names.get(nhwc_input)}, nhwc_output,
            shape=[output_shape[0], output_shape[2], output_shape[3], output_shape[1]], dtype="float32",
            params={
                "kernel": [int(value) for value in _attribute(node, "kernel_shape")],
                "stride": [int(value) for value in _attribute(node, "strides", [1, 1])],
                "pads": pads,
                "padding": pads[:2],
                "data_layout": "NHWC",
            }, source_node=source_node, source_op=node.op_type,
        )
        self._nhwc_to_nchw(
            nhwc_output, node.output[0], output_shape=output_shape,
            source_node=source_node, source_op=node.op_type,
        )

    def _emit_global_average_pool(self, node, index: int) -> None:
        source_node = _source_name(node, index)
        input_name = self.resolve(node.input[0])
        self._require_execution_dtype(
            input_name, {"float32"}, source_node=source_node,
            source_op=node.op_type, role="activation",
        )
        output_shape = self.shape_of(node.output[0])
        nhwc_input, _ = self._nchw_to_nhwc(
            input_name, stem=node.output[0], source_node=source_node, source_op=node.op_type
        )
        nhwc_output = f"{node.output[0]}__nhwc_output"
        self._add_node(
            "GlobalAveragePool", {"input": self.names.get(nhwc_input)}, nhwc_output,
            shape=[output_shape[0], output_shape[2], output_shape[3], output_shape[1]], dtype="float32",
            params={"data_layout": "NHWC"}, source_node=source_node, source_op=node.op_type,
        )
        self._nhwc_to_nchw(
            nhwc_output, node.output[0], output_shape=output_shape,
            source_node=source_node, source_op=node.op_type,
        )

    def _input_definitions(self) -> dict[str, dict[str, Any]]:
        definitions = {}
        graph_inputs = [value for value in self.model.graph.input if value.name not in self.initializer_names]
        specialized = set(self.specializations)
        for index, value in enumerate(graph_inputs):
            canonical = self.names.get(value.name)
            if value.name in self.arrays and any(key in {value.name, canonical} for key in specialized):
                continue
            shape = self.shape_map.get(value.name, [])
            if not shape or any(
                not (
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension > 0
                ) and not (
                    isinstance(dimension, str)
                    and self.shape_environment.get(dimension) is not None
                )
                for dimension in shape
            ):
                raise ExporterError(Diagnostic(
                    "VXINPUT_SHAPE",
                    f"public input {value.name!r} has unresolved shape {shape}; "
                    "provide a concrete input shape or caller bounds",
                    "logical-shapes",
                    source_node=value.name,
                ))
            source_dtype = self.dtype_map.get(value.name, "")
            binding = self._binding(self.input_dtype_bindings, value.name, canonical)
            dtype = binding or source_dtype
            if dtype not in _RUNTIME_DTYPES:
                raise ExporterError(Diagnostic(
                    "VXINPUT_DTYPE",
                    f"public input {value.name!r} has unsupported dtype {source_dtype!r}; provide an explicit representable --input-dtype",
                    "dtype-legalize",
                    source_node=value.name,
                ))
            if binding and binding != source_dtype:
                if not (
                    source_dtype in {"bool", "int64"} and binding == "int32"
                ):
                    raise ExporterError(Diagnostic(
                        "VXINPUT_DTYPE_REWRITE",
                        f"cannot rewrite public input {value.name!r} from {source_dtype} to {binding}",
                        "dtype-legalize",
                        source_node=value.name,
                    ))
                self.abi_changes.append({"kind": "input-dtype", "name": value.name, "source": source_dtype, "exported": binding})
            definitions[canonical] = {"shape": shape, "dtype": dtype}
            descriptor = self.activation_quantization.get(self.resolve(value.name))
            if descriptor is not None:
                if dtype not in {"int8", "uint8"}:
                    raise ExporterError(Diagnostic(
                        "VXQ_INPUT_DTYPE",
                        f"canonical quantization metadata for {value.name!r} requires I8/U8 storage",
                        "quant-fold",
                        source_node=value.name,
                    ))
                self.tensor_quantization[canonical] = dict(descriptor)
        return definitions

    def lower(
        self,
        *,
        allow_silu_numerical_migration: bool = False,
        allow_quantized_bias_folding_numerical_migration: bool = False,
        allow_static_qdq_qbatch_matmul_numerical_migration: bool = False,
        allow_static_qdq_groupnorm_silu_numerical_migration: bool = False,
        enable_static_qdq_layout_optimization: bool = True,
        enable_exact_common_subexpression_elimination: bool = False,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        consumers: dict[str, list[int]] = defaultdict(list)
        public_input_sources = {
            value.name for value in self.model.graph.input if value.name not in self.initializer_names
        }
        for index, node in enumerate(self.model.graph.node):
            for name in node.input:
                consumers[name].append(index)
        for index, node in enumerate(self.model.graph.node):
            if index in self.structural_shape_nodes:
                self.skipped_nodes += 1
                continue
            if index in self.qconv_replacements:
                self._emit_qconv_region(self.qconv_replacements[index])
                continue
            if index in self.qlinear_replacements:
                region = self.qlinear_replacements[index]
                self._emit_qlinear_region(region)
                continue
            if index in self.qbatch_matmul_replacements:
                region = self.qbatch_matmul_replacements[index]
                self._emit_qbatch_matmul_region(region)
                continue
            if index in self.qargmax_replacements:
                region = self.qargmax_replacements[index]
                self._emit_qargmax_region(region)
                continue
            if index in self.w8a8_skip:
                self.skipped_nodes += 1
                continue
            if index in self.groupnorm_replacements:
                region = self.groupnorm_replacements[index]
                self._emit_group_norm_region(region)
                self.skipped_nodes += len(region.skip) - 1
                continue
            if index in self.groupnorm_skip:
                continue
            if index in self.gelu_replacements:
                region = self.gelu_replacements[index]
                self._require_execution_dtype(
                    region.input, {"float32"}, source_node=region.source_node,
                    source_op="Erf-GELU", role="input",
                )
                self._add_node(
                    "GELU", {"input": self.names.get(region.input)}, region.output,
                    dtype="float32", params={"approximate": "none"},
                    source_node=region.source_node, source_op="Erf-GELU",
                )
                self.skipped_nodes += len(region.skip) - 1
                continue
            if index in self.gelu_skip:
                continue
            if index in self.attention_replacements:
                self._emit_attention(self.attention_replacements[index])
                self.skipped_nodes += len(self.attention_replacements[index].skip) - 1
                continue
            if index in self.attention_skip:
                continue
            if index in self.folded:
                self.skipped_nodes += 1
                continue
            source_node = _source_name(node, index)
            op = node.op_type
            output = node.output[0] if node.output else ""
            if op == "DequantizeLinear":
                if len(node.input) < 2:
                    raise ExporterError(Diagnostic("VXDQ_ARITY", f"{source_node} has too few inputs", "quant-fold"))
                raw = self.resolve(node.input[0])
                scale = self.resolve(node.input[1])
                zero = self.resolve(node.input[2]) if len(node.input) > 2 and node.input[2] else ""
                raw_dtype = self.dtype_of(raw)
                default_zero_dtype = {
                    "uint8": np.uint8,
                    "int8": np.int8,
                    "int32": np.int32,
                    "float32": np.float32,
                }.get(raw_dtype, np.int8)
                zero_array = self._array(zero) if zero else np.asarray(0, dtype=default_zero_dtype)
                axis = int(_attribute(node, "axis", 1))
                self.dq[output] = _DQ(raw, scale, zero, axis)
                if raw in public_input_sources:
                    descriptor = self._per_tensor_byte_descriptor(
                        scale_name=scale,
                        zero_name=zero,
                        storage_dtype=raw_dtype,
                    )
                    if descriptor is not None and not self._register_activation_quantization(raw, descriptor):
                        raise ExporterError(Diagnostic(
                            "VXQ_INPUT_DESCRIPTOR_CONFLICT",
                            f"public byte input {raw!r} is dequantized with conflicting affine descriptors",
                            "quant-fold",
                            source_node=source_node,
                            constraint="one immutable per-tensor descriptor per byte tensor",
                        ))
                if self._array(raw) is not None and self._array(scale) is not None:
                    self.arrays[output] = _dequantize(self._array(raw), self._array(scale), zero_array, axis).astype(np.float32)
                    self.shape_map[output] = list(self.arrays[output].shape)
                    self.dtype_map[output] = "float32"
                    # A fully immutable DQ value is a constant.  Consumers can
                    # either retain its raw descriptor (Linear/Gemm weight-only
                    # folding) or use the exactly decoded F32 initializer.
                    self.skipped_nodes += 1
                    continue
                scale_array = self._array(scale)
                if scale_array is None:
                    raise ExporterError(Diagnostic("VXDQ_SCALE", f"{source_node} scale must be immutable", "quant-fold"))
                zero_name = zero or f"{output}__zero"
                if not zero:
                    self.arrays[zero_name] = np.asarray(0, dtype=default_zero_dtype)
                self._add_node(
                    "DequantizeLinear",
                    {"input": self.names.get(raw), "scale": self.ensure_weight(scale, force_float32=True), "zero_point": self.ensure_weight(zero_name)},
                    output, shape=self.shape_of(raw), dtype="float32",
                    source_node=source_node, source_op=op,
                )
                continue
            if op == "QuantizeLinear":
                self._require_execution_dtype(
                    self.resolve(node.input[0]), {"float32"}, source_node=source_node,
                    source_op=op, role="input",
                )
                scale = self._array(node.input[1]) if len(node.input) > 1 else None
                zero = self._array(node.input[2]) if len(node.input) > 2 and node.input[2] else np.asarray(0, dtype=np.uint8)
                if scale is None or np.asarray(scale).size != 1 or np.asarray(zero).size != 1:
                    raise ExporterError(Diagnostic(
                        "VXQ_PER_AXIS_ACTIVATION", f"{source_node} requires per-tensor activation quantization", "quant-fold", source_node=source_node
                    ))
                dtype = _runtime_dtype_for_array(np.asarray(zero))
                descriptor = _quantization(scale, zero, int(_attribute(node, "axis", 1)))
                zero_name = node.input[2] if len(node.input) > 2 and node.input[2] else f"{output}__zero"
                if zero_name not in self.arrays:
                    self.arrays[zero_name] = np.asarray(zero)
                self._add_node(
                    "QuantizeLinear",
                    {"input": self.names.get(self.resolve(node.input[0])), "scale": self.ensure_weight(node.input[1], force_float32=True), "zero_point": self.ensure_weight(zero_name)},
                    output, shape=self.shape_of(node.input[0]), dtype=dtype,
                    quantization=descriptor, source_node=source_node,
                    source_op=op,
                )
                continue
            if op == "MatMul":
                self._emit_matmul(node, index)
            elif op == "Gemm":
                self._emit_gemm(node, index)
            elif op in {"Equal", "GreaterOrEqual"}:
                self._emit_i32_comparison(node, index)
            elif op == "Not":
                self._emit_not(node, index)
            elif op == "Where":
                self._emit_where(node, index)
            elif op == "Expand":
                self._emit_expand(node, index)
            elif op == "Slice":
                self._emit_slice(node, index)
            elif op == "Conv":
                self._emit_conv(node, index)
            elif op in {"GroupNorm", "GroupNormalization"}:
                self._emit_group_norm(node, index)
            elif op in {"MaxPool", "AveragePool"}:
                self._emit_pool(node, index)
            elif op == "GlobalAveragePool":
                self._emit_global_average_pool(node, index)
            elif op in {"Add", "Sub", "Mul", "Div"}:
                left = self.resolve(node.input[0])
                right = self.resolve(node.input[1])
                self._require_execution_dtype(
                    left, {"float32"}, source_node=source_node,
                    source_op=op, role="left operand",
                )
                self._require_execution_dtype(
                    right, {"float32"}, source_node=source_node,
                    source_op=op, role="right operand",
                )
                inferred = _broadcast_shape(
                    self.shape_of(left), self.shape_of(right), node=source_node,
                )
                output_shape = self.shape_of(output) or inferred
                if output_shape != inferred:
                    raise ExporterError(Diagnostic(
                        "VXBROADCAST_OUTPUT",
                        f"{source_node} output shape {output_shape} does not match "
                        f"broadcast result {inferred}",
                        "logical-shapes",
                        source_node=source_node,
                        source_op=op,
                    ))
                inputs = {
                    "a": self._broadcast_execution_ref(
                        left, output_shape=output_shape, dtype="float32",
                        stem=output, role="left", source_node=source_node,
                        source_op=op,
                    ),
                    "b": self._broadcast_execution_ref(
                        right, output_shape=output_shape, dtype="float32",
                        stem=output, role="right", source_node=source_node,
                        source_op=op,
                    ),
                }
                self._add_node(
                    op, inputs, output, shape=output_shape, dtype="float32",
                    source_node=source_node, source_op=op,
                )
            elif op in {"Softmax", "LogSoftmax"}:
                self._emit_softmax(node, index)
            elif op == "Gather":
                self._emit_gather(node, index)
            elif op in {"ReduceSum", "ReduceMean"}:
                self._emit_reduction(node, index)
            elif op == "ArgMax":
                self._emit_argmax(node, index)
            elif op in {"Gelu", "GELU"}:
                self._require_execution_dtype(
                    self.resolve(node.input[0]), {"float32"}, source_node=source_node,
                    source_op=op, role="input",
                )
                approximate = _attribute(node, "approximate", b"none")
                if isinstance(approximate, bytes):
                    approximate = approximate.decode("utf-8")
                if approximate not in {"none", "tanh"}:
                    raise ExporterError(Diagnostic("VXGELU_APPROX", f"{source_node} has unsupported approximation {approximate!r}", "lower"))
                self._add_node(
                    "GELU", {"input": self.names.get(self.resolve(node.input[0]))}, output,
                    dtype="float32", params={"approximate": approximate}, source_node=source_node, source_op=op,
                )
            elif op == "LayerNormalization":
                self._require_execution_dtype(
                    self.resolve(node.input[0]), {"float32"}, source_node=source_node,
                    source_op=op, role="activation",
                )
                axis = _normalize_axis(int(_attribute(node, "axis", -1)), len(self.shape_of(node.input[0])), node=source_node)
                if axis != len(self.shape_of(node.input[0])) - 1:
                    raise ExporterError(Diagnostic("VXLAYERNORM_AXIS", f"{source_node} must normalize only the last axis", "fuse-layernorm"))
                if any(
                    name and (consumers.get(name) or name in self.graph_output_names)
                    for name in node.output[1:]
                ):
                    raise ExporterError(Diagnostic("VXLAYERNORM_STATS", f"{source_node} observable statistics outputs are unsupported", "fuse-layernorm"))
                inputs = {"input": self.names.get(self.resolve(node.input[0])), "weight": self.ensure_weight(node.input[1])}
                if len(node.input) > 2 and node.input[2]:
                    inputs["bias"] = self.ensure_weight(node.input[2])
                self._add_node(
                    "LayerNorm", inputs, output, dtype="float32",
                    params={"eps": float(_attribute(node, "epsilon", 1e-5)), "d_model": self.shape_of(node.input[0])[-1]},
                    source_node=source_node, source_op=op,
                )
            elif op == "Dropout":
                training = self._array(node.input[2]) if len(node.input) > 2 and node.input[2] else None
                if training is not None and bool(np.asarray(training).reshape(-1)[0]):
                    raise ExporterError(Diagnostic(
                        "VXDROPOUT_TRAINING", f"{source_node} is a training Dropout", "canonicalize", source_node=source_node
                    ))
                if len(node.output) > 1 and node.output[1] and consumers.get(node.output[1]):
                    raise ExporterError(Diagnostic("VXDROPOUT_MASK", f"{source_node} exposes its dropout mask", "canonicalize"))
                self.alias[output] = self.resolve(node.input[0])
                self.shape_map[output] = self.shape_of(node.input[0])
                self.dtype_map[output] = self.dtype_of(node.input[0])
                self.skipped_nodes += 1
            elif op in {"Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze"}:
                input_name = self.resolve(node.input[0])
                input_ref, dtype = self._execution_ref(
                    input_name,
                    allowed={"float32", "int32", "int8", "uint8"},
                    source_node=source_node,
                    source_op=op,
                    role="input",
                )
                output_shape = self.shape_of(output)
                if op == "Identity" and (
                    not output_shape or any(
                        not isinstance(dimension, (int, str)) or dimension == 0
                        for dimension in output_shape
                    )
                ):
                    output_shape = self.shape_of(input_name)
                params: dict[str, Any] = {}
                if op == "Reshape":
                    params = {"shape": list(output_shape)}
                elif op == "Flatten":
                    params = {"axis": int(_attribute(node, "axis", 1))}
                elif op in {"Squeeze", "Unsqueeze"}:
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
                    if op == "Squeeze" and not axes:
                        axes = [
                            axis for axis, dimension in enumerate(
                                self.shape_of(input_name)
                            )
                            if dimension == 1
                        ]
                    if not axes:
                        raise ExporterError(Diagnostic(
                            "VXSHAPE_AXES",
                            f"{source_node} requires immutable {op} axes",
                            "logical-shapes",
                            source_node=source_node,
                            source_op=op,
                        ))
                    params = {"axes": axes}
                self._add_node(
                    op, {"input": input_ref}, output,
                    shape=output_shape,
                    dtype=dtype,
                    params=params,
                    source_node=source_node, source_op=op,
                )
            elif op == "Transpose":
                input_name = self.resolve(node.input[0])
                input_ref, dtype = self._execution_ref(
                    input_name,
                    allowed={"float32", "int32"},
                    source_node=source_node,
                    source_op=op,
                    role="input",
                )
                permutation = list(_attribute(node, "perm", list(reversed(range(len(self.shape_of(input_name)))))))
                self._add_node(
                    "Transpose", {"input": input_ref}, output, dtype=dtype,
                    params={"perm": permutation}, source_node=source_node, source_op=op,
                )
            elif op == "Concat":
                inputs = {}
                dtypes = set()
                input_shapes: list[list[int | str]] = []
                for position, name in enumerate(node.input):
                    resolved = self.resolve(name)
                    reference, dtype = self._execution_ref(
                        resolved,
                        allowed={"float32", "int32", "int8", "uint8"},
                        source_node=source_node,
                        source_op=op,
                        role=f"input {position}",
                    )
                    inputs[f"input{position}"] = reference
                    dtypes.add(dtype)
                    input_shapes.append(self.shape_of(resolved))
                if len(dtypes) != 1:
                    raise ExporterError(Diagnostic("VXCONCAT_DTYPE", f"{source_node} inputs have different dtypes", "dtype-legalize"))
                axis = int(_attribute(node, "axis"))
                output_shape = self.shape_of(output)
                expected_shape = self._canonical_concat_output_shape(
                    node,
                    input_shapes,
                    axis=axis,
                    dtype=next(iter(dtypes)),
                    source_node=source_node,
                )
                output_shape = expected_shape
                self._add_node(
                    "Concat", inputs, output, shape=output_shape,
                    dtype=next(iter(dtypes)),
                    params={"axis": axis}, source_node=source_node, source_op=op,
                )
            elif op == "Clip":
                input_name = self.resolve(node.input[0])
                input_ref, dtype = self._execution_ref(
                    input_name,
                    allowed={"float32", "int32"},
                    source_node=source_node,
                    source_op=op,
                    role="input",
                )
                minimum = self._array(node.input[1]) if len(node.input) > 1 and node.input[1] else None
                maximum = self._array(node.input[2]) if len(node.input) > 2 and node.input[2] else None
                if minimum is not None and np.asarray(minimum).size != 1:
                    raise ExporterError(Diagnostic("VXCLIP_BOUND", f"{source_node} min must be an immutable scalar", "lower"))
                if maximum is not None and np.asarray(maximum).size != 1:
                    raise ExporterError(Diagnostic("VXCLIP_BOUND", f"{source_node} max must be an immutable scalar", "lower"))
                params = {}
                if minimum is not None:
                    raw_minimum = np.asarray(minimum).reshape(-1)[0]
                    params["min"] = int(raw_minimum) if dtype == "int32" else float(raw_minimum)
                if maximum is not None:
                    raw_maximum = np.asarray(maximum).reshape(-1)[0]
                    params["max"] = int(raw_maximum) if dtype == "int32" else float(raw_maximum)
                if any(not math.isfinite(value) for value in params.values()):
                    raise ExporterError(Diagnostic("VXCLIP_BOUND", f"{source_node} bounds must be finite", "lower"))
                self._add_node(
                    "Clip", {"input": input_ref}, output,
                    dtype=dtype, params=params, source_node=source_node, source_op=op,
                )
            elif op in {"Sigmoid", "Tanh", "Relu"}:
                self._require_execution_dtype(
                    self.resolve(node.input[0]), {"float32"}, source_node=source_node,
                    source_op=op, role="input",
                )
                mapped = "ReLU" if op == "Relu" else op
                self._add_node(
                    mapped, {"input": self.names.get(self.resolve(node.input[0]))}, output,
                    dtype="float32", source_node=source_node, source_op=op,
                )
            elif op == "Cast":
                target = _onnx_dtype_name(int(_attribute(node, "to")))
                if index in self.identity_bool_casts | self.identity_int64_casts:
                    input_name = self.resolve(node.input[0])
                    if self.execution_dtype_of(input_name) != "int32":
                        raise ExporterError(Diagnostic(
                            "VXCAST_I32_REPRESENTATION",
                            f"{source_node} source {target} identity Cast is not "
                            "represented as runtime int32",
                            "dtype-legalize",
                            source_node=source_node,
                            source_op=op,
                            constraint=(
                                "source BOOL/INT64 represented by an exact "
                                "public int32 ABI binding"
                            ),
                        ))
                    self.alias[output] = input_name
                    self.shape_map[output] = self.shape_of(input_name)
                    self.dtype_map[output] = "int32"
                    self.skipped_nodes += 1
                    continue
                if target not in _RUNTIME_DTYPES:
                    raise ExporterError(Diagnostic("VXCAST_DTYPE", f"{source_node} casts to unsupported {target}", "dtype-legalize"))
                input_ref, _ = self._execution_ref(
                    node.input[0],
                    allowed={"float32", "int32", "int8", "uint8"},
                    source_node=source_node,
                    source_op=op,
                    role="input",
                )
                self._add_node(
                    "Cast", {"input": input_ref}, output,
                    dtype=target, params={"to": target}, source_node=source_node, source_op=op,
                )
            elif op in {"If", "Loop", "Scan"}:
                raise ExporterError(Diagnostic(
                    "VXCONTROL_FLOW", f"{source_node} retains unsupported ONNX {op}", "control-flow",
                    source_node=source_node, source_op=op, constraint="branch-free DAG", required_pass="control_flow",
                ))
            else:
                raise ExporterError(Diagnostic(
                    "VXONNX_UNSUPPORTED",
                    f"unsupported ONNX op {op!r} at {source_node}",
                    "lower",
                    source_node=source_node,
                    source_op=op,
                    constraint="enabled semantics-preserving lowering",
                ))

        exported_outputs = []
        produced_exports = {
            descriptor.get("tensor")
            for lowered in self.nodes
            for descriptor in (lowered.get("outputs") or {}).values()
            if isinstance(descriptor, Mapping)
            and isinstance(descriptor.get("tensor"), str)
        }
        for output in self.model.graph.output:
            resolved = self.resolve(output.name)
            wanted = self.names.get(output.name)
            current = self.names.get(resolved)
            if resolved in self.arrays and wanted not in produced_exports:
                constant_ref = self.ensure_weight(
                    resolved,
                    array=np.asarray(self.arrays[resolved]),
                    preferred=f"{wanted}_constant",
                )
                self._add_node(
                    "Identity", {"input": constant_ref}, output.name,
                    shape=self.shape_of(resolved), dtype=self._output_dtype(output.name, self.dtype_of(resolved)),
                    source_node=f"output:{output.name}", source_op="Constant",
                )
            elif current != wanted:
                self._add_node(
                    "Identity", {"input": current}, output.name, shape=self.shape_of(resolved), dtype=self._output_dtype(output.name, self.dtype_of(resolved)),
                    source_node=f"output:{output.name}", source_op="Identity",
                )
            exported_outputs.append(wanted)

        inputs = self._input_definitions()
        graph = {
            "format": "volvox-graph/v1",
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
                for constraint in self.shape_environment.dimensions
            },
            "inputs": inputs,
            "outputs": exported_outputs,
            "nodes": self.nodes,
        }
        quantization_report = externalize_quantization(
            graph, self.weights, self.tensor_quantization
        )
        graph, optimized_weights, typed_report = optimize_runtime_package(
            graph,
            self.weights,
            source_name=f"{Path(self.model_path).name}:lowered",
            allow_silu_numerical_migration=allow_silu_numerical_migration,
            allow_quantized_bias_folding_numerical_migration=(
                allow_quantized_bias_folding_numerical_migration
            ),
            allow_static_qdq_qbatch_matmul_numerical_migration=(
                allow_static_qdq_qbatch_matmul_numerical_migration
            ),
            allow_static_qdq_groupnorm_silu_numerical_migration=(
                allow_static_qdq_groupnorm_silu_numerical_migration
            ),
            enable_static_qdq_layout_optimization=(
                enable_static_qdq_layout_optimization
            ),
            enable_exact_common_subexpression_elimination=(
                enable_exact_common_subexpression_elimination
            ),
            shape_profile=(
                None if self.shape_environment.dimensions else {}
            ),
        )
        self.weights = {
            name: np.asarray(value) for name, value in optimized_weights.items()
        }
        package_class = classify_package(graph, self.weights)
        self.publication_report = {
            "onnx": Path(self.model_path).name,
            "opset": self.opset,
            "frontend": "target-aware-onnx/v1",
            "source_ir": {
                "dialect": self.source_ir.dialect.value,
                "fingerprint": self.source_ir.fingerprint(),
                "nodes": len(self.source_ir.nodes),
                "tensors": len(self.source_ir.tensors),
                "opsets": dict(self.source_ir.opsets),
            },
            "float_storage": self._float_storage(),
            "package_class": package_class,
            "folded_nodes": len(self.folded),
            "skipped_nodes": self.skipped_nodes,
            "features": dict(self.features),
            "abi_changes": list(self.abi_changes),
            "node_sources": list(self.node_sources),
            "quantization_parameters": {
                "tensors": quantization_report.tensors,
                "scales_created": quantization_report.scales_created,
                "zero_points_created": quantization_report.zero_points_created,
                "parameters_reused": quantization_report.parameters_reused,
            },
            "typed_optimizer": serialize_pipeline_report(typed_report),
        }
        return graph, self.weights


def compile_onnx_model(
    model_path: str,
    out_path: str,
    *,
    weight_dtype: str = "auto",
    output_names=None,
    image_normalizations=None,
    input_shapes=None,
    dimension_bounds=None,
    anonymous_dimension_bounds=None,
    input_dtypes=None,
    output_dtypes=None,
    specialize_inputs=None,
    registered_domains=None,
    allow_silu_numerical_migration: bool = False,
    allow_quantized_bias_folding_numerical_migration: bool = False,
    allow_static_qdq_qbatch_matmul_numerical_migration: bool = False,
    allow_static_qdq_groupnorm_silu_numerical_migration: bool = False,
    enable_static_qdq_layout_optimization: bool = True,
    enable_exact_common_subexpression_elimination: bool = False,
    report_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Compile one bounded or constant-only ONNX DAG into closed current v1.

    ``report_callback`` receives a detached JSON-compatible provenance,
    optimizer, and affine-publication report. Executable ``graph.json`` never
    carries those non-runtime fields.
    """

    import json
    from safetensors.numpy import save_file

    logger = log or (lambda message: print(message))
    logger(f"[Export] Loading ONNX graph {model_path}...")
    compiler = OnnxCompiler(
        model_path,
        weight_dtype=weight_dtype,
        output_names=output_names,
        image_normalizations=image_normalizations,
        input_shapes=input_shapes,
        dimension_bounds=dimension_bounds,
        anonymous_dimension_bounds=anonymous_dimension_bounds,
        input_dtypes=input_dtypes,
        output_dtypes=output_dtypes,
        specialize_inputs=specialize_inputs,
        registered_domains=registered_domains,
    )
    graph, weights = compiler.lower(
        allow_silu_numerical_migration=allow_silu_numerical_migration,
        allow_quantized_bias_folding_numerical_migration=(
            allow_quantized_bias_folding_numerical_migration
        ),
        allow_static_qdq_qbatch_matmul_numerical_migration=(
            allow_static_qdq_qbatch_matmul_numerical_migration
        ),
        allow_static_qdq_groupnorm_silu_numerical_migration=(
            allow_static_qdq_groupnorm_silu_numerical_migration
        ),
        enable_static_qdq_layout_optimization=(
            enable_static_qdq_layout_optimization
        ),
        enable_exact_common_subexpression_elimination=(
            enable_exact_common_subexpression_elimination
        ),
    )
    if report_callback is not None:
        report_callback(json.loads(json.dumps(
            compiler.publication_report,
            sort_keys=True,
            allow_nan=False,
        )))
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(destination))
    graph_path = destination.parent / "graph.json"
    graph_path.write_text(
        json.dumps(graph, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger(
        f"[Export] ONNX lowered nodes={len(graph['nodes'])} weights={len(weights)} "
        f"package_class={compiler.publication_report['package_class']} "
        f"features={','.join(sorted(compiler.publication_report['features'])) or 'none'}"
    )
    logger(f"[Export] Wrote {destination} and {graph_path}")
    return graph
