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
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

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














def _set_value_info_shape(value_info, shape: Iterable[int | str]) -> None:
    tensor_shape = value_info.type.tensor_type.shape
    del tensor_shape.dim[:]
    for extent in shape:
        dimension = tensor_shape.dim.add()
        if isinstance(extent, str):
            dimension.dim_param = extent
        else:
            dimension.dim_value = int(extent)
















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




















from .frontend_onnx_emit import OnnxEmissionMixin


from .frontend_onnx_recognize import OnnxRecognitionMixin


from .frontend_onnx_fold import OnnxConstantFoldingMixin
from .frontend_onnx_common import (
    _Attention,
    _DQ,
    _GeluRegion,
    _GroupNormRegion,
    _LinearBiasRegion,
    _QArgMaxRegion,
    _QBatchMatMulRegion,
    _QConvRegion,
    _QLinearRegion,
    _RUNTIME_DTYPES,
    _attribute,
    _broadcast_shape,
    _concat_shape,
    _dequantize,
    _normalize_axis,
    _onnx_dtype_name,
    _product,
    _quantization,
    _runtime_dtype_for_array,
    _shape,
    _source_name,
    _to_i32_checked,
)


class OnnxCompiler(OnnxConstantFoldingMixin, OnnxRecognitionMixin, OnnxEmissionMixin):
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
        self.linear_bias_fusions: dict[int, _LinearBiasRegion] = {}
        self.linear_bias_skip: set[int] = set()
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
        self._normalize_einsum()
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
        self._fold_invariant_symbolic_shape_programs()
        self._recognize_identity_abi_casts()
        self._recognize_w8a8()
        self._recognize_exact_gelu()
        self._recognize_group_norm()
        self._recognize_linear_bias()
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

    def _logical_shapes_equivalent(
        self,
        left: Sequence[int | str],
        right: Sequence[int | str],
    ) -> bool:
        """Prove exact shape equality, including singleton symbol domains.

        Distinct non-singleton symbols are not interchangeable even when their
        numeric bounds happen to match: their request-time values need not be
        correlated.  A singleton domain, however, is exactly the corresponding
        concrete extent over every legal request and is safe to use when ONNX
        shape inference resolves stale public metadata to that extent.
        """

        if len(left) != len(right):
            return False

        def singleton_extent(dimension: int | str) -> int | None:
            if isinstance(dimension, int):
                return dimension
            constraint = self.shape_environment.get(dimension)
            if constraint is None or constraint.min != constraint.max:
                return None
            return constraint.min

        return all(
            left_dimension == right_dimension
            or (
                singleton_extent(left_dimension) is not None
                and singleton_extent(left_dimension)
                == singleton_extent(right_dimension)
            )
            for left_dimension, right_dimension in zip(left, right)
        )

    def _broadcast_execution_ref(
        self,
        name: str,
        *,
        output_shape: list[int | str],
        dtype: str,
        stem: str,
        role: str,
        source_node: str,
        source_op: str,
    ) -> str:
        """Make an ONNX broadcast operand exact-shape for canonical runtimes."""

        resolved = self.resolve(name)
        input_shape = self.shape_of(resolved)
        if self._logical_shapes_equivalent(input_shape, output_shape):
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
