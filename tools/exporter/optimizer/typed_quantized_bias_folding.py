"""Fold post-dequantization F32 dense bias into an I32 accumulator bias.

Already-quantized producer graphs often spell a dense projection as::

    QLinear -> DequantizeLinear -> Add(immutable F32 bias)

The extra Add breaks the byte island.  This pass can move that bias into the
QLinear accumulator domain using the *existing* input and per-channel weight
scales.  It never reconstructs or requantizes the byte weight and never
calibrates an activation.

Moving the Add across QLinear's output requantization changes a visible
rounding boundary, so this is an explicit numerical migration.  The pass also
proves finite F32 scale arithmetic and a conservative I32 accumulator bound;
otherwise it leaves the graph unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import numpy as np

from ..ir import IRDialect, OpNode, TensorValue, ValuePort
from ..pipeline import IRPass, PassContract, PassResult


_BYTE_DTYPES = frozenset({"int8", "uint8"})
_DENSE_OPS = frozenset({"QLinear", "QMatMul", "QGemm"})
_INT32 = np.iinfo(np.int32)


@dataclass(frozen=True)
class _FoldPlan:
    dense_index: int
    dequantize_index: int
    add_index: int
    old_bias: str
    float_bias: str
    intermediate: str
    result: str
    folded_bias: np.ndarray
    maximum_bias_error: float


def _params(node: OpNode) -> Mapping[str, Any] | None:
    if not node.attributes:
        return {}
    if (
        len(node.attributes) != 1
        or node.attributes[0].name != "params"
        or node.attributes[0].kind != "volvox.params"
        or not isinstance(node.attributes[0].value, Mapping)
    ):
        return None
    return node.attributes[0].value


class RuntimeQuantizedBiasFoldingPass(IRPass):
    """Absorb a proven F32 post-DQ bias into QLinear's I32 bias."""

    name = "runtime-quantized-bias-folding"
    contract = PassContract.preserving(IRDialect.RUNTIME, repeatable=True)

    def __init__(
        self,
        tensor_data: MutableMapping[str, Any],
        *,
        allow_numerical_migration: bool,
    ) -> None:
        if not isinstance(tensor_data, MutableMapping):
            raise TypeError("quantized bias folding requires mutable tensor data")
        if allow_numerical_migration is not True:
            raise ValueError(
                "quantized bias folding requires explicit numerical-migration opt-in"
            )
        self.tensor_data = tensor_data
        self.refused = 0

    def run(self, graph) -> PassResult:
        graph_snapshot = graph.clone()
        data_snapshot = dict(self.tensor_data)
        folded: list[str] = []
        notes: list[str] = []
        self.refused = 0
        try:
            while True:
                plan, considered = self._find_plan(graph)
                self.refused += considered
                if plan is None:
                    break
                dense_name = graph.nodes[plan.dense_index].name
                self._apply(graph, plan)
                folded.append(dense_name)
                notes.append(
                    f"folded post-DQ F32 bias into {dense_name!r} I32 "
                    f"accumulator bias (max bias packing error "
                    f"{plan.maximum_bias_error:.9g})"
                )
            if folded:
                graph.verify(IRDialect.RUNTIME)
        except Exception:
            graph.restore(graph_snapshot)
            self.tensor_data.clear()
            self.tensor_data.update(data_snapshot)
            raise
        if self.refused:
            notes.append(
                f"refused {self.refused} unproved quantized bias candidate(s)"
            )
        return PassResult(
            len(folded),
            touched_nodes=tuple(folded),
            notes=tuple(notes),
        )

    def _find_plan(self, graph) -> tuple[_FoldPlan | None, int]:
        graph.invalidate_analyses()
        use_def = graph.use_def()
        refused = 0
        for dense_index, dense in enumerate(graph.nodes):
            if dense.op_type not in _DENSE_OPS:
                continue
            skeleton = self._skeleton(graph, use_def, dense_index)
            if skeleton is None:
                continue
            plan = self._finish_plan(graph, skeleton)
            if plan is not None:
                return plan, refused
            refused += 1
        return None, refused

    @staticmethod
    def _skeleton(graph, use_def, dense_index: int) -> dict[str, Any] | None:
        dense = graph.nodes[dense_index]
        inputs = dense.input_map()
        outputs = dense.output_map()
        activation_ports = set(inputs) & {"input", "x", "a"}
        if (
            len(activation_ports) != 1
            or set(inputs) != {*activation_ports, "weight", "bias"}
            or set(outputs) != {"out"}
            or _params(dense) not in ({}, {"weight_layout": "OUT_IN"})
        ):
            return None
        byte_output = outputs["out"]
        byte_uses = use_def.consumers.get(byte_output, ())
        if byte_output in graph.outputs or len(byte_uses) != 1:
            return None
        dequantize_index = byte_uses[0].node_index
        dequantize = graph.nodes[dequantize_index]
        dq_inputs = dequantize.input_map()
        dq_outputs = dequantize.output_map()
        if (
            dequantize.op_type != "DequantizeLinear"
            or set(dq_inputs) != {"input", "scale", "zero_point"}
            or set(dq_outputs) != {"out"}
            or dq_inputs["input"] != byte_output
            or _params(dequantize) != {}
        ):
            return None
        intermediate = dq_outputs["out"]
        intermediate_uses = use_def.consumers.get(intermediate, ())
        if intermediate in graph.outputs or len(intermediate_uses) != 1:
            return None
        add_index = intermediate_uses[0].node_index
        add = graph.nodes[add_index]
        add_inputs = add.input_map()
        add_outputs = add.output_map()
        if (
            add.op_type != "Add"
            or set(add_inputs) != {"a", "b"}
            or set(add_outputs) != {"out"}
            or _params(add) not in ({}, {"relu": 0})
        ):
            return None
        dynamic_ports = [
            port for port, value in add_inputs.items() if value == intermediate
        ]
        if len(dynamic_ports) != 1:
            return None
        float_bias = add_inputs["b" if dynamic_ports[0] == "a" else "a"]
        return {
            "dense_index": dense_index,
            "dequantize_index": dequantize_index,
            "add_index": add_index,
            "activation": inputs[next(iter(activation_ports))],
            "weight": inputs["weight"],
            "old_bias": inputs["bias"],
            "byte_output": byte_output,
            "intermediate": intermediate,
            "float_bias": float_bias,
            "result": add_outputs["out"],
        }

    def _finish_plan(self, graph, match: Mapping[str, Any]) -> _FoldPlan | None:
        activation = graph.tensors.get(match["activation"])
        weight = graph.tensors.get(match["weight"])
        byte_output = graph.tensors.get(match["byte_output"])
        intermediate = graph.tensors.get(match["intermediate"])
        result = graph.tensors.get(match["result"])
        if any(value is None for value in (
            activation, weight, byte_output, intermediate, result,
        )):
            return None
        assert activation is not None and weight is not None
        assert byte_output is not None and intermediate is not None and result is not None
        if (
            activation.dtype not in _BYTE_DTYPES
            or weight.dtype not in _BYTE_DTYPES
            or byte_output.dtype not in _BYTE_DTYPES
            or intermediate.dtype != "float32"
            or result.dtype != "float32"
            or not all(value.concrete for value in (
                activation, weight, byte_output, intermediate, result,
            ))
            or activation.rank < 1
            or weight.rank != 2
            or byte_output.rank != activation.rank
            or activation.shape[:-1] != byte_output.shape[:-1]
            or weight.shape != (byte_output.shape[-1], activation.shape[-1])
            or intermediate.shape != byte_output.shape
            or result.shape != intermediate.shape
            or activation.quantization is None
            or activation.quantization.scheme != "per_tensor"
            or weight.quantization is None
            or weight.quantization.scheme != "per_axis"
            or weight.quantization.axis != 0
            or byte_output.quantization is None
            or byte_output.quantization.scheme != "per_tensor"
        ):
            return None

        dequantize = graph.nodes[match["dequantize_index"]]
        dq_inputs = dequantize.input_map()
        if (
            dq_inputs["scale"] != byte_output.quantization.scale
            or dq_inputs["zero_point"] != byte_output.quantization.zero_point
        ):
            return None

        d_out = int(weight.shape[0])
        arrays = {
            "input_scale": self._initializer(
                graph, activation.quantization.scale, np.float32, (1,),
            ),
            "input_zero": self._initializer(
                graph,
                activation.quantization.zero_point,
                np.int8 if activation.dtype == "int8" else np.uint8,
                (1,),
            ),
            "weight": self._initializer(
                graph,
                match["weight"],
                np.int8 if weight.dtype == "int8" else np.uint8,
                tuple(int(value) for value in weight.shape),
            ),
            "weight_scale": self._initializer(
                graph, weight.quantization.scale, np.float32, (d_out,),
            ),
            "weight_zero": self._initializer(
                graph,
                weight.quantization.zero_point,
                np.int8 if weight.dtype == "int8" else np.uint8,
                (d_out,),
            ),
            "old_bias": self._initializer(
                graph, match["old_bias"], np.int32, (d_out,),
            ),
        }
        if any(value is None for value in arrays.values()):
            return None
        float_bias = self._initializer_any_shape(
            graph, match["float_bias"], np.float32,
        )
        if (
            float_bias is None
            or float_bias.ndim < 1
            or float_bias.shape[-1] != d_out
            or any(int(value) != 1 for value in float_bias.shape[:-1])
        ):
            return None
        try:
            if np.broadcast_shapes(result.shape, float_bias.shape) != result.shape:
                return None
        except ValueError:
            return None

        input_scale = arrays["input_scale"]
        weight_scales = arrays["weight_scale"]
        old_bias = arrays["old_bias"]
        assert input_scale is not None and weight_scales is not None
        assert old_bias is not None
        bias = np.ascontiguousarray(float_bias.reshape(d_out))
        if not bool(np.all(np.isfinite(bias))):
            return None
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            accumulator_scales = np.multiply(
                np.float32(input_scale[0]), weight_scales, dtype=np.float32,
            )
            transformed = np.divide(bias, accumulator_scales, dtype=np.float32)
        if (
            not bool(np.all(np.isfinite(accumulator_scales)))
            or bool(np.any(accumulator_scales <= np.float32(0.0)))
            or not bool(np.all(np.isfinite(transformed)))
        ):
            return None
        rounded = np.rint(transformed)
        if bool(np.any(rounded < _INT32.min)) or bool(np.any(rounded > _INT32.max)):
            return None
        combined = old_bias.astype(np.int64) + rounded.astype(np.int64)
        if bool(np.any(combined < _INT32.min)) or bool(np.any(combined > _INT32.max)):
            return None

        input_zero = arrays["input_zero"]
        weight_values = arrays["weight"]
        weight_zero = arrays["weight_zero"]
        assert input_zero is not None and weight_values is not None
        assert weight_zero is not None
        storage = np.iinfo(
            np.int8 if activation.dtype == "int8" else np.uint8
        )
        input_magnitude = max(
            abs(int(storage.min) - int(input_zero[0])),
            abs(int(storage.max) - int(input_zero[0])),
        )
        centered_weight = (
            weight_values.astype(np.int64) - weight_zero.astype(np.int64)[:, None]
        )
        for channel in range(d_out):
            dot_bound = input_magnitude * sum(
                abs(int(value)) for value in centered_weight[channel]
            )
            low = int(combined[channel]) - dot_bound
            high = int(combined[channel]) + dot_bound
            if low < _INT32.min or high > _INT32.max:
                return None

        folded = np.ascontiguousarray(combined.astype(np.int32))
        reconstructed = np.multiply(
            rounded.astype(np.float32), accumulator_scales, dtype=np.float32,
        )
        error = np.abs(np.subtract(reconstructed, bias, dtype=np.float32))
        return _FoldPlan(
            dense_index=int(match["dense_index"]),
            dequantize_index=int(match["dequantize_index"]),
            add_index=int(match["add_index"]),
            old_bias=str(match["old_bias"]),
            float_bias=str(match["float_bias"]),
            intermediate=str(match["intermediate"]),
            result=str(match["result"]),
            folded_bias=folded,
            maximum_bias_error=float(np.max(error, initial=np.float32(0.0))),
        )

    def _initializer(
        self,
        graph,
        name: str,
        dtype: Any,
        shape: tuple[int, ...],
    ) -> np.ndarray | None:
        value = self._initializer_any_shape(graph, name, dtype)
        if value is None or value.shape != shape:
            return None
        return value

    def _initializer_any_shape(
        self,
        graph,
        name: str,
        dtype: Any,
    ) -> np.ndarray | None:
        tensor = graph.tensors.get(name)
        value = self.tensor_data.get(name)
        if tensor is None or not tensor.initializer or value is None:
            return None
        try:
            array = np.asarray(value)
        except Exception:
            return None
        if (
            array.dtype != np.dtype(dtype)
            or tuple(array.shape) != tensor.shape
            or str(array.dtype) != tensor.dtype
        ):
            return None
        return array

    def _apply(self, graph, plan: _FoldPlan) -> None:
        dense = graph.nodes[plan.dense_index]
        dequantize = graph.nodes[plan.dequantize_index]
        add = graph.nodes[plan.add_index]
        folded_name = self._fresh_name(graph, f"{plan.old_bias}.folded")
        self.tensor_data[folded_name] = plan.folded_bias
        graph.add_tensor(TensorValue(
            name=folded_name,
            shape=tuple(int(value) for value in plan.folded_bias.shape),
            dtype="int32",
            source_dtype="int32",
            initializer=True,
            metadata={
                "optimizer_precision_authoring": "post-dq-bias-to-i32",
                "source_bias": plan.float_bias,
            },
        ))
        dense.inputs = tuple(
            ValuePort(
                port.name,
                folded_name if port.name == "bias" else port.value,
                port.position,
            )
            for port in dense.inputs
        )
        dense.provenance = (*dense.provenance, *add.provenance)
        dense.metadata = {
            **dense.metadata,
            "optimizer_fusion": {
                "kind": "post-dq-bias-to-i32",
                "semantic_contract": "numerical-migration-opt-in",
                "source_add": add.name,
            },
        }
        dequantize.outputs = tuple(
            ValuePort(
                port.name,
                plan.result if port.name == "out" else port.value,
                port.position,
            )
            for port in dequantize.outputs
        )
        dequantize.provenance = (*dequantize.provenance, *add.provenance)
        del graph.nodes[plan.add_index]
        graph.tensors.pop(plan.intermediate, None)

        for feature, names in tuple(graph.features.items()):
            retained = [name for name in names if name != add.name]
            if retained:
                graph.features[feature] = retained
            else:
                del graph.features[feature]

        live_inputs = {
            name for node in graph.nodes for name in node.input_map().values()
        }
        protected = set(graph.inputs) | set(graph.outputs)
        for name in (plan.old_bias, plan.float_bias):
            if name in live_inputs or name in protected:
                continue
            tensor = graph.tensors.get(name)
            if tensor is not None and tensor.initializer:
                graph.tensors.pop(name)
                self.tensor_data.pop(name, None)
        graph.invalidate_analyses()

    def _fresh_name(self, graph, stem: str) -> str:
        occupied = set(graph.tensors) | set(self.tensor_data)
        candidate = stem
        suffix = 1
        while candidate in occupied:
            suffix += 1
            candidate = f"{stem}.{suffix}"
        return candidate


__all__ = ["RuntimeQuantizedBiasFoldingPass"]
