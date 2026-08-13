from __future__ import annotations

import json
from dataclasses import replace
import unittest

import numpy as np

from tools.exporter.ir import (
    AffineQuantization,
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    TensorValue,
)
from tools.exporter.optimizer.compiled_model import (
    CompiledModelPlan,
    CompiledModelPlanningError,
    KernelPredicateEvidence,
    build_compiled_model_plan,
)
from tools.exporter.optimizer.quantized_regions import (
    QuantizedRegionCandidateAnalysis,
)
from tools.exporter.optimizer.target import TargetEnvironment
from tools.exporter.shape_system import ShapeEnvironment


def _tensor(
    graph: GraphIR,
    name: str,
    shape: tuple[int | str, ...],
    dtype: str = "float32",
    *,
    initializer: bool = False,
    public_input: bool = False,
    public_output: bool = False,
    quantization: AffineQuantization | None = None,
) -> None:
    graph.add_tensor(TensorValue(
        name=name,
        shape=shape,
        dtype=dtype,
        source_dtype=dtype,
        initializer=initializer,
        public_input=public_input,
        public_output=public_output,
        quantization=quantization,
    ))


def _pointwise_graph(operator: str = "ReLU") -> GraphIR:
    graph = GraphIR("volvoxai", "pointwise.json", dialect=IRDialect.RUNTIME)
    _tensor(graph, "x", (2, 3), public_input=True)
    _tensor(graph, "y", (2, 3), public_output=True)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "pointwise", operator, {"input": "x"}, {"out": "y"},
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


def _average_pool_graph() -> GraphIR:
    graph = GraphIR("volvoxai", "average-pool.json", dialect=IRDialect.RUNTIME)
    _tensor(graph, "x", (1, 4, 4, 1), public_input=True)
    _tensor(graph, "y", (1, 2, 2, 1), public_output=True)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "pool",
        "AveragePool2D",
        {"input": "x"},
        {"out": "y"},
        attributes=(OpAttribute("params", "volvox.params", {
            "kernel": [2, 2],
            "stride": [2, 2],
            "pads": [0, 0, 0, 0],
            "padding": [0, 0],
            "data_layout": "NHWC",
        }),),
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


def _long_pointwise_graph(node_count: int) -> GraphIR:
    graph = GraphIR("volvoxai", "long-pointwise.json", dialect=IRDialect.RUNTIME)
    _tensor(graph, "x", (1,), public_input=True)
    graph.inputs.append("x")
    previous = "x"
    for index in range(node_count):
        output = f"v{index}"
        _tensor(graph, output, (1,), public_output=index == node_count - 1)
        graph.add_node(OpNode.from_maps(
            f"relu{index}", "ReLU", {"input": previous}, {"out": output},
        ))
        previous = output
    graph.outputs.append(previous)
    graph.verify(IRDialect.RUNTIME)
    return graph


def _qlinear_graph() -> GraphIR:
    graph = GraphIR("volvoxai", "qlinear.json", dialect=IRDialect.RUNTIME)
    _tensor(graph, "x.scale", (1,), initializer=True)
    _tensor(graph, "x.zero", (1,), "int8", initializer=True)
    _tensor(
        graph,
        "x",
        (1, 3),
        "int8",
        public_input=True,
        quantization=AffineQuantization(
            "per_tensor", "x.scale", "x.zero",
        ),
    )
    _tensor(graph, "w.scale", (2,), initializer=True)
    _tensor(graph, "w.zero", (2,), "int8", initializer=True)
    _tensor(
        graph,
        "w",
        (2, 3),
        "int8",
        initializer=True,
        quantization=AffineQuantization(
            "per_axis", "w.scale", "w.zero", axis=0,
        ),
    )
    _tensor(graph, "bias", (2,), "int32", initializer=True)
    _tensor(graph, "y.scale", (1,), initializer=True)
    _tensor(graph, "y.zero", (1,), "int8", initializer=True)
    _tensor(
        graph,
        "y",
        (1, 2),
        "int8",
        public_output=True,
        quantization=AffineQuantization(
            "per_tensor", "y.scale", "y.zero",
        ),
    )
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "dense", "QLinear",
        {"input": "x", "weight": "w", "bias": "bias"},
        {"out": "y"},
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


def _dynamic_qlinear_graph(maximum_batch: int) -> GraphIR:
    graph = GraphIR("volvoxai", "qlinear-dynamic.json", dialect=IRDialect.RUNTIME)
    graph.shape_environment = ShapeEnvironment(({
        "name": "batch",
        "min": 1,
        "max": maximum_batch,
        "multiple_of": 1,
    },))
    _tensor(graph, "x.scale", (1,), initializer=True)
    _tensor(graph, "x.zero", (1,), "int8", initializer=True)
    _tensor(
        graph,
        "x",
        ("batch", 3),
        "int8",
        public_input=True,
        quantization=AffineQuantization(
            "per_tensor", "x.scale", "x.zero",
        ),
    )
    _tensor(graph, "w.scale", (2,), initializer=True)
    _tensor(graph, "w.zero", (2,), "int8", initializer=True)
    _tensor(
        graph,
        "w",
        (2, 3),
        "int8",
        initializer=True,
        quantization=AffineQuantization(
            "per_axis", "w.scale", "w.zero", axis=0,
        ),
    )
    _tensor(graph, "bias", (2,), "int32", initializer=True)
    _tensor(graph, "y.scale", (1,), initializer=True)
    _tensor(graph, "y.zero", (1,), "int8", initializer=True)
    _tensor(
        graph,
        "y",
        ("batch", 2),
        "int8",
        public_output=True,
        quantization=AffineQuantization(
            "per_tensor", "y.scale", "y.zero",
        ),
    )
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "dense", "QLinear",
        {"input": "x", "weight": "w", "bias": "bias"},
        {"out": "y"},
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


def _matmul_graph() -> GraphIR:
    graph = GraphIR("volvoxai", "matmul.json", dialect=IRDialect.RUNTIME)
    _tensor(graph, "x", (1, 3), public_input=True)
    _tensor(graph, "w", (2, 3), initializer=True)
    _tensor(graph, "y", (1, 2), public_output=True)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "dense", "Linear", {"input": "x", "weight": "w"}, {"out": "y"},
        attributes=(OpAttribute(
            "params", "volvox.params", {"weight_layout": "dout_din"},
        ),),
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


def _closed_qdq_graph() -> GraphIR:
    graph = GraphIR("volvoxai", "closed-qdq.json", dialect=IRDialect.RUNTIME)
    _tensor(graph, "x.scale", (1,), initializer=True)
    _tensor(graph, "x.zero", (1,), "int8", initializer=True)
    _tensor(
        graph,
        "x",
        (2, 3),
        "int8",
        public_input=True,
        quantization=AffineQuantization(
            "per_tensor", "x.scale", "x.zero",
        ),
    )
    _tensor(graph, "decoded", (2, 3))
    _tensor(graph, "activated", (2, 3))
    _tensor(graph, "y.scale", (1,), initializer=True)
    _tensor(graph, "y.zero", (1,), "int8", initializer=True)
    _tensor(
        graph,
        "y",
        (2, 3),
        "int8",
        public_output=True,
        quantization=AffineQuantization(
            "per_tensor", "y.scale", "y.zero",
        ),
    )
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "decode",
        "DequantizeLinear",
        {"input": "x", "scale": "x.scale", "zero_point": "x.zero"},
        {"out": "decoded"},
    ))
    graph.add_node(OpNode.from_maps(
        "relu", "ReLU", {"input": "decoded"}, {"out": "activated"},
    ))
    graph.add_node(OpNode.from_maps(
        "encode",
        "QuantizeLinear",
        {"input": "activated", "scale": "y.scale", "zero_point": "y.zero"},
        {"out": "y"},
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


def _target(
    compile_backend: str,
    *,
    tune_backend: str = "wasm",
    backend_profile: str = "portable",
    features: frozenset[str] = frozenset(),
) -> TargetEnvironment:
    return TargetEnvironment(
        backend_profile=backend_profile,
        compile_backend=compile_backend,
        tune_backend=tune_backend,
        compile_features=features,
    )


def _weights(graph: GraphIR) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for name, tensor in graph.tensors.items():
        if not tensor.initializer:
            continue
        dtype = {
            "float32": np.float32,
            "int32": np.int32,
            "int8": np.int8,
            "uint8": np.uint8,
        }[tensor.dtype]
        fill = 1 if tensor.dtype == "float32" else 0
        values[name] = np.full(tensor.shape, fill, dtype=dtype)
    return values


class CompiledModelPlanTests(unittest.TestCase):
    def test_wasm_and_native_plans_share_one_unchanged_portable_graph(self):
        graph = _pointwise_graph()
        fingerprint = graph.fingerprint()

        wasm = build_compiled_model_plan(
            graph, _weights(graph), _target("wasm", tune_backend="native-cpu"),
        )
        native = build_compiled_model_plan(
            graph, _weights(graph), _target("native-cpu", tune_backend="wasm"),
        )

        self.assertEqual(graph.fingerprint(), fingerprint)
        self.assertEqual(wasm.source_graph_fingerprint, fingerprint)
        self.assertEqual(native.source_graph_fingerprint, fingerprint)
        self.assertEqual(wasm.backend_profile_members, native.backend_profile_members)
        self.assertEqual(wasm.nodes[0].operator, native.nodes[0].operator)
        self.assertEqual(wasm.nodes[0].route_id, "relu")
        self.assertEqual(native.nodes[0].route_id, "native-run-node")
        self.assertNotEqual(wasm.plan_id, native.plan_id)
        json.dumps(wasm.to_dict(), allow_nan=False)

    def test_tune_backend_does_not_change_compiled_or_portable_plan(self):
        graph = _pointwise_graph()
        first = build_compiled_model_plan(
            graph, _weights(graph), _target("wasm", tune_backend="wasm"),
        )
        second = build_compiled_model_plan(
            graph, _weights(graph), _target("wasm", tune_backend="native-cpu"),
        )
        self.assertEqual(first.plan_id, second.plan_id)
        self.assertEqual(first.to_dict(), second.to_dict())

        with self.assertRaisesRegex(
            CompiledModelPlanningError, "do not admit operator fallback",
        ):
            build_compiled_model_plan(
                graph,
                _weights(graph),
                TargetEnvironment(
                    backend_profile="portable",
                    compile_backend="wasm",
                    tune_backend="wasm",
                    allow_operator_fallback=True,
                ),
            )

    def test_profile_legality_is_not_narrowed_by_compile_backend(self):
        graph = _average_pool_graph()
        with self.assertRaisesRegex(
            CompiledModelPlanningError,
            "backend profile descriptor validation",
        ):
            build_compiled_model_plan(graph, _weights(graph), _target("wasm"))

        wasm_only = build_compiled_model_plan(
            graph,
            _weights(graph),
            _target("wasm", backend_profile="wasm"),
        )
        self.assertEqual(wasm_only.backend_profile_members, ("wasm",))

    def test_compiled_plan_runs_full_profile_descriptor_validation(self):
        graph = _long_pointwise_graph(1025)
        with self.assertRaisesRegex(
            CompiledModelPlanningError, "VXCAP_NATIVE_NODES",
        ):
            build_compiled_model_plan(graph, {}, _target("wasm"))

    def test_variant_selection_requires_features_and_predicate_evidence(self):
        graph = _qlinear_graph()
        target = _target(
            "native-cpu",
            features=frozenset({"x86.avx2"}),
        )
        tensors = _weights(graph)
        inspected = build_compiled_model_plan(graph, tensors, target)
        self.assertIn(
            "native-cpu.qlinear.avx2",
            tuple(value.id for value in inspected.nodes[0].variants),
        )
        with self.assertRaisesRegex(
            CompiledModelPlanningError, "requires executable predicate",
        ):
            build_compiled_model_plan(
                graph,
                tensors,
                target,
                selected_variants={"dense": "native-cpu.qlinear.avx2"},
            )

        selected = build_compiled_model_plan(
            graph,
            tensors,
            target,
            selected_variants={"dense": "native-cpu.qlinear.avx2"},
            predicate_evaluator=lambda context: KernelPredicateEvidence.accept(
                context, "native x86 dimensions and layout were verified",
            ),
        )
        self.assertEqual(
            selected.nodes[0].selected_variant_id,
            "native-cpu.qlinear.avx2",
        )
        self.assertEqual(
            selected.nodes[0].selection_evidence.predicate_id,
            "native.qlinear.x86-eligible",
        )
        evidence = selected.nodes[0].selection_evidence
        self.assertEqual(
            evidence.kernel_variant_id,
            "native-cpu.qlinear.avx2",
        )
        self.assertEqual(
            evidence.shape_domain_proof_identity,
            selected.shape_domain_proof.proof_identity,
        )
        self.assertEqual(
            evidence.shape_function_id,
            selected.shape_domain_proof.nodes[0].shape_function_id,
        )
        self.assertEqual(
            evidence.domain_facts,
            selected.shape_domain_proof.nodes[0].facts,
        )
        with self.assertRaisesRegex(
            CompiledModelPlanningError, "not feature/backend/operator eligible",
        ):
            build_compiled_model_plan(
                graph,
                tensors,
                _target("native-cpu"),
                selected_variants={"dense": "native-cpu.qlinear.avx2"},
                predicate_evaluator=lambda context: KernelPredicateEvidence.accept(
                    context, "unreachable",
                ),
            )

    def test_qlinear_variant_cannot_be_claimed_for_float_matmul(self):
        graph = _matmul_graph()
        inspected = build_compiled_model_plan(
            graph, _weights(graph), _target("cpu-js"),
        )
        self.assertEqual(inspected.nodes[0].variants, ())
        with self.assertRaisesRegex(
            CompiledModelPlanningError, "not feature/backend/operator eligible",
        ):
            build_compiled_model_plan(
                graph,
                _weights(graph),
                _target("cpu-js"),
                selected_variants={"dense": "cpu-js.qlinear.reference"},
                predicate_evaluator=lambda context: KernelPredicateEvidence.accept(
                    context, "fabricated",
                ),
            )

    def test_compiled_plan_round_trip_is_strict_and_content_addressed(self):
        graph = _qlinear_graph()
        plan = build_compiled_model_plan(
            graph,
            _weights(graph),
            _target("native-cpu", features=frozenset({"x86.avx2"})),
            selected_variants={"dense": "native-cpu.qlinear.avx2"},
            predicate_evaluator=lambda context: KernelPredicateEvidence.accept(
                context, "verified",
            ),
        )
        encoded = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
        self.assertNotIn("shape_system", encoded["shape_domain_proof"])
        self.assertNotIn(
            "shape_system",
            encoded["nodes"][0]["selection_evidence"],
        )
        self.assertEqual(CompiledModelPlan.from_dict(encoded), plan)

        encoded = plan.to_dict()
        encoded["shape_domain_proof"]["shape_system"] = (
            "volvox-bounded-shape/v1"
        )
        with self.assertRaisesRegex(ValueError, "invalid fields"):
            CompiledModelPlan.from_dict(encoded)

        encoded = plan.to_dict()
        encoded["nodes"][0]["selection_evidence"]["shape_system"] = (
            "volvox-bounded-shape/v1"
        )
        with self.assertRaisesRegex(ValueError, "extra"):
            CompiledModelPlan.from_dict(encoded)

        encoded = plan.to_dict()
        encoded["plan_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "plan ID"):
            CompiledModelPlan.from_dict(encoded)

        encoded = plan.to_dict()
        encoded["compile_device_fingerprint"] = "different-device"
        with self.assertRaisesRegex(ValueError, "not bound"):
            CompiledModelPlan.from_dict(encoded)

        encoded = plan.to_dict()
        encoded["shape_domain_proof"]["dimensions"] = [{
            "name": "fabricated",
            "min": 1,
            "max": 2,
            "multiple_of": 1,
        }]
        with self.assertRaisesRegex(ValueError, "identity is stale"):
            CompiledModelPlan.from_dict(encoded)

        encoded = plan.to_dict()
        encoded["nodes"][0]["selection_evidence"]["kernel_variant_id"] = (
            "native-cpu.qlinear.fabricated"
        )
        with self.assertRaisesRegex(ValueError, "different variant"):
            CompiledModelPlan.from_dict(encoded)

        encoded = plan.to_dict()
        encoded["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "extra"):
            CompiledModelPlan.from_dict(encoded)

    def test_predicate_evidence_is_bound_to_the_whole_dynamic_domain(self):
        target = _target(
            "native-cpu",
            backend_profile="native-cpu",
            features=frozenset({"x86.avx2"}),
        )
        first_graph = _dynamic_qlinear_graph(8)
        first = build_compiled_model_plan(
            first_graph,
            _weights(first_graph),
            target,
            selected_variants={"dense": "native-cpu.qlinear.avx2"},
            predicate_evaluator=lambda context: KernelPredicateEvidence.accept(
                context, "physical predicate passed for the proven domain",
            ),
        )
        first_evidence = first.nodes[0].selection_evidence
        self.assertEqual(
            first_evidence.shape_constraints,
            ("batch:min=1,max=8,multiple_of=1",),
        )

        second_graph = _dynamic_qlinear_graph(16)
        second = build_compiled_model_plan(
            second_graph,
            _weights(second_graph),
            target,
            selected_variants={"dense": "native-cpu.qlinear.avx2"},
            predicate_evaluator=lambda context: KernelPredicateEvidence.accept(
                context, "physical predicate passed for the proven domain",
            ),
        )
        self.assertNotEqual(
            first.shape_domain_proof.proof_identity,
            second.shape_domain_proof.proof_identity,
        )
        self.assertNotEqual(first.plan_id, second.plan_id)
        self.assertEqual(
            second.nodes[0].selection_evidence.shape_constraints,
            ("batch:min=1,max=16,multiple_of=1",),
        )

        def stale_evaluator(context):
            return replace(
                first_evidence,
                source_graph_fingerprint=context.source_graph_fingerprint,
            )

        with self.assertRaisesRegex(
            CompiledModelPlanningError, "different graph, node, backend, device",
        ):
            build_compiled_model_plan(
                second_graph,
                _weights(second_graph),
                target,
                selected_variants={"dense": "native-cpu.qlinear.avx2"},
                predicate_evaluator=stale_evaluator,
            )

    def test_predicate_evaluation_isolated_from_source_graph_and_weights(self):
        graph = _qlinear_graph()
        weights = _weights(graph)
        graph_before = graph.fingerprint()
        weight_before = weights["w"].copy()

        def mutating_predicate(context):
            context.graph.metadata["mutated"] = True
            context.tensors["w"][0, 0] = 99
            return KernelPredicateEvidence.accept(context, "invalid mutation")

        with self.assertRaisesRegex(
            CompiledModelPlanningError, "mutated its inputs",
        ):
            build_compiled_model_plan(
                graph,
                weights,
                _target("native-cpu", features=frozenset({"x86.avx2"})),
                selected_variants={"dense": "native-cpu.qlinear.avx2"},
                predicate_evaluator=mutating_predicate,
            )
        self.assertEqual(graph.fingerprint(), graph_before)
        np.testing.assert_array_equal(weights["w"], weight_before)

    def test_backend_regions_remain_unselected_opportunities(self):
        graph = _closed_qdq_graph()
        fingerprint = graph.fingerprint()
        candidates = QuantizedRegionCandidateAnalysis().run(graph)
        backend = candidates.compiled_model_fusions
        self.assertEqual(len(backend), 1)

        plan = build_compiled_model_plan(
            graph,
            _weights(graph),
            _target("wasm"),
        )

        self.assertEqual(graph.fingerprint(), fingerprint)
        self.assertEqual(plan.source_graph_fingerprint, fingerprint)
        opportunity = plan.region_opportunities[0]
        self.assertEqual(opportunity.candidate_id, backend[0].candidate_id)
        self.assertIn("preserve-portable-runtime-graph", opportunity.requirements)

    def test_weight_values_are_part_of_compiled_plan_identity(self):
        graph = _qlinear_graph()
        first_weights = _weights(graph)
        second_weights = {name: value.copy() for name, value in first_weights.items()}
        second_weights["w"][0, 0] = 1
        first = build_compiled_model_plan(
            graph, first_weights, _target("native-cpu"),
        )
        second = build_compiled_model_plan(
            graph, second_weights, _target("native-cpu"),
        )
        self.assertNotEqual(
            first.source_weights_fingerprint,
            second.source_weights_fingerprint,
        )
        self.assertNotEqual(first.plan_id, second.plan_id)


if __name__ == "__main__":
    unittest.main()
