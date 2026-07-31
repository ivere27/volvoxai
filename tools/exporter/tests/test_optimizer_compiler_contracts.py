from __future__ import annotations

import unittest

from tools.exporter.ir import GraphIR, IRDialect, OpNode, TensorValue
from tools.exporter.optimizer.analysis import (
    AnalysisError,
    AnalysisManager,
    GraphAnalysis,
    UseDefAnalysis,
)
from tools.exporter.optimizer.candidate import (
    Candidate,
    CostDirection,
    CostMetric,
    CostOrigin,
    CostVector,
    LegalityProof,
    RewritePolicy,
    RewriteSemantics,
)
from tools.exporter.optimizer.target import TargetEnvironment
from tools.exporter.pipeline import (
    IRPass,
    PassContract,
    PassResult,
    VerifiedPipeline,
)


def canonical_graph() -> GraphIR:
    graph = GraphIR("onnx", "fixture.onnx", dialect=IRDialect.CANONICAL)
    graph.add_tensor(TensorValue(
        "x", (1,), "float32", "float32", public_input=True,
    ))
    graph.add_tensor(TensorValue(
        "y", (1,), "float32", "float32", public_output=True,
    ))
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "identity", "Identity", {"input": "x"}, {"out": "y"},
    ))
    graph.outputs.append("y")
    graph.verify()
    return graph


def target_environment() -> TargetEnvironment:
    return TargetEnvironment(
        backend_profile="portable",
        compile_backend="native-cpu",
        tune_backend="native-cpu",
        compile_features=frozenset({"avx2"}),
        tune_features=frozenset({"avx2"}),
        compile_device_fingerprint="cpu-a",
        tune_device_fingerprint="cpu-a",
    )


class NodeCountAnalysis(GraphAnalysis[int]):
    name = "node-count"

    def __init__(self) -> None:
        self.runs = 0

    def run(self, graph: GraphIR, analyses: AnalysisManager) -> int:
        del analyses
        self.runs += 1
        return len(graph.nodes)


class DoubleNodeCountAnalysis(GraphAnalysis[int]):
    name = "double-node-count"

    def __init__(self) -> None:
        self.runs = 0

    def run(self, graph: GraphIR, analyses: AnalysisManager) -> int:
        del graph
        self.runs += 1
        return int(analyses.get("node-count")) * 2


class MutatingAnalysis(GraphAnalysis[int]):
    name = "mutating"

    def run(self, graph: GraphIR, analyses: AnalysisManager) -> int:
        del analyses
        graph.metadata["not-an-analysis"] = True
        return 1


class RenamePass(IRPass):
    name = "candidate-rename"
    contract = PassContract.preserving(IRDialect.CANONICAL)

    def run(self, graph: GraphIR) -> PassResult:
        graph.nodes[0].op_type = "Alias"
        return PassResult(1, touched_nodes=("identity",))


class StoreMutatingPass(IRPass):
    name = "store-mutation"
    contract = PassContract.preserving(IRDialect.CANONICAL)

    def __init__(self, store) -> None:
        self.store = store

    def run(self, graph: GraphIR) -> PassResult:
        graph.metadata["store-mutation"] = True
        self.store["new"] = [4, 5, 6]
        self.store["existing"].append(3)
        return PassResult(1)


class ForcedFailurePass(IRPass):
    name = "forced-failure"
    contract = PassContract.preserving(IRDialect.CANONICAL)

    def run(self, graph: GraphIR) -> PassResult:
        graph.metadata["failure-mutation"] = True
        raise RuntimeError("forced downstream failure")


class TestCandidate(Candidate):
    __test__ = False

    def __init__(
        self,
        graph: GraphIR,
        semantics: frozenset[RewriteSemantics],
    ) -> None:
        self._source_fingerprint = graph.fingerprint()
        self._semantics = semantics

    @property
    def candidate_id(self) -> str:
        return "test.rename"

    @property
    def source_fingerprint(self) -> str:
        return self._source_fingerprint

    @property
    def semantics(self) -> frozenset[RewriteSemantics]:
        return self._semantics

    def legality(
        self,
        graph: GraphIR,
        analyses: AnalysisManager,
        target: TargetEnvironment,
    ) -> LegalityProof:
        uses = analyses.get("use-def")
        if "y" not in uses.producers:
            return LegalityProof.reject(graph, target, "output has no producer")
        return LegalityProof.accept(graph, target, "output producer is known")

    def estimate_cost(
        self,
        graph: GraphIR,
        analyses: AnalysisManager,
        target: TargetEnvironment,
    ) -> CostVector:
        del graph, analyses, target
        return CostVector((CostMetric(
            "dispatches",
            1,
            "count",
            CostOrigin.STATIC_ESTIMATE,
        ),))

    def build_pass(
        self,
        graph: GraphIR,
        analyses: AnalysisManager,
        target: TargetEnvironment,
    ) -> IRPass:
        del graph, analyses, target
        return RenamePass()


class AnalysisManagerTests(unittest.TestCase):
    def test_cache_tracks_graph_fingerprint_and_dependencies(self):
        graph = canonical_graph()
        count = NodeCountAnalysis()
        double = DoubleNodeCountAnalysis()
        analyses = AnalysisManager(graph, (count, double))

        self.assertEqual(analyses.get("double-node-count"), 2)
        self.assertEqual(analyses.get("double-node-count"), 2)
        self.assertEqual((count.runs, double.runs), (1, 1))
        self.assertEqual(
            analyses.cached_analyses,
            ("double-node-count", "node-count"),
        )

        analyses.invalidate("node-count")
        self.assertEqual(analyses.cached_analyses, ())
        self.assertEqual(analyses.get("double-node-count"), 2)
        self.assertEqual((count.runs, double.runs), (2, 2))

        graph.metadata["revision"] = 2
        self.assertEqual(analyses.get("double-node-count"), 2)
        self.assertEqual((count.runs, double.runs), (3, 3))

    def test_builtin_use_def_analysis_is_typed(self):
        graph = canonical_graph()
        analyses = AnalysisManager(graph)
        use_def = analyses.require(UseDefAnalysis())
        self.assertEqual(use_def.producers["y"].node_index, 0)
        self.assertEqual(use_def.consumers["x"][0].port, "input")

    def test_analysis_mutation_is_rolled_back(self):
        graph = canonical_graph()
        before = graph.fingerprint()
        analyses = AnalysisManager(graph, (MutatingAnalysis(),))
        with self.assertRaisesRegex(AnalysisError, "mutated its input graph"):
            analyses.get("mutating")
        self.assertEqual(graph.fingerprint(), before)
        self.assertNotIn("not-an-analysis", graph.metadata)

    def test_duplicate_provider_names_fail_closed(self):
        graph = canonical_graph()
        analyses = AnalysisManager(graph, (NodeCountAnalysis(),))
        with self.assertRaisesRegex(AnalysisError, "already registered"):
            analyses.register(NodeCountAnalysis())


class TargetEnvironmentTests(unittest.TestCase):
    def test_profile_compile_and_tune_identities_are_independent(self):
        target = TargetEnvironment(
            backend_profile="portable",
            compile_backend="native-cpu",
            tune_backend="wasm",
            compile_features=frozenset({"avx2", "fma"}),
            tune_features=frozenset({"simd128"}),
        )
        self.assertEqual(target.backend_profile, "portable")
        self.assertEqual(target.compile_backend, "native-cpu")
        self.assertEqual(target.tune_backend, "wasm")
        self.assertTrue(target.uses_cross_backend_tuning)
        self.assertFalse(target.tuning_matches_compile_device)

        reordered = TargetEnvironment(
            backend_profile="portable",
            compile_backend="native-cpu",
            tune_backend="wasm",
            compile_features=frozenset({"fma", "avx2"}),
            tune_features=frozenset({"simd128"}),
        )
        self.assertEqual(
            target.legality_fingerprint(), reordered.legality_fingerprint(),
        )
        self.assertEqual(
            target.compile_fingerprint(), reordered.compile_fingerprint(),
        )
        self.assertEqual(
            target.measurement_fingerprint(),
            reordered.measurement_fingerprint(),
        )

    def test_target_fingerprints_separate_legality_compile_and_measurement(self):
        base = target_environment()
        other_tune = TargetEnvironment(
            backend_profile=base.backend_profile,
            compile_backend=base.compile_backend,
            tune_backend="wasm",
            compile_features=base.compile_features,
            tune_features=frozenset({"simd128"}),
            compile_device_fingerprint=base.compile_device_fingerprint,
            tune_device_fingerprint="browser-b",
        )
        self.assertEqual(
            base.legality_fingerprint(), other_tune.legality_fingerprint(),
        )
        self.assertEqual(
            base.compile_fingerprint(), other_tune.compile_fingerprint(),
        )
        self.assertNotEqual(
            base.measurement_fingerprint(), other_tune.measurement_fingerprint(),
        )

        other_compile = TargetEnvironment(
            backend_profile=base.backend_profile,
            compile_backend="wasm",
            tune_backend=base.tune_backend,
            tune_features=base.tune_features,
            tune_device_fingerprint=base.tune_device_fingerprint,
        )
        self.assertEqual(
            base.legality_fingerprint(), other_compile.legality_fingerprint(),
        )
        self.assertNotEqual(
            base.compile_fingerprint(), other_compile.compile_fingerprint(),
        )

    def test_matching_tune_device_is_explicit(self):
        target = target_environment()
        self.assertFalse(target.uses_cross_backend_tuning)
        self.assertTrue(target.tuning_matches_compile_device)

    def test_target_identifiers_are_validated(self):
        with self.assertRaisesRegex(ValueError, "compile backend"):
            TargetEnvironment("portable", "", "wasm")
        with self.assertRaisesRegex(TypeError, "collection"):
            TargetEnvironment(
                "portable",
                "native-cpu",
                "native-cpu",
                compile_features="avx2",  # type: ignore[arg-type]
            )


class CandidateContractTests(unittest.TestCase):
    def test_rewrite_policy_requires_explicit_numerical_opt_in(self):
        numerical = frozenset({RewriteSemantics.NUMERICAL_MIGRATION})
        self.assertFalse(RewritePolicy.exact().permits(numerical))
        self.assertTrue(RewritePolicy.qualified().permits(numerical))
        with self.assertRaisesRegex(ValueError, "exact rewrite"):
            RewritePolicy.qualified().permits(frozenset({
                RewriteSemantics.EXACT,
                RewriteSemantics.NUMERICAL_MIGRATION,
            }))

    def test_candidate_builds_an_ir_pass_for_verified_application(self):
        graph = canonical_graph()
        analyses = AnalysisManager(graph, (UseDefAnalysis(),))
        target = target_environment()
        candidate = TestCandidate(
            graph,
            frozenset({RewriteSemantics.EXACT}),
        )

        proof = candidate.preflight(
            graph, analyses, target, RewritePolicy.exact(),
        ).merge(candidate.legality(graph, analyses, target))
        self.assertTrue(proof.legal)
        self.assertTrue(proof.matches(graph, target))
        self.assertEqual(
            candidate.estimate_cost(graph, analyses, target)
            .get("dispatches")
            .value,  # type: ignore[union-attr]
            1,
        )

        rewrite = candidate.build_pass(graph, analyses, target)
        self.assertIsInstance(rewrite, IRPass)
        VerifiedPipeline([rewrite]).run(graph)
        self.assertEqual(graph.nodes[0].op_type, "Alias")
        stale = candidate.preflight(
            graph, analyses, target, RewritePolicy.exact(),
        )
        self.assertFalse(stale.legal)
        self.assertIn("different graph revision", stale.failures[0])

    def test_policy_rejection_is_a_legality_failure(self):
        graph = canonical_graph()
        analyses = AnalysisManager(graph)
        candidate = TestCandidate(
            graph,
            frozenset({RewriteSemantics.NUMERICAL_MIGRATION}),
        )
        proof = candidate.preflight(
            graph,
            analyses,
            target_environment(),
            RewritePolicy.exact(),
        )
        self.assertFalse(proof.legal)
        self.assertIn("numerical-migration", proof.failures[0])

    def test_legality_proofs_are_bound_to_graph_and_target(self):
        graph = canonical_graph()
        target = target_environment()
        accepted = LegalityProof.accept(graph, target, "shape is concrete")
        rejected = LegalityProof.reject(graph, target, "kernel is unavailable")
        merged = accepted.merge(rejected)
        self.assertFalse(merged.legal)
        self.assertEqual(merged.facts, ("shape is concrete",))
        self.assertEqual(merged.failures, ("kernel is unavailable",))

        graph.metadata["changed"] = True
        self.assertFalse(accepted.matches(graph, target))
        with self.assertRaisesRegex(ValueError, "different graph"):
            accepted.merge(LegalityProof.accept(graph, target, "new graph"))

    def test_sparse_cost_vectors_keep_workloads_separate(self):
        vector = CostVector((
            CostMetric(
                "latency",
                2.5,
                "ms",
                CostOrigin.MEASURED,
                workload="decode",
            ),
            CostMetric(
                "latency",
                40.0,
                "ms",
                CostOrigin.MEASURED,
                workload="prefill",
            ),
            CostMetric(
                "accuracy",
                0.96,
                "ratio",
                CostOrigin.MEASURED,
                direction=CostDirection.MAXIMIZE,
            ),
        ))
        self.assertEqual(vector.get("latency", workload="decode").value, 2.5)
        self.assertIsNone(vector.get("peak-memory"))
        with self.assertRaisesRegex(ValueError, "repeat"):
            vector.merge(CostVector((CostMetric(
                "latency",
                3.0,
                "ms",
                CostOrigin.STATIC_ESTIMATE,
                workload="decode",
            ),)))


class VerifiedPipelineTransactionTests(unittest.TestCase):
    def test_atomic_failure_restores_graph_and_mutable_tensor_store(self):
        graph = canonical_graph()
        fingerprint = graph.fingerprint()
        store = {"existing": [1, 2]}
        pipeline = VerifiedPipeline(
            (StoreMutatingPass(store), ForcedFailurePass()),
            mutable_stores=(store,),
        )

        with self.assertRaisesRegex(Exception, "forced downstream failure"):
            pipeline.run(graph)

        self.assertEqual(graph.fingerprint(), fingerprint)
        self.assertEqual(store, {"existing": [1, 2]})


if __name__ == "__main__":
    unittest.main()
