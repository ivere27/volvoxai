from __future__ import annotations

import unittest

from tools.exporter.errors import Diagnostic, ExporterError
from tools.exporter.ir import GraphIR, IRDialect, OpNode, TensorValue
from tools.exporter.pipeline import (
    IRPass,
    PassContract,
    PassGroup,
    PipelineMetadata,
    PassResult,
    VerifiedPipeline,
)


def canonical_graph() -> GraphIR:
    graph = GraphIR("onnx", "fixture.onnx", dialect=IRDialect.CANONICAL)
    graph.add_tensor(TensorValue(
        "x", (1,), "float32", "float32", public_input=True))
    graph.add_tensor(TensorValue(
        "y", (1,), "float32", "float32", public_output=True))
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "identity", "Identity", {"input": "x"}, {"out": "y"}))
    graph.outputs.append("y")
    graph.verify()
    return graph


class RenamePass(IRPass):
    name = "rename-identity"
    contract = PassContract.preserving(IRDialect.CANONICAL)

    def run(self, graph: GraphIR) -> PassResult:
        graph.nodes[0].op_type = "Alias"
        return PassResult(1, touched_nodes=("identity",))


class BrokenPass(IRPass):
    name = "broken"
    contract = PassContract.preserving(IRDialect.CANONICAL)

    def run(self, graph: GraphIR) -> PassResult:
        graph.nodes[0] = OpNode.from_maps(
            "identity", "Identity", {"input": "missing"}, {"out": "y"})
        return PassResult(1)


class CountdownPass(IRPass):
    name = "countdown"
    contract = PassContract.preserving(IRDialect.CANONICAL, repeatable=True)

    def __init__(self) -> None:
        self.remaining = 2

    def run(self, graph: GraphIR) -> PassResult:
        if not self.remaining:
            return PassResult(0)
        self.remaining -= 1
        graph.metadata["countdown"] = self.remaining
        return PassResult(1)


class PipelineTests(unittest.TestCase):
    def test_pipeline_metadata_rejects_non_v1_contract_versions(self):
        values = {
            "registry_schema_version": 1,
            "registry_sha256": "optimizer-registry-sha",
            "kernel_registry_sha256": "kernel-registry-sha",
            "recipe_id": "runtime-package",
            "recipe_version": 1,
            "backend_profile": "portable",
            "backend_profile_members": ("cpu-js", "wasm"),
            "compile_backend": "wasm",
            "tune_backend": "cpu-js",
            "group_ids": ("runtime-validation",),
            "pass_ids": ("runtime-vocabulary",),
            "allowed_semantics": ("exact",),
            "allow_calibration": False,
            "allow_public_abi_change": False,
        }
        metadata = PipelineMetadata(**values)
        self.assertEqual(metadata.registry_schema_version, 1)
        self.assertEqual(metadata.recipe_version, 1)

        for field in ("registry_schema_version", "recipe_version"):
            with self.subTest(field=field):
                invalid = {**values, field: 2}
                with self.assertRaisesRegex(ValueError, "both be exactly 1"):
                    PipelineMetadata(**invalid)

    def test_verified_pass_runs_and_reports_fingerprints(self):
        graph = canonical_graph()
        report = VerifiedPipeline([RenamePass()]).run(graph)
        self.assertEqual(graph.nodes[0].op_type, "Alias")
        self.assertEqual(report.total_changes, 1)
        self.assertNotEqual(report.runs[0].before, report.runs[0].after)

    def test_invalid_rewrite_rolls_back_whole_pipeline(self):
        graph = canonical_graph()
        before = graph.fingerprint()
        with self.assertRaises(ExporterError):
            VerifiedPipeline([RenamePass(), BrokenPass()]).run(graph)
        self.assertEqual(graph.fingerprint(), before)

    def test_fixed_point_is_explicit_and_converges(self):
        graph = canonical_graph()
        countdown = CountdownPass()
        report = VerifiedPipeline([
            PassGroup((countdown,), fixed_point=True, max_iterations=4),
        ]).run(graph)
        self.assertEqual(countdown.remaining, 0)
        self.assertEqual(report.total_changes, 2)
        self.assertEqual(len(report.runs), 3)

    def test_change_count_mismatch_is_rejected(self):
        class Liar(IRPass):
            name = "liar"
            contract = PassContract.preserving(IRDialect.CANONICAL)

            def run(self, graph: GraphIR) -> PassResult:
                graph.metadata["changed"] = True
                return PassResult(0)

        graph = canonical_graph()
        before = graph.fingerprint()
        with self.assertRaises(ExporterError) as caught:
            VerifiedPipeline([Liar()]).run(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXOPT004")
        self.assertEqual(graph.fingerprint(), before)

    def test_pass_metrics_are_structured_nonnegative_counters(self):
        result = PassResult(
            0,
            metrics=(("candidates", 3), ("refused", 1)),
        )
        self.assertEqual(dict(result.metrics), {"candidates": 3, "refused": 1})
        for metrics in (
            (("duplicate", 1), ("duplicate", 2)),
            (("negative", -1),),
        ):
            with self.subTest(metrics=metrics):
                with self.assertRaisesRegex(ValueError, "pass metrics"):
                    PassResult(0, metrics=metrics)

    def test_pass_diagnostics_are_typed_and_propagated(self):
        diagnostic = Diagnostic(
            code="VXTEST001",
            message="kept one candidate",
            stage="test-retained",
            source_node="identity",
        )

        class AuditPass(IRPass):
            name = "audit"
            contract = PassContract.preserving(IRDialect.CANONICAL)

            def run(self, graph: GraphIR) -> PassResult:
                return PassResult(0, diagnostics=(diagnostic,))

        report = VerifiedPipeline([AuditPass()]).run(canonical_graph())
        self.assertEqual(report.runs[0].diagnostics, (diagnostic,))
        with self.assertRaisesRegex(TypeError, "pass diagnostics"):
            PassResult(0, diagnostics=("not-a-diagnostic",))

    def test_dialect_contract_is_enforced(self):
        graph = canonical_graph()
        graph.dialect = IRDialect.SOURCE
        with self.assertRaises(ExporterError) as caught:
            VerifiedPipeline([RenamePass()]).run(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXOPT002")


if __name__ == "__main__":
    unittest.main()
