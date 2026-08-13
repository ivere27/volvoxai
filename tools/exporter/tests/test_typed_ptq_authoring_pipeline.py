from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.exporter.errors import ExporterError
from tools.exporter.generated.optimizer_registry import PASSES
from tools.exporter.optimizer.candidate import RewritePolicy, RewriteSemantics
from tools.exporter.optimizer.registry_resolver import (
    PASS_FACTORIES,
    OptimizerRegistryError,
    RegistryPipelineRequest,
    RuntimePassFactoryContext,
    resolve_runtime_pipeline,
)
from tools.exporter.optimizer.target import TargetEnvironment
from tools.exporter.optimizer.typed_pipeline import (
    author_runtime_ptq_graph,
    author_runtime_ptq_package,
    serialize_pipeline_report,
)
from tools.exporter.optimizer.typed_ptq_authoring import RuntimePTQAuthoringPass
from tools.exporter.reference_executor import execute_reference
from tools.exporter.runtime_ir import import_runtime_package
from tools.exporter.typed_ptq import (
    CalibrationTable,
    PTQConfig,
    calibration_profile_from_ranges,
)


def _linear_package() -> tuple[dict, dict[str, np.ndarray]]:
    document = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "dense",
            "opType": "Linear",
            "inputs": {"input": "x", "weight": "weight", "bias": "bias"},
            "outputs": {"out": {
                "tensor": "y", "shape": [2, 2], "dtype": "float32",
            }},
            "params": {"weight_layout": "din_dout"},
        }],
    }
    tensors = {
        "weight": np.asarray([
            [0.5, -0.25],
            [1.0, 0.75],
            [-0.5, 0.25],
        ], dtype=np.float32),
        "bias": np.asarray([0.125, -0.25], dtype=np.float32),
    }
    return document, tensors


def _calibrated_fixture():
    document, tensors = _linear_package()
    graph = import_runtime_package(document, tensors)
    sample = np.asarray([
        [1.0, -2.0, 0.5],
        [-0.75, 0.25, 2.0],
    ], dtype=np.float32)
    execution = execute_reference(graph, tensors, {"x": sample})
    table = CalibrationTable(graph)
    table.observe_reference(execution)
    return document, tensors, table.profile()


def _target() -> TargetEnvironment:
    return TargetEnvironment(
        backend_profile="portable",
        compile_backend="wasm",
        tune_backend="native-cpu",
    )


def _authoring_policy() -> RewritePolicy:
    return RewritePolicy(frozenset({
        RewriteSemantics.EXACT,
        RewriteSemantics.QUANTIZATION_AUTHORING,
    }))


class RuntimePTQAuthoringPipelineTests(unittest.TestCase):
    def test_portable_broadcast_ptq_registry_contract_emits_expand_then_qadd(self):
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {
                "x": {"shape": [1, 2, 4], "dtype": "float32"},
                "route": {"shape": [1, 1, 4], "dtype": "float32"},
            },
            "outputs": ["y"],
            "nodes": [
                {
                    "id": "expand-route",
                    "opType": "Expand",
                    "inputs": {"input": "route"},
                    "outputs": {"out": {
                        "tensor": "expanded_route",
                        "shape": [1, 2, 4],
                        "dtype": "float32",
                    }},
                    "params": {"shape": [1, 2, 4]},
                },
                {
                    "id": "broadcast-add",
                    "opType": "Add",
                    "inputs": {"a": "x", "b": "expanded_route"},
                    "outputs": {"out": {
                        "tensor": "y",
                        "shape": [1, 2, 4],
                        "dtype": "float32",
                    }},
                    "params": {},
                },
            ],
        }
        graph = import_runtime_package(document, {})
        sample = {
            "x": np.asarray(
                [[[-1.0, 0.0, 1.0, 2.0], [0.5, -0.5, 1.5, -1.5]]],
                dtype=np.float32,
            ),
            "route": np.asarray(
                [[[0.25, -0.25, 0.5, -0.5]]], dtype=np.float32,
            ),
        }
        table = CalibrationTable(graph)
        table.observe_reference(execute_reference(graph, {}, sample))

        report = author_runtime_ptq_graph(
            graph,
            {},
            table.profile(),
            shape_profile={},
            target_environment=_target(),
        )

        descriptor = PASSES["runtime-ptq-authoring"]
        self.assertEqual(descriptor["version"], 1)
        self.assertIn("Expand", descriptor["emitted_operators"])
        self.assertEqual(len(descriptor["target_rules"]), 1)
        self.assertIn(
            "Expand", descriptor["target_rules"][0]["required_operators"],
        )
        self.assertEqual(report.metadata.backend_profile, "portable")
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            [
                "Expand",
                "QuantizeLinear",
                "QuantizeLinear",
                "QAdd",
                "DequantizeLinear",
            ],
        )
        expand = graph.nodes[0]
        route_quantize = graph.nodes[2]
        qadd = graph.nodes[3]
        self.assertEqual(
            route_quantize.input_map()["input"], expand.output_map()["out"],
        )
        self.assertEqual(
            qadd.input_map()["b"], route_quantize.output_map()["out"],
        )
        self.assertEqual(
            graph.tensors[route_quantize.output_map()["out"]].quantization,
            graph.tensors[qadd.input_map()["b"]].quantization,
        )

    def test_registry_requires_both_authoring_policy_and_calibration_opt_in(self):
        _, tensors, calibration = _calibrated_fixture()
        config = PTQConfig(activation_dtype="uint8")
        context = RuntimePassFactoryContext(
            tensor_data=dict(tensors),
            ptq_calibration=calibration,
            ptq_config=config,
            ptq_selected_nodes=("dense",),
        )

        def request(*, policy: RewritePolicy, allow_calibration: bool):
            return RegistryPipelineRequest(
                factory_context=context,
                target=_target(),
                rewrite_policy=policy,
                allow_calibration=allow_calibration,
                selection_features=frozenset({"ptq-authoring"}),
            )

        with self.assertRaisesRegex(
            OptimizerRegistryError,
            "rewrite policy rejects.*quantization-authoring",
        ):
            resolve_runtime_pipeline(request(
                policy=RewritePolicy.exact(),
                allow_calibration=True,
            ))
        with self.assertRaisesRegex(
            OptimizerRegistryError,
            "requires explicit calibration policy",
        ):
            resolve_runtime_pipeline(request(
                policy=_authoring_policy(),
                allow_calibration=False,
            ))

        pipeline = resolve_runtime_pipeline(request(
            policy=_authoring_policy(),
            allow_calibration=True,
        ))
        self.assertEqual(pipeline.metadata.recipe_id, "runtime-ptq-authoring")
        self.assertEqual(
            pipeline.metadata.group_ids,
            (
                "runtime-validation",
                "runtime-ptq-authoring-stage",
                "runtime-ptq-post-cleanup-fixed-point",
            ),
        )
        authored = next(
            item
            for group in pipeline.groups
            for item in group.passes
            if item.name == "runtime-ptq-authoring"
        )
        self.assertIsInstance(authored, RuntimePTQAuthoringPass)
        self.assertIs(authored.calibration, calibration)
        self.assertIs(authored.config, config)
        self.assertEqual(authored.selected_nodes, ("dense",))

        implementation = (
            "tools.exporter.optimizer.typed_ptq_authoring:"
            "RuntimePTQAuthoringPass"
        )
        with self.assertRaisesRegex(
            OptimizerRegistryError,
            "requires an explicit CalibrationProfile",
        ):
            PASS_FACTORIES[implementation].build(RuntimePassFactoryContext(
                tensor_data={},
            ))

    def test_selection_feature_and_calibration_profile_must_match(self):
        _, tensors, calibration = _calibrated_fixture()
        for context, features in (
            (
                RuntimePassFactoryContext(
                    tensor_data=dict(tensors),
                    ptq_calibration=calibration,
                ),
                (),
            ),
            (
                RuntimePassFactoryContext(tensor_data=dict(tensors)),
                ("ptq-authoring",),
            ),
        ):
            with self.subTest(features=features):
                request = RegistryPipelineRequest(
                    factory_context=context,
                    target=_target(),
                    rewrite_policy=_authoring_policy(),
                    allow_calibration=True,
                    selection_features=frozenset(features),
                )
                with self.assertRaisesRegex(
                    OptimizerRegistryError,
                    "selection must exactly match.*CalibrationProfile",
                ):
                    resolve_runtime_pipeline(request)

    def test_author_entry_uses_exact_observed_graph_then_only_post_cleanup(self):
        document, tensors, calibration = _calibrated_fixture()
        source_document = copy.deepcopy(document)
        source_tensors = {
            name: value.copy() for name, value in tensors.items()
        }

        authored, authored_tensors, report = author_runtime_ptq_package(
            document,
            tensors,
            calibration,
            shape_profile={},
            target_environment=_target(),
        )

        self.assertEqual(document, source_document)
        self.assertEqual(set(tensors), set(source_tensors))
        for name, expected in source_tensors.items():
            np.testing.assert_array_equal(tensors[name], expected)
        self.assertEqual(
            [node["opType"] for node in authored["nodes"]],
            ["QuantizeLinear", "QLinear", "DequantizeLinear"],
        )
        self.assertNotIn("weight", authored_tensors)
        self.assertNotIn("bias", authored_tensors)
        restored = import_runtime_package(authored, authored_tensors)
        self.assertEqual(restored.nodes[1].op_type, "QLinear")

        self.assertEqual(report.metadata.recipe_id, "runtime-ptq-authoring")
        self.assertEqual(
            report.metadata.pass_ids,
            (
                "runtime-vocabulary",
                "runtime-ptq-authoring",
                "runtime-canonicalize",
                "redundant-qdq",
                "runtime-dead-code",
            ),
        )
        author_index = next(
            index for index, run in enumerate(report.runs)
            if run.name == "runtime-ptq-authoring"
        )
        self.assertEqual(
            tuple(run.name for run in report.runs[:author_index]),
            ("runtime-vocabulary",),
        )

        stale_document = copy.deepcopy(document)
        stale_document["nodes"][0]["id"] = "renamed-after-calibration"
        with self.assertRaises(ExporterError) as caught:
            author_runtime_ptq_package(
                stale_document,
                tensors,
                calibration,
                shape_profile={},
                target_environment=_target(),
            )
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ047")
        for name, expected in source_tensors.items():
            np.testing.assert_array_equal(tensors[name], expected)

    def test_typed_graph_entry_is_registry_authoritative_and_transactional(self):
        document, tensors, calibration = _calibrated_fixture()
        graph = import_runtime_package(document, tensors)
        mutable_tensors = dict(tensors)
        report = author_runtime_ptq_graph(
            graph,
            mutable_tensors,
            calibration,
            shape_profile={},
            target_environment=_target(),
        )
        self.assertEqual(report.metadata.recipe_id, "runtime-ptq-authoring")
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QLinear", "DequantizeLinear"],
        )
        authoring_run = next(
            run for run in report.runs
            if run.name == "runtime-ptq-authoring"
        )
        self.assertEqual(dict(authoring_run.metrics)["nodes_quantized"], 1)
        self.assertEqual(
            dict(authoring_run.metrics)["retained_f32_instances"], 0,
        )

        stale_graph = import_runtime_package(document, tensors)
        stale_graph.nodes[0].name = "renamed-after-calibration"
        stale_tensors = {
            name: value.copy() for name, value in tensors.items()
        }
        before = stale_graph.fingerprint()
        with self.assertRaises(ExporterError) as caught:
            author_runtime_ptq_graph(
                stale_graph,
                stale_tensors,
                calibration,
                shape_profile={},
                target_environment=_target(),
            )
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ047")
        self.assertEqual(stale_graph.fingerprint(), before)
        for name, expected in tensors.items():
            np.testing.assert_array_equal(stale_tensors[name], expected)

    def test_symbolic_authoring_runs_the_whole_pipeline_and_keeps_abi(self):
        document, tensors = _linear_package()
        document["dimensions"] = {"B": {"min": 1, "max": 4}}
        document["inputs"]["x"]["shape"] = ["B", 3]
        document["nodes"][0]["outputs"]["out"]["shape"] = ["B", 2]
        graph = import_runtime_package(document, tensors)
        calibration = calibration_profile_from_ranges(
            graph,
            {
                "x": {
                    "min": -2.0, "max": 2.0, "samples": 2, "elements": 18,
                },
                "y": {
                    "min": -1.0, "max": 1.0, "samples": 2, "elements": 12,
                },
            },
            sample_count=2,
            sample_digest="d" * 64,
        )

        report = author_runtime_ptq_graph(
            graph,
            tensors,
            calibration,
            shape_profile=None,
            target_environment=_target(),
        )

        self.assertEqual(
            [(item.name, item.min, item.max)
             for item in graph.shape_environment.dimensions],
            [("B", 1, 4)],
        )
        self.assertEqual(graph.tensors["x"].shape, ("B", 3))
        self.assertEqual(graph.tensors["y"].shape, ("B", 2))
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QLinear", "DequantizeLinear"],
        )
        authoring = next(
            run for run in report.runs if run.name == "runtime-ptq-authoring"
        )
        canonicalize = next(
            run for run in report.runs if run.name == "runtime-canonicalize"
        )
        self.assertFalse(authoring.skipped)
        # `runtime-canonicalize` is symbolic-preserving since the optimizer
        # migration, so a profile-free symbolic authoring run executes it
        # instead of skipping it, and the ABI above still holds.
        self.assertFalse(canonicalize.skipped)

    def test_implicit_broadcast_fails_before_ptq_authoring(self):
        shape = [1, 2, 4]
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {
                "x": {"shape": shape, "dtype": "float32"},
                "route": {"shape": [4], "dtype": "float32"},
            },
            "outputs": ["y"],
            "nodes": [{
                "id": "broadcast-add",
                "opType": "Add",
                "inputs": {"a": "x", "b": "route"},
                "outputs": {"out": {
                    "tensor": "y", "shape": shape, "dtype": "float32",
                }},
                "params": {},
            }],
        }
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR035")
        self.assertIn("exactly equal shapes", caught.exception.diagnostic.message)


if __name__ == "__main__":
    unittest.main()
