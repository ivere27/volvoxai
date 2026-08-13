from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper

from tools.exporter.errors import ExporterError
from tools.exporter.frontend_onnx import OnnxCompiler, compile_onnx_model
from tools.exporter.importers.onnx import import_onnx_source
from tools.exporter.ir import GraphIR, IRDialect, OpAttribute
from tools.exporter.optimizer.typed_pipeline import optimize_runtime_package
from tools.exporter.operator_shape_contracts import OperatorShapeContract
from tools.exporter.pipeline import (
    IRPass,
    PassContract,
    PassResult,
    VerifiedPipeline,
)
from tools.exporter.reference_executor import execute_reference
from tools.exporter.runtime_ir import (
    export_runtime_package,
    import_runtime_package,
)
from tools.exporter.shape_system import ShapeEnvironment


def dynamic_identity_document() -> dict:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {
            "batch": {"min": 1, "max": 8},
        },
        "inputs": {
            "x": {"dtype": "float32", "shape": ["batch", 4]},
        },
        "nodes": [{
            "id": "identity",
            "opType": "Identity",
            "inputs": {"input": "x"},
            "outputs": {
                "out": {
                    "tensor": "y",
                    "dtype": "float32",
                    "shape": ["batch", 4],
                },
            },
            "params": {},
        }],
        "outputs": ["y"],
    }


def dynamic_reshape_expand_document() -> dict:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {
            "batch": {"min": 1, "max": 8},
        },
        "inputs": {
            "x": {"dtype": "float32", "shape": ["batch", 4]},
        },
        "nodes": [
            {
                "id": "reshape",
                "opType": "Reshape",
                "inputs": {"input": "x"},
                "outputs": {
                    "out": {
                        "tensor": "reshaped",
                        "dtype": "float32",
                        "shape": ["batch", 1, 4],
                    },
                },
                "params": {"shape": ["batch", 1, 4]},
            },
            {
                "id": "expand",
                "opType": "Expand",
                "inputs": {"input": "reshaped"},
                "outputs": {
                    "out": {
                        "tensor": "expanded",
                        "dtype": "float32",
                        "shape": ["batch", 3, 4],
                    },
                },
                "params": {"shape": ["batch", 3, 4]},
            },
        ],
        "outputs": ["expanded"],
    }


def dynamic_resize_document() -> dict:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {
            "batch": {"min": 1, "max": 8},
            "height": {"min": 1, "max": 256},
            "width": {"min": 1, "max": 256},
            "resized_height": {"min": 1, "max": 512},
            "resized_width": {"min": 1, "max": 512},
        },
        "inputs": {
            "x": {
                "dtype": "float32",
                "shape": ["batch", "height", "width", 4],
            },
        },
        "nodes": [{
            "id": "resize",
            "opType": "Resize",
            "inputs": {"input": "x"},
            "outputs": {
                "out": {
                    "tensor": "resized",
                    "dtype": "float32",
                    "shape": [
                        "batch", "resized_height", "resized_width", 4,
                    ],
                },
            },
            "params": {"mode": "nearest"},
        }],
        "outputs": ["resized"],
    }


def symbolic_identity_onnx(*, anonymous: bool = False):
    shape = [None if anonymous else "batch", 4]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"], name="identity")],
            "dynamic_identity",
            [x],
            [y],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )


def symbolic_add_onnx(right_shape):
    left_shape = ["batch", "sequence", 4]
    left = helper.make_tensor_value_info("left", TensorProto.FLOAT, left_shape)
    right = helper.make_tensor_value_info("right", TensorProto.FLOAT, right_shape)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, left_shape)
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["left", "right"], ["output"], name="add")],
            "dynamic_add",
            [left, right],
            [output],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )


class MutateThenFailConcretePass(IRPass):
    name = "test-mutate-then-fail-concrete"
    contract = PassContract.concrete_profile_only(IRDialect.RUNTIME)

    def run(self, graph: GraphIR) -> PassResult:
        graph.nodes[0].op_type = "Mul"
        raise RuntimeError("forced failure after mutation")


class NoOpConcretePass(IRPass):
    name = "test-no-op-concrete"
    contract = PassContract.concrete_profile_only(IRDialect.RUNTIME)

    def run(self, graph: GraphIR) -> PassResult:
        return PassResult(0)


class WidenConstraintPass(IRPass):
    name = "test-widen-constraint"
    contract = PassContract.constraint_refining(IRDialect.RUNTIME)

    def run(self, graph: GraphIR) -> PassResult:
        graph.shape_environment = ShapeEnvironment(({
            "name": "batch", "min": 1, "max": 16,
        },))
        return PassResult(1)


class AddConstraintPass(IRPass):
    name = "test-add-constraint"
    contract = PassContract.constraint_refining(IRDialect.RUNTIME)

    def run(self, graph: GraphIR) -> PassResult:
        graph.shape_environment = ShapeEnvironment((
            {"name": "batch", "min": 1, "max": 8},
            {"name": "sequence", "min": 1, "max": 32},
        ))
        return PassResult(1)


class RefineAndMutatePublicInputPass(IRPass):
    name = "test-refine-and-mutate-public-input"
    contract = PassContract.constraint_refining(IRDialect.RUNTIME)

    def run(self, graph: GraphIR) -> PassResult:
        graph.shape_environment = ShapeEnvironment(({
            "name": "batch", "min": 2, "max": 8,
        },))
        graph.tensors["x"].shape = ("batch", 5)
        return PassResult(1)


class MutatePreservingEnvironmentPass(IRPass):
    name = "test-mutate-preserving-environment"
    contract = PassContract.preserving(IRDialect.RUNTIME)

    def run(self, graph: GraphIR) -> PassResult:
        graph.shape_environment = ShapeEnvironment(({
            "name": "batch", "min": 2, "max": 8,
        },))
        return PassResult(1)


class MutatePreservingPublicInputPass(IRPass):
    name = "test-mutate-preserving-public-input"
    contract = PassContract.preserving(IRDialect.RUNTIME)

    def run(self, graph: GraphIR) -> PassResult:
        graph.tensors["x"].shape = ("batch", 5)
        return PassResult(1)


class DynamicRuntimeIRTests(unittest.TestCase):
    def test_actual_onnx_frontend_publishes_only_bounded_closed_v1(self):
        with tempfile.TemporaryDirectory(prefix="volvox-onnx-dynamic-") as directory:
            path = Path(directory) / "identity.onnx"
            onnx.save(symbolic_identity_onnx(), path)

            compiler = OnnxCompiler(
                str(path),
                dimension_bounds={"batch": {"min": 1, "max": 8}},
            )
            document, weights = compiler.lower()

        self.assertEqual(weights, {})
        self.assertEqual(set(document), {
            "format", "dimensions", "inputs", "nodes", "outputs",
        })
        self.assertEqual(document["dimensions"], {
            "batch": {"min": 1, "max": 8},
        })
        self.assertEqual(document["inputs"], {
            "input0": {"dtype": "float32", "shape": ["batch", 4]},
        })
        self.assertEqual(document["nodes"][0], {
            "id": "node_0",
            "opType": "Identity",
            "inputs": {"input": "input0"},
            "outputs": {"out": {
                "tensor": "output0",
                "dtype": "float32",
                "shape": ["batch", 4],
            }},
            "params": {},
        })
        self.assertEqual(document["outputs"], ["output0"])
        self.assertEqual(
            compiler.publication_report["source_ir"]["dialect"], "source",
        )

    def test_actual_onnx_frontend_requires_bounds_or_explicit_static_binding(self):
        with tempfile.TemporaryDirectory(prefix="volvox-onnx-dynamic-") as directory:
            path = Path(directory) / "identity.onnx"
            onnx.save(symbolic_identity_onnx(), path)
            with self.assertRaises(ExporterError) as missing:
                OnnxCompiler(str(path))
            self.assertEqual(missing.exception.diagnostic.code, "VXIR054")

            document, _weights = OnnxCompiler(
                str(path), input_shapes={"x": [3, 4]},
            ).lower()

        self.assertEqual(document["dimensions"], {})
        self.assertEqual(document["inputs"]["input0"]["shape"], [3, 4])
        self.assertEqual(
            document["nodes"][0]["outputs"]["out"]["shape"], [3, 4],
        )

    def test_actual_onnx_frontend_preserves_symbolic_broadcast_axes(self):
        bounds = {
            "batch": {"min": 1, "max": 8},
            "sequence": {"min": 1, "max": 512},
        }
        cases = (
            ("trailing_vector", [4]),
            ("unit_batch", [1, "sequence", 4]),
        )
        with tempfile.TemporaryDirectory(prefix="volvox-onnx-dynamic-") as directory:
            for name, right_shape in cases:
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.onnx"
                    onnx.save(symbolic_add_onnx(right_shape), path)
                    document, _weights = OnnxCompiler(
                        str(path), dimension_bounds=bounds,
                    ).lower()
                    self.assertEqual(
                        document["nodes"][-1]["outputs"]["out"]["shape"],
                        ["batch", "sequence", 4],
                    )
                    self.assertEqual(document["nodes"][-1]["opType"], "Add")
                    self.assertIn("Expand", {
                        node["opType"] for node in document["nodes"]
                    })
                    self.assertEqual(document["dimensions"], {
                        "batch": {"min": 1, "max": 8},
                        "sequence": {"min": 1, "max": 512},
                    })

    def test_onnx_compile_report_is_out_of_band_and_observable(self):
        with tempfile.TemporaryDirectory(prefix="volvox-onnx-report-") as directory:
            root = Path(directory)
            source = root / "identity.onnx"
            destination = root / "model.safetensors"
            onnx.save(symbolic_identity_onnx(), source)
            reports = []

            document = compile_onnx_model(
                str(source),
                str(destination),
                dimension_bounds={"batch": {"min": 1, "max": 8}},
                report_callback=reports.append,
                log=lambda _message: None,
            )

            persisted = json.loads((root / "graph.json").read_text())
        self.assertEqual(document, persisted)
        self.assertNotIn("source", document)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["source_ir"]["dialect"], "source")
        self.assertEqual(reports[0]["package_class"], "fp32")
        self.assertIn("typed_optimizer", reports[0])
        self.assertIn("quantization_parameters", reports[0])

    def test_onnx_dim_param_survives_bounds_import_optimization_and_emission(self):
        source = import_onnx_source(
            symbolic_identity_onnx(),
            dimension_bounds={"batch": {"min": 1, "max": 8}},
        )
        self.assertEqual(source.tensors["x"].shape, ("batch", 4))
        self.assertEqual(source.tensors["y"].shape, ("batch", 4))
        self.assertEqual(source.shape_environment.dimensions[0].name, "batch")

        optimized, tensors, report = optimize_runtime_package(
            dynamic_identity_document(),
            {},
        )
        self.assertEqual(tensors, {})
        self.assertEqual(optimized["dimensions"], {
            "batch": {"min": 1, "max": 8},
        })
        self.assertEqual(optimized["inputs"]["x"]["shape"], ["batch", 4])
        self.assertEqual(
            optimized["nodes"][0]["outputs"]["out"]["shape"],
            ["batch", 4],
        )
        self.assertGreater(len(report.runs), 0)

    def test_missing_named_and_anonymous_bounds_fail_without_invented_maxima(self):
        with self.assertRaises(ExporterError) as named:
            import_onnx_source(
                symbolic_identity_onnx(),
                dimension_bounds={},
            )
        self.assertEqual(named.exception.diagnostic.code, "VXIR054")

        with self.assertRaises(ExporterError) as anonymous:
            import_onnx_source(
                symbolic_identity_onnx(anonymous=True),
                dimension_bounds={},
            )
        self.assertEqual(anonymous.exception.diagnostic.code, "VXIR055")

    def test_explicit_binding_emits_constant_only_graph_in_same_v1_schema(self):
        graph = import_runtime_package(dynamic_identity_document(), {})
        document, tensors = export_runtime_package(
            graph,
            {},
            shape_profile={"batch": 3},
        )
        self.assertEqual(tensors, {})
        self.assertEqual(document["format"], "volvox-graph/v1")
        self.assertEqual(document["dimensions"], {})
        self.assertEqual(document["inputs"]["x"]["shape"], [3, 4])
        output = document["nodes"][0]["outputs"]["out"]
        self.assertEqual(output, {
            "tensor": "y", "dtype": "float32", "shape": [3, 4],
        })
        self.assertNotIn("outputs_shape", document["nodes"][0])
        self.assertNotIn("outputs_dtype", document["nodes"][0])

    def test_dynamic_and_constant_publications_are_numerically_identical(self):
        source = import_runtime_package(dynamic_reshape_expand_document(), {})
        dynamic_document, dynamic_tensors = export_runtime_package(source, {})
        constant_document, constant_tensors = export_runtime_package(
            source,
            {},
            shape_profile={"batch": 3},
        )
        self.assertEqual(
            dynamic_document["format"],
            constant_document["format"],
        )
        self.assertNotEqual(dynamic_document["dimensions"], {})
        self.assertEqual(constant_document["dimensions"], {})

        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        dynamic_result = execute_reference(
            import_runtime_package(dynamic_document, dynamic_tensors),
            dynamic_tensors,
            {"x": values},
        )
        constant_result = execute_reference(
            import_runtime_package(constant_document, constant_tensors),
            constant_tensors,
            {"x": values},
        )
        np.testing.assert_array_equal(
            dynamic_result.outputs["expanded"],
            constant_result.outputs["expanded"],
        )
        self.assertEqual(
            dynamic_result.outputs["expanded"].shape,
            (3, 3, 4),
        )

    def test_profile_binding_concretizes_only_shape_schema_fields_and_ts_loads(self):
        graph = import_runtime_package(dynamic_reshape_expand_document(), {})
        document, tensors = export_runtime_package(
            graph,
            {},
            shape_profile={"batch": 3},
        )
        self.assertEqual(tensors, {})
        self.assertEqual(document["dimensions"], {})
        self.assertEqual(document["inputs"]["x"]["shape"], [3, 4])
        self.assertEqual(
            [node["params"]["shape"] for node in document["nodes"]],
            [[3, 1, 4], [3, 3, 4]],
        )
        self.assertEqual(
            [node["outputs"]["out"]["shape"] for node in document["nodes"]],
            [[3, 1, 4], [3, 3, 4]],
        )
        import_runtime_package(document, {})

        repository = Path(__file__).resolve().parents[3]
        script = """
import { ModelLoader } from './ts/core/ModelLoader.ts';
let source = '';
for await (const chunk of process.stdin) source += chunk;
const loaded = await ModelLoader.load([], {
  graphUrl: 'graph.json',
  fetch: async () => ({ ok: true, text: async () => source }),
});
const graph = loaded.graph;
process.stdout.write(JSON.stringify({
  outputs: graph.outputs,
  shapes: graph.nodes.map((node) => node.params.shape),
}));
"""
        loaded = subprocess.run(
            ["node", "--import", "tsx", "--input-type=module", "-e", script],
            cwd=repository,
            input=json.dumps(document),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(loaded.returncode, 0, loaded.stderr)
        self.assertEqual(json.loads(loaded.stdout), {
            "outputs": ["expanded"],
            "shapes": [[3, 1, 4], [3, 3, 4]],
        })

        # Binding is schema-scoped: an arbitrary semantic string equal to a
        # symbol is not recursively rewritten.
        graph.nodes[0].attributes = (OpAttribute(
            "params",
            "volvox.params",
            {"shape": ["batch", 1, 4], "debug_label": "batch"},
        ),)
        bound = graph.bind_shape_profile({"batch": 2})
        params = bound.nodes[0].attributes[0].value
        self.assertEqual(params["shape"], [2, 1, 4])
        self.assertEqual(params["debug_label"], "batch")

    def test_closed_schema_rejects_unknown_root_input_and_node_fields(self):
        cases = (
            (lambda document: document.update({"source": {}}), "VXRTIR038"),
            (
                lambda document: document["inputs"]["x"].update(
                    {"source_name": "tokens"}
                ),
                "VXRTIR039",
            ),
            (
                lambda document: document["nodes"][0].update(
                    {"source_name": "identity.onnx"}
                ),
                "VXRTIR040",
            ),
        )
        for mutate, code in cases:
            with self.subTest(code=code):
                document = copy.deepcopy(dynamic_identity_document())
                mutate(document)
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(document, {})
                self.assertEqual(caught.exception.diagnostic.code, code)

    def test_bank_declarations_survive_import_and_reject_bad_dimensions(self):
        document = copy.deepcopy(dynamic_identity_document())
        document["banks"] = {"adapters.down": "batch"}
        # A declared bank passes the root-field allowlist and the dimension check.
        import_runtime_package(document, {})

        for bad in ({"adapters.down": "missing"}, {"adapters.down": 3}, []):
            with self.subTest(bad=bad):
                broken = copy.deepcopy(dynamic_identity_document())
                broken["banks"] = bad
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(broken, {})
                self.assertEqual(caught.exception.diagnostic.code, "VXRTIR041")

    def test_node_outputs_are_assertions_not_trusted_shape_inference(self):
        document = dynamic_identity_document()
        document["nodes"][0]["outputs"]["out"]["shape"] = ["batch", 5]
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR036")

    def test_dynamic_resize_uses_shared_bounded_domain_output_assertion(self):
        graph = import_runtime_package(dynamic_resize_document(), {})
        self.assertEqual(
            graph.tensors["resized"].shape,
            ("batch", "resized_height", "resized_width", 4),
        )

        document = dynamic_resize_document()
        document["nodes"][0]["outputs"]["out"]["shape"][-1] = 5
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR035")
        self.assertIn("bounded-domain inference", caught.exception.diagnostic.message)

    def test_former_target_fallback_routes_through_shared_domain_proofs(self):
        proved: list[str] = []
        original = OperatorShapeContract.prove_domain

        def record_proof(contract, request):
            proved.append(contract.operator)
            return original(contract, request)

        with patch.object(OperatorShapeContract, "prove_domain", record_proof):
            graph = import_runtime_package(dynamic_reshape_expand_document(), {})

        self.assertEqual(proved, ["Reshape", "Expand"])
        self.assertEqual(graph.tensors["expanded"].shape, ("batch", 3, 4))

    def test_symbolic_deferred_operator_has_no_output_assertion_fallback(self):
        document = dynamic_identity_document()
        document["inputs"]["x"]["dtype"] = "int8"
        node = document["nodes"][0]
        node["id"] = "requantize"
        node["opType"] = "RequantizeLinear"
        node["outputs"]["out"]["dtype"] = "int8"

        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, {})

        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR035")
        self.assertIn(
            "canonical bounded-domain inference",
            caught.exception.diagnostic.message,
        )
        self.assertIn("requantize", caught.exception.diagnostic.message)

    def test_concrete_only_pass_is_transactional_and_requires_profile(self):
        graph = import_runtime_package(dynamic_identity_document(), {})
        before = graph.fingerprint()
        pipeline = VerifiedPipeline((MutateThenFailConcretePass(),))
        with self.assertRaises(ExporterError) as missing:
            pipeline.run(graph)
        self.assertEqual(missing.exception.diagnostic.code, "VXOPT009")
        self.assertEqual(graph.fingerprint(), before)

        profiled = VerifiedPipeline(
            (MutateThenFailConcretePass(),),
            shape_profile={"batch": 2},
        )
        with self.assertRaises(ExporterError) as failed:
            profiled.run(graph)
        self.assertEqual(failed.exception.diagnostic.code, "VXOPT005")
        self.assertEqual(graph.fingerprint(), before)
        self.assertEqual(graph.tensors["x"].shape, ("batch", 4))

    def test_static_concrete_profile_is_validated_without_mutating_graph(self):
        graph = import_runtime_package(dynamic_identity_document(), {}).bind_shape_profile({
            "batch": 2,
        })
        before = graph.fingerprint()

        report = VerifiedPipeline(
            (NoOpConcretePass(),), shape_profile={},
        ).run(graph)
        self.assertEqual(report.total_changes, 0)
        self.assertEqual(graph.fingerprint(), before)

        with self.assertRaises(ExporterError) as caught:
            VerifiedPipeline(
                (NoOpConcretePass(),), shape_profile={"batch": 2},
            ).run(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR050")
        self.assertEqual(graph.fingerprint(), before)

    def test_constraint_refining_pass_preserves_symbol_set_and_cannot_widen(self):
        for malicious_pass in (
            WidenConstraintPass(),
            AddConstraintPass(),
            RefineAndMutatePublicInputPass(),
        ):
            with self.subTest(pass_name=malicious_pass.name):
                graph = import_runtime_package(dynamic_identity_document(), {})
                before = graph.fingerprint()
                with self.assertRaises(ExporterError) as caught:
                    VerifiedPipeline((malicious_pass,)).run(graph)
                self.assertEqual(caught.exception.diagnostic.code, "VXOPT010")
                self.assertEqual(graph.fingerprint(), before)

    def test_symbolic_preserving_pass_cannot_change_public_shape_contract(self):
        for malicious_pass in (
            MutatePreservingEnvironmentPass(),
            MutatePreservingPublicInputPass(),
        ):
            with self.subTest(pass_name=malicious_pass.name):
                graph = import_runtime_package(dynamic_identity_document(), {})
                before = graph.fingerprint()
                with self.assertRaises(ExporterError) as caught:
                    VerifiedPipeline((malicious_pass,)).run(graph)
                self.assertEqual(caught.exception.diagnostic.code, "VXOPT011")
                self.assertEqual(graph.fingerprint(), before)

    def test_retired_shape_system_field_is_rejected_as_unknown(self):
        document = dynamic_identity_document()
        document["shape_system"] = "volvox-bounded-shape/v1"
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR038")
        self.assertIn("unsupported field 'shape_system'", caught.exception.diagnostic.message)


if __name__ == "__main__":
    unittest.main()
