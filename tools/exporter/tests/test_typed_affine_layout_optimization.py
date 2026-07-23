from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.exporter.generated.optimizer_registry import PASSES
from tools.exporter.optimizer.typed_affine_canonicalization import (
    RuntimeAffineReferenceCanonicalizationPass,
)
from tools.exporter.optimizer.typed_qdq_layout import (
    RuntimePointwiseTransposeHoistPass,
    RuntimeQDQTransposeCancellationPass,
)
from tools.exporter.optimizer.typed_pipeline import optimize_runtime_package
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import ReferenceExecutor
from tools.exporter.runtime_ir import export_runtime_package, import_runtime_package


def _affine(scale: str, zero: str) -> dict[str, str]:
    return {
        "scheme": "per_tensor",
        "scale_tensor": scale,
        "zero_point_tensor": zero,
    }


def _duplicate_affines(*, different: bool = False):
    return {
        "a_scale": np.asarray([0.125], dtype=np.float32),
        "a_zero": np.asarray([-3], dtype=np.int8),
        "b_scale": np.asarray([0.25 if different else 0.125], dtype=np.float32),
        "b_zero": np.asarray([-3], dtype=np.int8),
    }


def _transpose_dq_transpose_package(*, different: bool = False):
    tensors = _duplicate_affines(different=different)
    document = {
        "format": "volvox-graph/v1",
        "inputs": {"x": {"shape": [1, 2, 3, 4], "dtype": "int8"}},
        "nodes": [
            {
                "id": "to-nchw",
                "opType": "Transpose",
                "inputs": {"input": "x"},
                "outputs": {"out": "xt"},
                "outputs_shape": {"out": [1, 4, 2, 3]},
                "outputs_dtype": {"out": "int8"},
                "params": {"perm": [0, 3, 1, 2]},
            },
            {
                "id": "dq",
                "opType": "DequantizeLinear",
                "inputs": {
                    "input": "xt", "scale": "b_scale", "zero_point": "b_zero",
                },
                "outputs": {"out": "f"},
                "outputs_shape": {"out": [1, 4, 2, 3]},
                "outputs_dtype": {"out": "float32"},
            },
            {
                "id": "to-nhwc",
                "opType": "Transpose",
                "inputs": {"input": "f"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [1, 2, 3, 4]},
                "outputs_dtype": {"out": "float32"},
                "params": {"perm": [0, 2, 3, 1]},
            },
        ],
        "outputs": ["y"],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "x": _affine("a_scale", "a_zero"),
                "xt": _affine("b_scale", "b_zero"),
            },
        },
    }
    return document, tensors


def _pointwise_transpose_package():
    tensors = _duplicate_affines()
    document = {
        "format": "volvox-graph/v1",
        "inputs": {"x": {"shape": [1, 2, 3, 4], "dtype": "float32"}},
        "nodes": [
            {
                "id": "to-nchw", "opType": "Transpose",
                "inputs": {"input": "x"}, "outputs": {"out": "xt"},
                "outputs_shape": {"out": [1, 4, 2, 3]},
                "outputs_dtype": {"out": "float32"},
                "params": {"perm": [0, 3, 1, 2]},
            },
            {
                "id": "sigmoid", "opType": "Sigmoid",
                "inputs": {"input": "xt"}, "outputs": {"out": "sig"},
                "outputs_shape": {"out": [1, 4, 2, 3]},
                "outputs_dtype": {"out": "float32"},
            },
            {
                "id": "multiply", "opType": "Mul",
                "inputs": {"a": "xt", "b": "sig"},
                "outputs": {"out": "silu"},
                "outputs_shape": {"out": [1, 4, 2, 3]},
                "outputs_dtype": {"out": "float32"},
            },
            {
                "id": "quantize", "opType": "QuantizeLinear",
                "inputs": {
                    "input": "silu", "scale": "b_scale", "zero_point": "b_zero",
                },
                "outputs": {"out": "q"},
                "outputs_shape": {"out": [1, 4, 2, 3]},
                "outputs_dtype": {"out": "int8"},
            },
            {
                "id": "to-nhwc", "opType": "Transpose",
                "inputs": {"input": "q"}, "outputs": {"out": "y"},
                "outputs_shape": {"out": [1, 2, 3, 4]},
                "outputs_dtype": {"out": "int8"},
                "params": {"perm": [0, 2, 3, 1]},
            },
        ],
        "outputs": ["y"],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "q": _affine("b_scale", "b_zero"),
                "y": _affine("a_scale", "a_zero"),
            },
        },
    }
    return document, tensors


class RuntimeAffineLayoutOptimizationTests(unittest.TestCase):
    def test_default_exact_mode_preserves_abi_affines_and_results_while_simplifying(self):
        document, tensors = _transpose_dq_transpose_package()
        source_payloads = {
            name: (value.dtype.str, value.shape, value.tobytes())
            for name, value in tensors.items()
        }
        values = (np.arange(24, dtype=np.int16) - 12).astype(np.int8).reshape(
            1, 2, 3, 4,
        )
        expected = ReferenceExecutor(
            import_runtime_package(copy.deepcopy(document), tensors), tensors,
        ).run({"x": values}).outputs["y"]

        optimized, optimized_tensors, report = optimize_runtime_package(
            document, tensors,
        )
        actual = ReferenceExecutor(
            import_runtime_package(optimized, optimized_tensors),
            optimized_tensors,
        ).run({"x": values}).outputs["y"]

        self.assertEqual(optimized["inputs"], document["inputs"])
        self.assertEqual(optimized["outputs"], document["outputs"])
        self.assertEqual(len(document["nodes"]), 3)
        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["DequantizeLinear"],
        )
        self.assertEqual(report.metadata.selection_features, ())
        self.assertTrue(all(
            PASSES[pass_id]["semantics"] == "exact"
            for pass_id in report.metadata.pass_ids
        ))
        self.assertGreater(report.total_changes, 0)
        np.testing.assert_array_equal(actual, expected)

        # The duplicate affine names may be interned and then pruned, but the
        # canonical payload is one of the producer's byte-identical values.
        self.assertEqual(set(optimized_tensors), {"a_scale", "a_zero"})
        for name, value in optimized_tensors.items():
            self.assertEqual(
                (value.dtype.str, value.shape, value.tobytes()),
                source_payloads[name],
            )
        for name, value in tensors.items():
            self.assertEqual(
                (value.dtype.str, value.shape, value.tobytes()),
                source_payloads[name],
            )

    def test_default_pipeline_runs_layout_optimization_and_can_defer_it(self):
        document, tensors = _transpose_dq_transpose_package()

        optimized, _, _ = optimize_runtime_package(document, tensors)
        deferred, _, _ = optimize_runtime_package(
            document,
            tensors,
            enable_static_qdq_layout_optimization=False,
        )

        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["DequantizeLinear"],
        )
        self.assertEqual(
            [node["opType"] for node in deferred["nodes"]],
            ["Transpose", "DequantizeLinear", "Transpose"],
        )

    def test_exact_duplicate_affines_are_interned_without_payload_changes(self):
        document, tensors = _transpose_dq_transpose_package()
        graph = import_runtime_package(document, tensors)
        original = {name: value.tobytes() for name, value in tensors.items()}

        report = VerifiedPipeline([
            RuntimeAffineReferenceCanonicalizationPass(tensors),
        ]).run(graph)

        self.assertGreater(report.total_changes, 0)
        self.assertEqual(graph.tensors["x"].quantization.scale, "a_scale")
        self.assertEqual(graph.tensors["xt"].quantization.scale, "a_scale")
        self.assertEqual(graph.nodes[1].input_map()["scale"], "a_scale")
        _, optimized_tensors = export_runtime_package(graph, tensors)
        for name, payload in original.items():
            self.assertEqual(tensors[name].tobytes(), payload)
            self.assertEqual(optimized_tensors[name].tobytes(), payload)

    def test_transpose_dq_inverse_transpose_cancels_byte_exactly(self):
        document, tensors = _transpose_dq_transpose_package()
        before = import_runtime_package(copy.deepcopy(document), tensors)
        after = import_runtime_package(copy.deepcopy(document), tensors)
        values = (np.arange(24, dtype=np.int16) - 12).astype(np.int8).reshape(
            1, 2, 3, 4,
        )
        expected = ReferenceExecutor(before, tensors).run({"x": values}).outputs["y"]

        report = VerifiedPipeline([
            RuntimeAffineReferenceCanonicalizationPass(tensors),
            RuntimeQDQTransposeCancellationPass(),
        ]).run(after)
        actual = ReferenceExecutor(after, tensors).run({"x": values}).outputs["y"]

        self.assertEqual(
            [node.op_type for node in after.nodes], ["DequantizeLinear"],
        )
        self.assertEqual(after.nodes[0].input_map()["input"], "x")
        self.assertEqual(after.nodes[0].output_map()["out"], "y")
        self.assertEqual(report.runs[1].changes, 1)
        np.testing.assert_array_equal(actual, expected)

    def test_different_affine_payload_refuses_cancellation(self):
        document, tensors = _transpose_dq_transpose_package()
        graph = import_runtime_package(document, tensors)
        tensors["b_scale"] = np.asarray([0.25], dtype=np.float32)
        report = VerifiedPipeline([
            RuntimeAffineReferenceCanonicalizationPass(tensors),
            RuntimeQDQTransposeCancellationPass(),
        ]).run(graph)

        self.assertEqual([node.op_type for node in graph.nodes], [
            "Transpose", "DequantizeLinear", "Transpose",
        ])
        self.assertEqual(report.runs[1].changes, 0)

    def test_pointwise_island_hoists_across_inverse_transposes_exactly(self):
        document, tensors = _pointwise_transpose_package()
        before = import_runtime_package(copy.deepcopy(document), tensors)
        after = import_runtime_package(copy.deepcopy(document), tensors)
        values = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(1, 2, 3, 4)
        expected = ReferenceExecutor(before, tensors).run({"x": values}).outputs["y"]

        report = VerifiedPipeline([
            RuntimeAffineReferenceCanonicalizationPass(tensors),
            RuntimePointwiseTransposeHoistPass(),
        ]).run(after)
        actual = ReferenceExecutor(after, tensors).run({"x": values}).outputs["y"]

        self.assertEqual(
            [node.op_type for node in after.nodes],
            ["Sigmoid", "Mul", "QuantizeLinear"],
        )
        self.assertEqual(after.tensors["sig"].shape, (1, 2, 3, 4))
        self.assertEqual(after.tensors["silu"].shape, (1, 2, 3, 4))
        self.assertEqual(after.nodes[-1].output_map(), {"out": "y"})
        self.assertEqual(report.runs[1].changes, 1)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
