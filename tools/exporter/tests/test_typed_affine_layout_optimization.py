from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.exporter.generated.optimizer_registry import PASSES
from tools.exporter.ir import AffineQuantization
from tools.exporter.optimizer.typed_affine_canonicalization import (
    RuntimeAffineReferenceCanonicalizationPass,
)
from tools.exporter.optimizer.typed_qdq_layout import (
    RuntimePointwiseTransposeHoistPass,
    RuntimeQDQTransposeCancellationPass,
)
from tools.exporter.optimizer.typed_pipeline import optimize_runtime_package
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import execute_reference
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
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 2, 3, 4], "dtype": "int8"}},
        "nodes": [
            {
                "id": "to-nchw",
                "opType": "Transpose",
                "inputs": {"input": "x"},
                "outputs": {"out": {
                    "tensor": "xt", "shape": [1, 4, 2, 3], "dtype": "int8",
                }},
                "params": {"perm": [0, 3, 1, 2]},
            },
            {
                "id": "dq",
                "opType": "DequantizeLinear",
                "inputs": {
                    "input": "xt", "scale": "b_scale", "zero_point": "b_zero",
                },
                "outputs": {"out": {
                    "tensor": "f", "shape": [1, 4, 2, 3], "dtype": "float32",
                }},
                "params": {},
            },
            {
                "id": "to-nhwc",
                "opType": "Transpose",
                "inputs": {"input": "f"},
                "outputs": {"out": {
                    "tensor": "y", "shape": [1, 2, 3, 4], "dtype": "float32",
                }},
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
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 2, 3, 4], "dtype": "float32"}},
        "nodes": [
            {
                "id": "to-nchw", "opType": "Transpose",
                "inputs": {"input": "x"}, "outputs": {"out": {
                    "tensor": "xt", "shape": [1, 4, 2, 3], "dtype": "float32",
                }},
                "params": {"perm": [0, 3, 1, 2]},
            },
            {
                "id": "sigmoid", "opType": "Sigmoid",
                "inputs": {"input": "xt"}, "outputs": {"out": {
                    "tensor": "sig", "shape": [1, 4, 2, 3], "dtype": "float32",
                }},
                "params": {},
            },
            {
                "id": "multiply", "opType": "Mul",
                "inputs": {"a": "xt", "b": "sig"},
                "outputs": {"out": {
                    "tensor": "silu", "shape": [1, 4, 2, 3], "dtype": "float32",
                }},
                "params": {},
            },
            {
                "id": "quantize", "opType": "QuantizeLinear",
                "inputs": {
                    "input": "silu", "scale": "b_scale", "zero_point": "b_zero",
                },
                "outputs": {"out": {
                    "tensor": "q", "shape": [1, 4, 2, 3], "dtype": "int8",
                }},
                "params": {},
            },
            {
                "id": "to-nhwc", "opType": "Transpose",
                "inputs": {"input": "q"}, "outputs": {"out": {
                    "tensor": "y", "shape": [1, 2, 3, 4], "dtype": "int8",
                }},
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


def _unary_quantized_transpose_package(
    *,
    second_perm: list[int] | None = None,
    shared_unary: bool = False,
    intermediate_output: str | None = None,
):
    tensors = _duplicate_affines()
    inverse = [0, 2, 3, 1] if second_perm is None else second_perm
    final_shape = [[1, 4, 2, 3][axis] for axis in inverse]
    nodes = [
        {
            "id": "to-nchw", "opType": "Transpose",
            "inputs": {"input": "x"}, "outputs": {"out": {
                "tensor": "xt", "shape": [1, 4, 2, 3], "dtype": "float32",
            }},
            "params": {"perm": [0, 3, 1, 2]},
        },
        {
            "id": "silu", "opType": "SiLU",
            "inputs": {"input": "xt"}, "outputs": {"out": {
                "tensor": "activated", "shape": [1, 4, 2, 3],
                "dtype": "float32",
            }},
            "params": {},
        },
        {
            "id": "quantize", "opType": "QuantizeLinear",
            "inputs": {
                "input": "activated", "scale": "b_scale",
                "zero_point": "b_zero",
            },
            "outputs": {"out": {
                "tensor": "q", "shape": [1, 4, 2, 3], "dtype": "int8",
            }},
            "params": {},
        },
        {
            "id": "from-nchw", "opType": "Transpose",
            "inputs": {"input": "q"}, "outputs": {"out": {
                "tensor": "y", "shape": final_shape, "dtype": "int8",
            }},
            "params": {"perm": inverse},
        },
    ]
    outputs = ["y"]
    if shared_unary:
        nodes.insert(2, {
            "id": "side-use", "opType": "Identity",
            "inputs": {"input": "activated"}, "outputs": {"out": {
                "tensor": "side", "shape": [1, 4, 2, 3],
                "dtype": "float32",
            }},
            "params": {},
        })
        outputs.append("side")
    if intermediate_output is not None:
        outputs.append(intermediate_output)
    document = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 2, 3, 4], "dtype": "float32"}},
        "nodes": nodes,
        "outputs": outputs,
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
        expected = execute_reference(
            import_runtime_package(copy.deepcopy(document), tensors),
            tensors,
            {"x": values},
        ).outputs["y"]

        optimized, optimized_tensors, report = optimize_runtime_package(
            document, tensors, shape_profile={},
        )
        actual = execute_reference(
            import_runtime_package(optimized, optimized_tensors),
            optimized_tensors,
            {"x": values},
        ).outputs["y"]

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

        optimized, _, _ = optimize_runtime_package(
            document, tensors, shape_profile={},
        )
        deferred, _, _ = optimize_runtime_package(
            document,
            tensors,
            enable_static_qdq_layout_optimization=False,
            shape_profile={},
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
        ], shape_profile={}).run(graph)

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
        expected = execute_reference(before, tensors, {"x": values}).outputs["y"]

        report = VerifiedPipeline([
            RuntimeAffineReferenceCanonicalizationPass(tensors),
            RuntimeQDQTransposeCancellationPass(),
        ], shape_profile={}).run(after)
        actual = execute_reference(after, tensors, {"x": values}).outputs["y"]

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
        ], shape_profile={}).run(graph)

        self.assertEqual([node.op_type for node in graph.nodes], [
            "Transpose", "DequantizeLinear", "Transpose",
        ])
        self.assertEqual(report.runs[1].changes, 0)

    def test_linear_silu_quantize_chain_cancels_inverse_transposes_exactly(self):
        document, tensors = _unary_quantized_transpose_package()
        before = import_runtime_package(copy.deepcopy(document), tensors)
        after = import_runtime_package(copy.deepcopy(document), tensors)
        values = np.linspace(-4.0, 4.0, 24, dtype=np.float32).reshape(
            1, 2, 3, 4,
        )
        expected = execute_reference(before, tensors, {"x": values}).outputs["y"]

        report = VerifiedPipeline([
            RuntimeAffineReferenceCanonicalizationPass(tensors),
            RuntimeQDQTransposeCancellationPass(),
        ], shape_profile={}).run(after)
        actual = execute_reference(after, tensors, {"x": values}).outputs["y"]

        self.assertEqual(
            [node.op_type for node in after.nodes], ["SiLU", "QuantizeLinear"],
        )
        self.assertEqual(after.nodes[0].input_map(), {"input": "x"})
        self.assertEqual(after.nodes[-1].output_map(), {"out": "y"})
        self.assertEqual(after.tensors["activated"].shape, (1, 2, 3, 4))
        self.assertEqual(after.tensors["activated"].dtype, "float32")
        self.assertEqual(after.tensors["y"].dtype, "int8")
        self.assertEqual(after.outputs, ["y"])
        self.assertTrue(after.tensors["y"].public_output)
        self.assertEqual(report.runs[1].changes, 1)
        np.testing.assert_array_equal(actual, expected)

    def test_linear_chain_refuses_noninverse_shared_public_and_per_axis_cases(self):
        cases = []

        document, tensors = _unary_quantized_transpose_package(
            second_perm=[0, 1, 2, 3],
        )
        cases.append(("non-inverse", document, tensors, None))

        document, tensors = _unary_quantized_transpose_package(shared_unary=True)
        cases.append(("multiple-consumer", document, tensors, None))

        document, tensors = _unary_quantized_transpose_package(
            intermediate_output="activated",
        )
        cases.append(("intermediate-public-output", document, tensors, None))

        document, tensors = _unary_quantized_transpose_package()
        cases.append(("per-axis-affine", document, tensors, "per-axis"))

        for label, document, tensors, mutation in cases:
            with self.subTest(case=label):
                graph = import_runtime_package(document, tensors)
                RuntimeAffineReferenceCanonicalizationPass(tensors).run(graph)
                if mutation == "per-axis":
                    canonical = graph.tensors["q"].quantization
                    assert canonical is not None
                    per_axis = AffineQuantization(
                        scheme="per_axis",
                        scale=canonical.scale,
                        zero_point=canonical.zero_point,
                        axis=0,
                    )
                    graph.tensors["q"].quantization = per_axis
                    graph.tensors["y"].quantization = per_axis
                before = graph.fingerprint()

                result = RuntimeQDQTransposeCancellationPass().run(graph)

                self.assertEqual(result.changes, 0)
                self.assertEqual(graph.fingerprint(), before)
                self.assertEqual(
                    [node.op_type for node in graph.nodes].count("Transpose"), 2,
                )

    def test_pointwise_island_hoists_across_inverse_transposes_exactly(self):
        document, tensors = _pointwise_transpose_package()
        before = import_runtime_package(copy.deepcopy(document), tensors)
        after = import_runtime_package(copy.deepcopy(document), tensors)
        values = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(1, 2, 3, 4)
        expected = execute_reference(before, tensors, {"x": values}).outputs["y"]

        report = VerifiedPipeline([
            RuntimeAffineReferenceCanonicalizationPass(tensors),
            RuntimePointwiseTransposeHoistPass(),
        ], shape_profile={}).run(after)
        actual = execute_reference(after, tensors, {"x": values}).outputs["y"]

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
