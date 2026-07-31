from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.exporter.ir import AffineQuantization, OpAttribute
from tools.exporter.optimizer.typed_singleton_transpose import (
    RuntimeSingletonTransposePass,
)
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import ReferenceExecutor
from tools.exporter.runtime_ir import export_runtime_package, import_runtime_package


def _transpose_package(
    *,
    shape=(2, 1, 3, 1, 4),
    permutation=(1, 0, 3, 2, 4),
    dtype="float32",
):
    output_shape = tuple(shape[axis] for axis in permutation)
    node = {
        "id": "move-singletons",
        "opType": "Transpose",
        "inputs": {"input": "x"},
        "outputs": {"out": "y"},
        "outputs_shape": {"out": list(output_shape)},
        "outputs_dtype": {"out": dtype},
    }
    node["params"] = {"perm": list(permutation)}
    document = {
        "format": "volvox-graph/v1",
        "inputs": {"x": {"shape": list(shape), "dtype": dtype}},
        "outputs": ["y"],
        "nodes": [node],
    }
    tensors = {}
    if dtype == "int8":
        tensors = {
            "scale": np.asarray([0.125], dtype=np.float32),
            "zero": np.asarray([-3], dtype=np.int8),
        }
        descriptor = {
            "scheme": "per_tensor",
            "scale_tensor": "scale",
            "zero_point_tensor": "zero",
        }
        document["quantization"] = {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {"x": dict(descriptor), "y": dict(descriptor)},
        }
    return document, tensors


class RuntimeSingletonTransposePassTests(unittest.TestCase):
    def test_singleton_axis_permutation_is_exact_for_float_and_int8(self):
        for dtype in ("float32", "int8"):
            with self.subTest(dtype=dtype):
                document, tensors = _transpose_package(dtype=dtype)
                before = import_runtime_package(document, tensors)
                after = import_runtime_package(document, tensors)
                values = np.arange(24, dtype=np.float32).reshape(2, 1, 3, 1, 4)
                if dtype == "int8":
                    values = (values.astype(np.int16) - 12).astype(np.int8)

                expected = ReferenceExecutor(before, tensors).run(
                    {"x": values},
                ).outputs["y"]
                report = VerifiedPipeline([
                    RuntimeSingletonTransposePass(),
                ]).run(after)
                actual = ReferenceExecutor(after, tensors).run(
                    {"x": values},
                ).outputs["y"]

                self.assertEqual(report.total_changes, 1)
                self.assertEqual(after.nodes[0].op_type, "Reshape")
                self.assertEqual(after.nodes[0].attributes, ())
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(
                    actual,
                    values.reshape(1, 2, 1, 3, 4),
                )

    def test_non_singleton_reordering_is_refused_without_mutation(self):
        document, tensors = _transpose_package(
            # The swapped non-singleton axes intentionally have equal extents:
            # matching input/output shapes alone is not an element-order proof.
            shape=(2, 1, 2, 1, 4),
            permutation=(1, 2, 3, 0, 4),
        )
        graph = import_runtime_package(document, tensors)
        before = graph.fingerprint()

        result = RuntimeSingletonTransposePass().run(graph)

        self.assertEqual(result.changes, 0)
        self.assertEqual(graph.fingerprint(), before)
        self.assertEqual(graph.nodes[0].op_type, "Transpose")

    def test_malformed_or_semantically_incompatible_nodes_are_refused(self):
        document, tensors = _transpose_package(dtype="int8")

        def extra_param(graph):
            graph.nodes[0].attributes = (
                OpAttribute(
                    "params", "volvox.params",
                    {"perm": [1, 0, 3, 2, 4], "unexpected": True},
                ),
            )

        def malformed_permutation(graph):
            graph.nodes[0].attributes = (
                OpAttribute(
                    "params", "volvox.params", {"perm": [1, 0, 3, 2, 2]},
                ),
            )

        def wrong_output_shape(graph):
            graph.tensors["y"].shape = (1, 2, 3, 1, 4)

        def different_affine(graph):
            graph.tensors["y"].quantization = AffineQuantization(
                scheme="per_tensor",
                scale="other-scale",
                zero_point="other-zero",
            )

        def noncanonical_input_port(graph):
            node = graph.nodes[0]
            node.inputs = tuple(
                type(port)("x", port.value, port.position)
                for port in node.inputs
            )

        for mutate in (
            extra_param,
            malformed_permutation,
            wrong_output_shape,
            different_affine,
            noncanonical_input_port,
        ):
            with self.subTest(mutate=mutate.__name__):
                graph = import_runtime_package(copy.deepcopy(document), tensors)
                mutate(graph)
                before = graph.fingerprint()

                result = RuntimeSingletonTransposePass().run(graph)

                self.assertEqual(result.changes, 0)
                self.assertEqual(graph.fingerprint(), before)

    def test_quantized_affines_and_payloads_are_not_changed(self):
        document, tensors = _transpose_package(dtype="int8")
        graph = import_runtime_package(document, tensors)
        source_affine = graph.tensors["x"].quantization
        output_affine = graph.tensors["y"].quantization
        original_scale = tensors["scale"].tobytes()
        original_zero = tensors["zero"].tobytes()

        VerifiedPipeline([RuntimeSingletonTransposePass()]).run(graph)
        optimized, optimized_tensors = export_runtime_package(graph, tensors)

        self.assertIs(graph.tensors["x"].quantization, source_affine)
        self.assertIs(graph.tensors["y"].quantization, output_affine)
        self.assertEqual(
            optimized["quantization"], document["quantization"],
        )
        self.assertEqual(optimized_tensors["scale"].tobytes(), original_scale)
        self.assertEqual(optimized_tensors["zero"].tobytes(), original_zero)
        self.assertIs(optimized_tensors["scale"], tensors["scale"])
        self.assertIs(optimized_tensors["zero"], tensors["zero"])


if __name__ == "__main__":
    unittest.main()
