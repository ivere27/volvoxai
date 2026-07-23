from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.exporter.ir import OpAttribute
from tools.exporter.optimizer.typed_qdq_layout import RuntimeQDQMovementPass
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import ReferenceExecutor
from tools.exporter.runtime_ir import (
    export_runtime_package,
    import_runtime_package,
)


_ALL_MOVEMENTS = (
    ("Reshape", [1, 2, 3], None),
    ("Transpose", [1, 3, 2], {"perm": [0, 2, 1]}),
    ("Squeeze", [3, 2], None),
    ("Unsqueeze", [1, 3, 2], None),
    ("Identity", [1, 3, 2], None),
)


def _layout_package(
    *,
    movements=_ALL_MOVEMENTS,
    output_affine: str = "same",
    shared_dequantized: bool = False,
    extra_output: str | None = None,
    initializer_source: bool = False,
):
    tensors = {
        "scale": np.asarray([0.25], dtype=np.float32),
        "zero": np.asarray([-3], dtype=np.int8),
    }
    if output_affine == "different":
        tensors.update({
            "other_scale": np.asarray([0.5], dtype=np.float32),
            "other_zero": np.asarray([1], dtype=np.int8),
        })
    if initializer_source:
        tensors["x"] = np.asarray(
            [[-128, -9, -3], [0, 17, 127]], dtype=np.int8,
        )
        inputs = {}
    else:
        inputs = {"x": {"shape": [2, 3], "dtype": "int8"}}

    nodes = [{
        "id": "dq",
        "opType": "DequantizeLinear",
        "inputs": {"input": "x", "scale": "scale", "zero_point": "zero"},
        "outputs": {"out": "f0"},
        "outputs_shape": {"out": [2, 3]},
        "outputs_dtype": {"out": "float32"},
    }]
    graph_outputs = ["y"]
    if shared_dequantized:
        nodes.append({
            "id": "side",
            "opType": "Identity",
            "inputs": {"input": "f0"},
            "outputs": {"out": "side"},
            "outputs_shape": {"out": [2, 3]},
            "outputs_dtype": {"out": "float32"},
        })
        graph_outputs.append("side")

    current = "f0"
    current_shape = [2, 3]
    for index, (op_type, output_shape, params) in enumerate(movements, 1):
        output = f"f{index}"
        node = {
            "id": f"movement-{index}",
            "opType": op_type,
            "inputs": {"input": current},
            "outputs": {"out": output},
            "outputs_shape": {"out": list(output_shape)},
            "outputs_dtype": {"out": "float32"},
        }
        if params is not None:
            node["params"] = copy.deepcopy(params)
        nodes.append(node)
        current = output
        current_shape = list(output_shape)

    q_scale = "other_scale" if output_affine == "different" else "scale"
    q_zero = "other_zero" if output_affine == "different" else "zero"
    nodes.append({
        "id": "q",
        "opType": "QuantizeLinear",
        "inputs": {"input": current, "scale": q_scale, "zero_point": q_zero},
        "outputs": {"out": "y"},
        "outputs_shape": {"out": current_shape},
        "outputs_dtype": {"out": "int8"},
    })
    if extra_output is not None:
        graph_outputs.append(extra_output)

    document = {
        "format": "volvox-graph/v1",
        "inputs": inputs,
        "outputs": graph_outputs,
        "nodes": nodes,
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "x": {
                    "scheme": "per_tensor",
                    "scale_tensor": "scale",
                    "zero_point_tensor": "zero",
                },
                "y": {
                    "scheme": "per_tensor",
                    "scale_tensor": q_scale,
                    "zero_point_tensor": q_zero,
                },
            },
        },
    }
    return document, tensors


class RuntimeQDQMovementPassTests(unittest.TestCase):
    def test_all_supported_movements_reuse_one_affine_and_remove_boundaries(self):
        document, tensors = _layout_package()
        graph = import_runtime_package(document, tensors)
        source_affine = graph.tensors["x"].quantization

        report = VerifiedPipeline([RuntimeQDQMovementPass()]).run(graph)

        self.assertEqual(report.runs[0].changes, 1)
        self.assertIn("reused affine refs 'scale'/'zero'", report.runs[0].notes[0])
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            [item[0] for item in _ALL_MOVEMENTS],
        )
        self.assertEqual(graph.nodes[0].input_map(), {"input": "x"})
        self.assertEqual(graph.nodes[-1].output_map(), {"out": "y"})
        self.assertNotIn("f0", graph.tensors)
        self.assertNotIn(f"f{len(_ALL_MOVEMENTS)}", graph.tensors)
        for index in range(1, len(_ALL_MOVEMENTS)):
            tensor = graph.tensors[f"f{index}"]
            self.assertEqual(tensor.dtype, "int8")
            self.assertIs(tensor.quantization, source_affine)

        optimized, optimized_tensors = export_runtime_package(graph, tensors)
        self.assertEqual(
            set(optimized["quantization"]["tensors"]),
            {"x", "f1", "f2", "f3", "f4", "y"},
        )
        for descriptor in optimized["quantization"]["tensors"].values():
            self.assertEqual(descriptor["scheme"], "per_tensor")
            self.assertEqual(descriptor["scale_tensor"], "scale")
            self.assertEqual(descriptor["zero_point_tensor"], "zero")
        np.testing.assert_array_equal(optimized_tensors["scale"], tensors["scale"])
        np.testing.assert_array_equal(optimized_tensors["zero"], tensors["zero"])

    def test_reshape_transpose_identity_is_byte_exact_against_reference(self):
        movements = (
            ("Reshape", [1, 2, 3], None),
            ("Transpose", [1, 3, 2], {"perm": [0, 2, 1]}),
            ("Identity", [1, 3, 2], None),
        )
        document, tensors = _layout_package(movements=movements)
        before = import_runtime_package(document, tensors)
        after = import_runtime_package(document, tensors)
        values = np.asarray(
            [[-128, -9, -3], [0, 17, 127]], dtype=np.int8,
        )

        expected = ReferenceExecutor(before, tensors).run({"x": values}).outputs["y"]
        VerifiedPipeline([RuntimeQDQMovementPass()]).run(after)
        actual = ReferenceExecutor(after, tensors).run({"x": values}).outputs["y"]

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual,
            values.reshape(1, 2, 3).transpose(0, 2, 1),
        )

    def test_byte_initializer_payload_is_never_decoded_or_rewritten(self):
        document, tensors = _layout_package(initializer_source=True)
        original_payload = tensors["x"].tobytes()
        graph = import_runtime_package(document, tensors)

        VerifiedPipeline([RuntimeQDQMovementPass()]).run(graph)
        _, optimized_tensors = export_runtime_package(graph, tensors)

        self.assertEqual(tensors["x"].tobytes(), original_payload)
        self.assertEqual(optimized_tensors["x"].tobytes(), original_payload)
        self.assertIs(optimized_tensors["x"], tensors["x"])

    def test_different_affines_refuse_without_mutation(self):
        document, tensors = _layout_package(output_affine="different")
        graph = import_runtime_package(document, tensors)
        before = graph.fingerprint()

        result = RuntimeQDQMovementPass().run(graph)

        self.assertEqual(result.changes, 0)
        self.assertEqual(graph.fingerprint(), before)

    def test_shared_float_consumer_refuses_without_mutation(self):
        document, tensors = _layout_package(shared_dequantized=True)
        graph = import_runtime_package(document, tensors)
        before = graph.fingerprint()

        result = RuntimeQDQMovementPass().run(graph)

        self.assertEqual(result.changes, 0)
        self.assertEqual(graph.fingerprint(), before)

    def test_intermediate_graph_output_refuses_but_final_output_is_preserved(self):
        document, tensors = _layout_package(extra_output="f2")
        graph = import_runtime_package(document, tensors)
        before = graph.fingerprint()

        refused = RuntimeQDQMovementPass().run(graph)

        self.assertEqual(refused.changes, 0)
        self.assertEqual(graph.fingerprint(), before)

        document, tensors = _layout_package()
        graph = import_runtime_package(document, tensors)
        changed = VerifiedPipeline([RuntimeQDQMovementPass()]).run(graph)
        self.assertEqual(changed.runs[0].changes, 1)
        self.assertEqual(graph.outputs, ["y"])
        self.assertTrue(graph.tensors["y"].public_output)
        self.assertEqual(graph.nodes[-1].output_map(), {"out": "y"})

    def test_malformed_shape_and_permutation_refuse_without_partial_rewrite(self):
        document, tensors = _layout_package()
        graph = import_runtime_package(document, tensors)
        graph.tensors["f1"].shape = (1, 2, 4)
        before = graph.fingerprint()
        self.assertEqual(RuntimeQDQMovementPass().run(graph).changes, 0)
        self.assertEqual(graph.fingerprint(), before)

        graph = import_runtime_package(document, tensors)
        transpose = next(node for node in graph.nodes if node.op_type == "Transpose")
        transpose.attributes = (
            OpAttribute("params", "volvox.params", {"perm": [0, 0, 1]}),
        )
        before = graph.fingerprint()
        self.assertEqual(RuntimeQDQMovementPass().run(graph).changes, 0)
        self.assertEqual(graph.fingerprint(), before)

        graph = import_runtime_package(document, tensors)
        reshape = next(node for node in graph.nodes if node.op_type == "Reshape")
        reshape.attributes = (
            OpAttribute("unexpected", "volvox.params", {}),
        )
        before = graph.fingerprint()
        self.assertEqual(RuntimeQDQMovementPass().run(graph).changes, 0)
        self.assertEqual(graph.fingerprint(), before)

        graph = import_runtime_package(document, tensors)
        graph.tensors["f2"].shape = (3, 1, 2)
        before = graph.fingerprint()
        self.assertEqual(RuntimeQDQMovementPass().run(graph).changes, 0)
        self.assertEqual(graph.fingerprint(), before)


if __name__ == "__main__":
    unittest.main()
