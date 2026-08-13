from __future__ import annotations

import unittest

import numpy as np

from tools.exporter.optimizer.typed_common_subexpression import (
    RuntimeCommonSubexpressionEliminationPass,
)
from tools.exporter.optimizer.typed_pipeline import optimize_runtime_package
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import execute_reference
from tools.exporter.runtime_ir import import_runtime_package


def _node(
    name: str,
    op_type: str,
    inputs: dict[str, str],
    output: str,
    shape: list[int],
    *,
    params: dict | None = None,
) -> dict:
    return {
        "id": name,
        "opType": op_type,
        "inputs": inputs,
        "outputs": {
            "out": {"tensor": output, "shape": shape, "dtype": "float32"}
        },
        "params": {} if params is None else params,
    }


def _package() -> tuple[dict, dict[str, np.ndarray]]:
    shape = [3, 2]
    document = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [2], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [
            _node("reshape-a", "Reshape", {"input": "x"}, "ra", [1, 2],
                  params={"shape": [1, 2]}),
            _node("reshape-b", "Reshape", {"input": "x"}, "rb", [1, 2],
                  params={"shape": [1, 2]}),
            _node("broadcast-a", "Expand", {"input": "ra"}, "ba", shape,
                  params={"shape": shape}),
            _node("broadcast-b", "Expand", {"input": "rb"}, "bb", shape,
                  params={"shape": shape}),
            _node("zero-a", "Expand", {"input": "positive-zero-a"}, "za", shape,
                  params={"shape": shape}),
            _node("zero-b", "Expand", {"input": "positive-zero-b"}, "zb", shape,
                  params={"shape": shape}),
            _node("negative-zero", "Expand", {"input": "negative-zero"}, "zn", shape,
                  params={"shape": shape}),
            _node("sum-a", "Add", {"a": "ba", "b": "za"}, "sa", shape),
            _node("sum-b", "Add", {"a": "bb", "b": "zb"}, "sb", shape),
            _node("sum-negative", "Add", {"a": "sa", "b": "zn"}, "sn", shape),
            _node("result", "Add", {"a": "sb", "b": "sn"}, "y", shape),
        ],
    }
    tensors = {
        "positive-zero-a": np.asarray([0.0], dtype=np.float32),
        "positive-zero-b": np.asarray([0.0], dtype=np.float32),
        "negative-zero": np.asarray([-0.0], dtype=np.float32),
    }
    return document, tensors


class RuntimeCommonSubexpressionEliminationPassTests(unittest.TestCase):
    def test_shares_only_exact_shape_and_byte_equivalent_computations(self):
        document, tensors = _package()
        before = import_runtime_package(document, tensors)
        after = import_runtime_package(document, tensors)
        inputs = {"x": np.asarray([1.25, -2.5], dtype=np.float32)}
        expected = execute_reference(before, tensors, inputs).outputs["y"]

        report = VerifiedPipeline((
            RuntimeCommonSubexpressionEliminationPass(tensors),
        ), shape_profile={}).run(after)
        actual = execute_reference(after, tensors, inputs).outputs["y"]

        self.assertEqual(report.total_changes, 3)
        self.assertEqual(
            [node.name for node in after.nodes if node.op_type == "Reshape"],
            ["reshape-a"],
        )
        self.assertEqual(
            [node.name for node in after.nodes if node.op_type == "Expand"],
            ["broadcast-a", "zero-a", "negative-zero"],
        )
        self.assertEqual(after.nodes[-1].input_map(), {"a": "sb", "b": "sn"})
        self.assertEqual(after.nodes[-3].input_map(), {"a": "ba", "b": "za"})
        np.testing.assert_array_equal(actual, expected)

    def test_pipeline_feature_is_explicit_and_default_remains_unchanged(self):
        document, tensors = _package()
        default, _, default_report = optimize_runtime_package(
            document, tensors, shape_profile={},
        )
        selected, _, selected_report = optimize_runtime_package(
            document,
            tensors,
            shape_profile={},
            enable_exact_common_subexpression_elimination=True,
        )

        self.assertEqual(len(default["nodes"]), 11)
        self.assertEqual(len(selected["nodes"]), 8)
        self.assertNotIn(
            "runtime-common-subexpression",
            default_report.metadata.pass_ids,
        )
        self.assertIn(
            "runtime-common-subexpression",
            selected_report.metadata.pass_ids,
        )
        self.assertEqual(
            selected_report.metadata.selection_features,
            ("exact-common-subexpression",),
        )


if __name__ == "__main__":
    unittest.main()
