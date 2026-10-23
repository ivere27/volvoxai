"""Delegated package export preserves the current closed graph contract."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import load_file

from tools.export_safetensors import (
    _require_closed_dynamic_v1,
    export_onnx_model,
)
from tools.exporter.execution_contract import stateless_execution


class DelegatedExportContractTests(unittest.TestCase):
    def test_onnx_export_preserves_stateless_execution_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = helper.make_model(helper.make_graph(
                [helper.make_node("Gemm", ["input", "weight", "bias"], ["output"])],
                "linear",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
                [numpy_helper.from_array(np.ones((2, 3), np.float32), "weight"),
                 numpy_helper.from_array(np.zeros(3, np.float32), "bias")],
            ), opset_imports=[helper.make_opsetid("", 17)])
            source = root / "model.onnx"
            destination = root / "export" / "model.safetensors"
            onnx.save(model, source)

            graph = export_onnx_model(str(source), str(destination))

            self.assertEqual(graph["execution"], stateless_execution(
                graph["inputs"], graph["outputs"],
            ))
            self.assertEqual(graph, json.loads(
                (destination.parent / "graph.json").read_text(encoding="utf-8"),
            ))
            self.assertEqual(len(load_file(destination)), 2)

    def test_current_execution_field_does_not_allow_unknown_root_fields(self):
        graph = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": {"input": {"shape": [1], "dtype": "float32"}},
            "nodes": [],
            "outputs": ["input"],
            "execution": stateless_execution(["input"], ["input"]),
        }
        _require_closed_dynamic_v1(graph)
        for name in ("source", "metadata", "execution_typo"):
            with self.subTest(field=name):
                invalid = copy.deepcopy(graph)
                invalid[name] = {}
                with self.assertRaisesRegex(RuntimeError, "non-closed graph root"):
                    _require_closed_dynamic_v1(invalid)


if __name__ == "__main__":
    unittest.main()
