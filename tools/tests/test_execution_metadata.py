"""Execution contracts survive RuntimeIR and explicit public ABI specialization."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.exporter.execution_contract import stateless_execution, tensor_feedback_execution
from tools.exporter.runtime_ir import import_runtime_package, export_runtime_package


class ExecutionMetadataTest(unittest.TestCase):
    def graph(self):
        return {"format": "volvox-graph/v1", "dimensions": {},
            "inputs": {"x": {"shape": [1], "dtype": "float32"}},
            "nodes": [{"id": "copy", "opType": "Identity", "inputs": {"input": "x"},
                "outputs": {"out": {"tensor": "y", "shape": [1], "dtype": "float32"}}, "params": {}}],
            "outputs": ["y"]}

    def test_stateless_authoring_uses_published_input_and_output_names(self):
        graph = self.graph()
        graph["execution"] = stateless_execution(graph["inputs"], graph["outputs"])
        imported = import_runtime_package(graph, {})
        document, _ = export_runtime_package(imported, {})
        self.assertEqual(document["execution"], graph["execution"])
        imported.metadata["execution"]["entrypoints"][0]["outputs"] = ["old_output"]
        document, _ = export_runtime_package(imported, {})
        self.assertEqual(document["execution"], graph["execution"])

    def test_feedback_contract_is_preserved_without_aliasing(self):
        graph = self.graph()
        graph["inputs"]["position"] = {"shape": [1], "dtype": "int32"}
        graph["execution"] = tensor_feedback_execution(graph["inputs"], graph["outputs"],
            feedback={"x": "y"}, position_input="position")
        expected = deepcopy(graph["execution"])
        imported = import_runtime_package(graph, {})
        graph["execution"]["tensor_feedback"]["position_input"] = "changed"
        document, _ = export_runtime_package(imported, {})
        self.assertEqual(document["execution"], expected)
        document["execution"]["tensor_feedback"]["bindings"].clear()
        again, _ = export_runtime_package(imported, {})
        self.assertEqual(again["execution"], expected)


if __name__ == "__main__":
    unittest.main()
