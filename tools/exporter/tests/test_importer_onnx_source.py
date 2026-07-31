from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.exporter.errors import ExporterError
from tools.exporter.importers.onnx import import_onnx_source
from tools.exporter.ir import IRDialect


class OnnxSourceImporterTests(unittest.TestCase):
    def test_preserves_optional_ports_symbolic_types_versions_and_raw_attributes(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
        node = helper.make_node("Clip", ["x", "", "max"], ["y"],
                                name="clip", doc_string="source node")
        maximum = helper.make_tensor("max", TensorProto.FLOAT, [], [6.0])
        model = helper.make_model(
            helper.make_graph([node], "g", [x], [y], [maximum]),
            opset_imports=[helper.make_opsetid("", 13)],
        )
        graph = import_onnx_source(model)
        graph.verify(IRDialect.SOURCE)
        self.assertEqual(graph.opsets, {"": 13})
        self.assertEqual(graph.tensors["x"].shape, ("batch", 4))
        self.assertEqual([port.value for port in graph.nodes[0].inputs],
                         ["x", None, "max"])
        self.assertEqual(graph.nodes[0].version, 13)
        self.assertTrue(graph.nodes[0].metadata["onnx_node_proto"])
        self.assertTrue(graph.tensors["max"].metadata["onnx_tensor_proto"])

    def test_preserves_external_initializer_storage_reference(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        weight = numpy_helper.from_array(np.asarray([2.0], dtype=np.float32), "w")
        model = helper.make_model(
            helper.make_graph([helper.make_node("Add", ["x", "w"], ["y"])],
                              "g", [x], [y], [weight]),
            opset_imports=[helper.make_opsetid("", 13)],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            onnx.save_model(model, path, save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            location="model.data", size_threshold=0)
            graph = import_onnx_source(path)
        ref = graph.tensors["w"].data
        self.assertIsNotNone(ref)
        self.assertEqual(ref.source_uri, "model.data")
        self.assertEqual(ref.byte_offset, 0)
        self.assertEqual(ref.byte_length, 4)

    def test_nested_graph_implicit_capture_is_preserved(self):
        cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        branch_output = helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [1])
        then_graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["branch_y"])],
            "then", [], [branch_output])
        else_graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["branch_y"])],
            "else", [], [branch_output])
        model = helper.make_model(
            helper.make_graph([
                helper.make_node("If", ["cond"], ["y"],
                                 then_branch=then_graph, else_branch=else_graph),
            ], "g", [cond, x], [y]),
            opset_imports=[helper.make_opsetid("", 13)],
        )
        graph = import_onnx_source(model)
        self.assertEqual(len(graph.nodes[0].regions), 2)
        for region in graph.nodes[0].regions:
            self.assertEqual(region.captures, ["x"])
            region.verify(IRDialect.SOURCE)

    def test_checker_failure_is_reported_before_import(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        invalid = helper.make_model(
            helper.make_graph([helper.make_node("Add", ["missing", "x"], ["y"])],
                              "g", [x], [y]),
            opset_imports=[helper.make_opsetid("", 13)],
        )
        with self.assertRaises(ExporterError) as caught:
            import_onnx_source(invalid)
        self.assertEqual(caught.exception.diagnostic.code, "VXONNX002")


if __name__ == "__main__":
    unittest.main()
