"""ONNX LSTM lowering accepts only the implemented recurrent semantics."""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tools.export_safetensors import export_onnx_model
from tools.exporter.errors import ExporterError
from tools.lstm_fixtures import lstm_model


class LstmExportContractTests(unittest.TestCase):
    def model(self):
        return lstm_model(np.zeros((5, 2, 3), np.float32), np.zeros((2, 16, 3), np.float32),
                          np.zeros((2, 16, 4), np.float32), direction=2)

    def export(self, model):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx.save(model, root / "source.onnx")
            return export_onnx_model(str(root / "source.onnx"), str(root / "model.safetensors"))

    def attribute(self, model, name, value):
        attributes = model.graph.node[0].attribute
        for index in range(len(attributes) - 1, -1, -1):
            if attributes[index].name == name:
                del attributes[index]
        attributes.append(helper.make_attribute(name, value))

    def test_default_activations_full_lengths_and_unused_states(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                model = self.model()
                model.graph.node[0].output.extend(["yh", "yc"])
                if explicit:
                    for name, value in (("layout", 0), ("input_forget", 0),
                                        ("activations", ["Sigmoid", "Tanh", "Tanh"] * 2)):
                        self.attribute(model, name, value)
                    model.graph.node[0].input[4] = "lengths"
                    model.graph.initializer.append(numpy_helper.from_array(np.array([5, 5], np.int32), "lengths"))
                graph = self.export(model)
                self.assertEqual(len(graph["nodes"]), 1)
                node = graph["nodes"][0]
                self.assertEqual((node["opType"], node["params"]), ("LSTM", {"direction": 2}))
                self.assertEqual(node["outputs"]["out"]["shape"], [5, 2, 2, 4])
                self.assertNotIn("sequence_lens", node["inputs"])

    def test_unsupported_recurrence_attributes_are_refused(self):
        for name, value, message in (("layout", 1, "time-major"), ("input_forget", 1, "separate input/forget"),
            ("clip", 1.0, "clipping"), ("activations", ["Relu", "Tanh", "Tanh"] * 2, "default activations"),
            ("activation_alpha", [0.0] * 6, "activation parameters"),
            ("activation_beta", [0.0] * 6, "activation parameters"), ("hidden_size", 3, "hidden_size")):
            with self.subTest(name=name):
                model = self.model()
                self.attribute(model, name, value)
                if name == "hidden_size":
                    model.graph.output[0].type.tensor_type.shape.dim[3].dim_value = value
                with self.assertRaisesRegex(ExporterError, message):
                    self.export(model)

    def test_observable_states_peepholes_and_partial_lengths_are_refused(self):
        for mode, message in (("state-output", "final-state"), ("state-consumer", "final-state"),
                              ("peephole", "peephole"), ("short-lengths", "full sequence"),
                              ("dynamic-lengths", "full sequence")):
            with self.subTest(mode=mode):
                model = self.model()
                node = model.graph.node[0]
                if mode.startswith("state"):
                    node.output.extend(["yh", "yc"])
                    output = "yh"
                    if mode == "state-consumer":
                        output = "last"
                        model.graph.node.append(helper.make_node("Identity", ["yh"], [output]))
                    model.graph.output.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, [2, 2, 4]))
                elif mode == "peephole":
                    node.input.append("peephole")
                    model.graph.initializer.append(numpy_helper.from_array(np.zeros((2, 12), np.float32), "peephole"))
                else:
                    node.input[4] = "lengths"
                    if mode == "short-lengths":
                        model.graph.initializer.append(numpy_helper.from_array(np.array([5, 3], np.int32), "lengths"))
                    else:
                        model.graph.input.append(helper.make_tensor_value_info("lengths", TensorProto.INT32, [2]))
                with self.assertRaisesRegex(ExporterError, message):
                    self.export(model)


if __name__ == "__main__":
    unittest.main()
