from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import flatbuffers

from tools.export_safetensors import export_tflite_model
from tools.exporter.capabilities import classify_package


def classify(nodes, inputs):
    return classify_package({"nodes": nodes, "inputs": inputs})


def _offset_vector(builder, values):
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependUOffsetTRelative(value)
    return builder.EndVector()


def _int32_vector(builder, values):
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependInt32(value)
    return builder.EndVector()


def _minimal_passthrough_tflite() -> bytes:
    """Build a schema-compatible one-tensor TFLite graph without TensorFlow."""

    builder = flatbuffers.Builder(256)
    tensor_name = builder.CreateString("input")
    graph_name = builder.CreateString("main")
    shape = _int32_vector(builder, [1])

    # Tensor { shape:[1], type:FLOAT32, buffer:0, name:"input" }.
    builder.StartObject(10)
    builder.PrependUOffsetTRelativeSlot(0, shape, 0)
    builder.PrependUint8Slot(1, 0, 0)
    builder.PrependUint32Slot(2, 0, 0)
    builder.PrependUOffsetTRelativeSlot(3, tensor_name, 0)
    tensor = builder.EndObject()
    tensors = _offset_vector(builder, [tensor])
    graph_inputs = _int32_vector(builder, [0])
    graph_outputs = _int32_vector(builder, [0])

    # SubGraph { tensors, inputs:[0], outputs:[0], operators:[] }.
    builder.StartObject(6)
    builder.PrependUOffsetTRelativeSlot(0, tensors, 0)
    builder.PrependUOffsetTRelativeSlot(1, graph_inputs, 0)
    builder.PrependUOffsetTRelativeSlot(2, graph_outputs, 0)
    builder.PrependUOffsetTRelativeSlot(4, graph_name, 0)
    subgraph = builder.EndObject()
    subgraphs = _offset_vector(builder, [subgraph])

    # Buffer { data:[] } and Model { version:3, subgraphs, buffers }.
    builder.StartObject(1)
    buffer = builder.EndObject()
    buffers = _offset_vector(builder, [buffer])
    builder.StartObject(8)
    builder.PrependUint32Slot(0, 3, 0)
    builder.PrependUOffsetTRelativeSlot(2, subgraphs, 0)
    builder.PrependUOffsetTRelativeSlot(4, buffers, 0)
    model = builder.EndObject()
    builder.Finish(model, file_identifier=b"TFL3")
    return bytes(builder.Output())


class TflitePackageClassificationTests(unittest.TestCase):
    def test_float_graph_is_not_stamped_w8a8(self):
        nodes = [{"opType": "Conv2D", "outputs_dtype": {"out": "float32"}}]
        self.assertEqual(
            classify(nodes, {"input0": {"dtype": "float32"}}),
            "fp32",
        )

    def test_complete_typed_graph_is_w8a8(self):
        nodes = [
            {"opType": "QConv2D", "outputs_dtype": {"out": "int8"}},
            {"opType": "MaxPool2D", "outputs_dtype": {"out": "int8"}},
            {"opType": "RequantizeLinear", "outputs_dtype": {"out": "int8"}},
        ]
        self.assertEqual(
            classify(nodes, {"input0": {"dtype": "int8"}}),
            "w8a8-v1",
        )

    def test_byte_graph_with_float_island_is_hybrid(self):
        nodes = [
            {"opType": "QConv2D", "outputs_dtype": {"out": "int8"}},
            {"opType": "Sigmoid", "outputs_dtype": {"out": "float32"}},
            {"opType": "QuantizeLinear", "outputs_dtype": {"out": "int8"}},
        ]
        self.assertEqual(
            classify(nodes, {"input0": {"dtype": "int8"}}),
            "hybrid",
        )

    def test_qdq_boundaries_without_byte_compute_are_not_w8a8(self):
        nodes = [
            {"opType": "DequantizeLinear", "outputs_dtype": {"out": "float32"}},
            {"opType": "QuantizeLinear", "outputs_dtype": {"out": "int8"}},
        ]
        self.assertEqual(
            classify(nodes, {"input0": {"dtype": "int8"}}),
            "hybrid",
        )

    def test_actual_tflite_export_assigns_package_class_before_serialization(self):
        with tempfile.TemporaryDirectory(prefix="volvoxai-tflite-class-") as temporary:
            directory = Path(temporary)
            source = directory / "model.tflite"
            output = directory / "model.safetensors"
            source.write_bytes(_minimal_passthrough_tflite())

            with contextlib.redirect_stdout(io.StringIO()):
                export_tflite_model(str(source), str(output))

            graph = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
            self.assertEqual(graph["source"]["package_class"], "fp32")
            self.assertNotIn("quantized_graph_contract", graph["source"])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
