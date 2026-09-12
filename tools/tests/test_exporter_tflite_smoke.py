from __future__ import annotations

from pathlib import Path
import sys
import unittest

import flatbuffers
from ai_edge_litert import schema_py_generated as schema


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.exporter.errors import ExporterError
from tools.exporter.importers.tflite import import_tflite_source
from tools.exporter.ir import IRDialect


def _minimal_passthrough_tflite() -> bytes:
    """Build a one-tensor TFLite source without importing TensorFlow."""

    model = schema.ModelT(
        version=3,
        subgraphs=[
            schema.SubGraphT(
                tensors=[
                    schema.TensorT(
                        shape=[1],
                        type=schema.TensorType.FLOAT32,
                        buffer=0,
                        name=b"input",
                        hasRank=True,
                    )
                ],
                inputs=[0],
                outputs=[0],
                operators=[],
                name=b"main",
            )
        ],
        buffers=[schema.BufferT()],
    )
    builder = flatbuffers.Builder(256)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytes(builder.Output())


class TfliteExporterSmokeTests(unittest.TestCase):
    def test_tiny_source_import_is_deterministic(self) -> None:
        source = _minimal_passthrough_tflite()
        first = import_tflite_source(source)
        second = import_tflite_source(source)

        first.verify(IRDialect.SOURCE)
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(first.metadata["tflite_model_bytes"], source)
        self.assertEqual(first.metadata["tflite_schema_provider"], "ai-edge-litert")
        self.assertEqual(first.inputs, ["@tflite/subgraph/0/tensor/0"])
        self.assertEqual(first.outputs, first.inputs)
        tensor = first.tensors[first.inputs[0]]
        self.assertEqual(tensor.shape, (1,))
        self.assertEqual(tensor.dtype, "float32")
        self.assertTrue(tensor.public_input)
        self.assertTrue(tensor.public_output)

    def test_malformed_source_fails_closed_and_importer_remains_reusable(self) -> None:
        with self.assertRaises(ExporterError) as caught:
            import_tflite_source(b"\x00\x00\x00\x00NOPE")
        self.assertEqual(caught.exception.diagnostic.code, "VXTFLITE003")

        recovered = import_tflite_source(_minimal_passthrough_tflite())
        recovered.verify(IRDialect.SOURCE)
        self.assertEqual(recovered.outputs, ["@tflite/subgraph/0/tensor/0"])


if __name__ == "__main__":
    unittest.main()
