from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import flatbuffers
from ai_edge_litert import schema_py_generated as schema

from tools.exporter.errors import ExporterError
from tools.exporter.importers.tflite import (
    import_tflite_source,
    tflite_subgraphs,
)
from tools.exporter.ir import IRDialect


def _source_model(*, invalid_signature: bool = False) -> bytes:
    activation_quantization = schema.QuantizationParametersT(
        min=[-1.0],
        max=[1.0],
        scale=[0.25],
        zeroPoint=[-3],
        quantizedDimension=0,
    )
    weight_quantization = schema.QuantizationParametersT(
        scale=[0.5, 0.25, 0.125, 0.0625],
        zeroPoint=[0, 0, 0, 0],
        detailsType=schema.QuantizationDetails.CustomQuantization,
        details=schema.CustomQuantizationT(custom=[9, 8]),
        quantizedDimension=0,
    )
    sparse_dimension = schema.DimensionMetadataT(
        format=schema.DimensionType.SPARSE_CSR,
        denseSize=4,
        arraySegmentsType=schema.SparseIndexVector.Int32Vector,
        arraySegments=schema.Int32VectorT(values=[0, 2, 4]),
        arrayIndicesType=schema.SparseIndexVector.Uint8Vector,
        arrayIndices=schema.Uint8VectorT(values=[0, 2, 1, 3]),
    )
    sparsity = schema.SparsityParametersT(
        traversalOrder=[0],
        blockMap=[],
        dimMetadata=[sparse_dimension],
    )
    tensors = [
        schema.TensorT(
            shape=[-1, 4],
            type=schema.TensorType.INT8,
            buffer=0,
            name=b"source_input",
            quantization=activation_quantization,
            shapeSignature=[-1, 4],
            hasRank=True,
        ),
        schema.TensorT(
            shape=[4],
            type=schema.TensorType.INT8,
            buffer=1,
            name=b"shared_weight",
            quantization=weight_quantization,
            sparsity=sparsity,
            hasRank=True,
        ),
        schema.TensorT(
            shape=[1, 4],
            type=schema.TensorType.INT8,
            buffer=0,
            name=b"mid",
            quantization=activation_quantization,
            hasRank=True,
        ),
        schema.TensorT(
            shape=[1, 4],
            type=schema.TensorType.INT8,
            buffer=0,
            name=b"source_output",
            quantization=activation_quantization,
            hasRank=True,
        ),
    ]
    operators = [
        schema.OperatorT(
            opcodeIndex=0,
            inputs=[0, 1],
            outputs=[2],
            builtinOptionsType=schema.BuiltinOptions.AddOptions,
            builtinOptions=schema.AddOptionsT(
                fusedActivationFunction=schema.ActivationFunctionType.RELU6,
                potScaleInt16=False,
            ),
            mutatingVariableInputs=[False, False],
            intermediates=[2],
            debugMetadataIndex=11,
        ),
        schema.OperatorT(
            opcodeIndex=1,
            inputs=[2, -1],
            outputs=[3],
            customOptions=[0, 255, 1],
            customOptionsFormat=schema.CustomOptionsFormat.FLEXBUFFERS,
        ),
    ]
    main = schema.SubGraphT(
        tensors=tensors,
        inputs=[0],
        outputs=[3],
        operators=operators,
        name=b"main",
        debugMetadataIndex=4,
    )
    auxiliary = schema.SubGraphT(
        tensors=[schema.TensorT(
            shape=[],
            type=schema.TensorType.FLOAT32,
            buffer=0,
            name=b"passthrough",
            hasRank=True,
            externalBuffer=7,
        )],
        inputs=[0],
        outputs=[0],
        operators=[],
        name=b"auxiliary",
    )
    model = schema.ModelT(
        version=3,
        operatorCodes=[
            schema.OperatorCodeT(
                deprecatedBuiltinCode=schema.BuiltinOperator.ADD,
                builtinCode=schema.BuiltinOperator.ADD,
                version=3,
            ),
            schema.OperatorCodeT(
                deprecatedBuiltinCode=schema.BuiltinOperator.CUSTOM,
                builtinCode=schema.BuiltinOperator.CUSTOM,
                customCode=b"volvox.custom",
                version=7,
            ),
        ],
        subgraphs=[main, auxiliary],
        description=b"source-faithful fixture",
        buffers=[
            schema.BufferT(),
            schema.BufferT(data=[1, 2, 3, 4]),
            schema.BufferT(data=list(b"metadata payload")),
        ],
        metadataBuffer=[2],
        metadata=[schema.MetadataT(name=b"TEST_METADATA", buffer=2)],
        signatureDefs=[
            schema.SignatureDefT(
                inputs=[schema.TensorMapT(name=b"question", tensorIndex=0)],
                outputs=[schema.TensorMapT(
                    name=b"answer",
                    tensorIndex=99 if invalid_signature else 3,
                )],
                signatureKey=b"serving_default",
                subgraphIndex=0,
            ),
            schema.SignatureDefT(
                inputs=[schema.TensorMapT(name=b"value", tensorIndex=0)],
                outputs=[schema.TensorMapT(name=b"result", tensorIndex=0)],
                signatureKey=b"aux",
                subgraphIndex=1,
            ),
        ],
        externalBufferGroups=[schema.ExternalBufferGroupT(name=b"weights.bin")],
        externalBuffers=[schema.ExternalBufferT(
            id=7,
            group=0,
            offset=4096,
            length=32,
            packing=b"none",
        )],
    )
    builder = flatbuffers.Builder(4096)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytes(builder.Output())


class TFLiteSourceImporterTests(unittest.TestCase):
    def test_preserves_all_subgraphs_opcodes_custom_options_and_signatures(self):
        raw = _source_model()
        graph = import_tflite_source(raw)
        graph.verify(IRDialect.SOURCE)

        self.assertEqual(graph.metadata["tflite_model_bytes"], raw)
        self.assertEqual(graph.metadata["tflite_schema_provider"],
                         "ai-edge-litert")
        self.assertEqual(len(tflite_subgraphs(graph)), 2)
        self.assertEqual(
            tflite_subgraphs(graph)[1].metadata["tflite_subgraph_index"], 1)
        self.assertEqual([node.op_type for node in graph.nodes],
                         ["ADD", "volvox.custom"])
        self.assertEqual([node.version for node in graph.nodes], [3, 7])
        self.assertEqual(graph.nodes[1].domain, "tflite.custom")
        self.assertEqual(graph.nodes[1].attributes[0].raw, b"\x00\xff\x01")
        self.assertEqual([port.value for port in graph.nodes[1].inputs], [
            "@tflite/subgraph/0/tensor/2", None,
        ])
        signatures = graph.metadata["tflite_signatures"]
        self.assertEqual([item["signature_key"] for item in signatures],
                         [b"serving_default", b"aux"])
        self.assertEqual(signatures[1]["subgraph_index"], 1)
        self.assertEqual(signatures[0]["outputs"][0]["tensor"],
                         "@tflite/subgraph/0/tensor/3")

    def test_preserves_shapes_types_buffers_quantization_and_sparsity(self):
        raw = _source_model()
        graph = import_tflite_source(raw)
        source_input = graph.tensors["@tflite/subgraph/0/tensor/0"]
        weight = graph.tensors["@tflite/subgraph/0/tensor/1"]

        self.assertEqual(source_input.shape, (None, 4))
        self.assertEqual(source_input.dtype, "int8")
        self.assertEqual(source_input.source_dtype, "tflite.TensorType.INT8")
        self.assertEqual(source_input.metadata["tflite_shape"], (-1, 4))
        self.assertEqual(source_input.metadata["tflite_shape_signature"], (-1, 4))
        self.assertTrue(source_input.metadata["tflite_has_rank"])
        quantization = weight.metadata["tflite_quantization"]
        self.assertIsNone(weight.quantization)
        self.assertEqual(quantization["scale"], (0.5, 0.25, 0.125, 0.0625))
        self.assertEqual(quantization["zero_point"], (0, 0, 0, 0))
        self.assertEqual(quantization["details_type_name"], "CustomQuantization")
        sparsity = weight.metadata["tflite_sparsity"]["object"]
        self.assertEqual(sparsity["traversalOrder"], (0,))
        self.assertEqual(
            sparsity["dimMetadata"][0]["arraySegments"]["values"],
            (0, 2, 4),
        )
        self.assertTrue(weight.initializer)
        self.assertIsNotNone(weight.data)
        self.assertEqual(weight.data.shard, "tflite.buffer/1")
        self.assertEqual(weight.data.byte_length, 4)
        self.assertEqual(weight.data.checksum,
                         hashlib.sha256(b"\x01\x02\x03\x04").hexdigest())
        self.assertEqual(
            graph.metadata["tflite_external_buffers"][0]["id"], 7)

    def test_builtin_options_have_generated_semantics_and_recoverable_source(self):
        raw = _source_model()
        first = import_tflite_source(raw)
        second = import_tflite_source(raw)
        add = first.nodes[0]
        option = add.attributes[0]

        self.assertEqual(option.name, "builtin_options")
        self.assertEqual(option.kind, "AddOptions")
        self.assertEqual(option.value["fusedActivationFunction"],
                         schema.ActivationFunctionType.RELU6)
        self.assertFalse(option.value["potScaleInt16"])
        self.assertTrue(option.raw)
        source_offset = add.metadata["tflite_builtin_options"]["table_offset"]
        self.assertIsInstance(source_offset, int)
        self.assertGreater(source_offset, 0)
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(
            add.provenance[0].location,
            "<memory>:subgraphs[0].operators[0]",
        )

    def test_path_storage_reference_and_schema_errors_are_diagnostic(self):
        raw = _source_model()
        with tempfile.TemporaryDirectory(prefix="volvoxai-tflite-source-") as directory:
            path = Path(directory) / "fixture.tflite"
            path.write_bytes(raw)
            graph = import_tflite_source(path)
            weight = graph.tensors["@tflite/subgraph/0/tensor/1"]
            self.assertEqual(weight.data.source_uri, str(path))

        with self.assertRaises(ExporterError) as invalid_identifier:
            import_tflite_source(b"\x00\x00\x00\x00NOPE")
        self.assertEqual(invalid_identifier.exception.diagnostic.code, "VXTFLITE003")

        with self.assertRaises(ExporterError) as invalid_signature:
            import_tflite_source(_source_model(invalid_signature=True))
        self.assertEqual(invalid_signature.exception.diagnostic.code, "VXTFLITE006")


if __name__ == "__main__":
    unittest.main()
