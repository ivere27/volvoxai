from __future__ import annotations

import unittest

from tools.exporter.errors import ExporterError
from tools.exporter.ir import (
    AffineQuantization,
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    Provenance,
    TensorValue,
    ValuePort,
)


def tensor(
    name: str,
    *,
    shape: tuple[int | None, ...] = (1, 4),
    dtype: str = "float32",
    initializer: bool = False,
    public_input: bool = False,
    public_output: bool = False,
) -> TensorValue:
    return TensorValue(
        name=name,
        shape=shape,
        dtype=dtype,
        source_dtype=dtype,
        initializer=initializer,
        public_input=public_input,
        public_output=public_output,
    )


class GraphIRTests(unittest.TestCase):
    def valid_graph(self) -> GraphIR:
        graph = GraphIR(source_format="onnx", source_name="fixture.onnx")
        graph.add_tensor(tensor("x", public_input=True))
        graph.add_tensor(tensor("bias", initializer=True))
        graph.add_tensor(tensor("y", public_output=True))
        graph.inputs.append("x")
        graph.add_node(OpNode.from_maps(
            name="add",
            op_type="Add",
            inputs={"a": "x", "b": "bias"},
            outputs={"out": "y"},
        ))
        graph.outputs.append("y")
        return graph

    def assert_error_code(self, graph: GraphIR, code: str) -> ExporterError:
        with self.assertRaises(ExporterError) as caught:
            graph.require_fixed_static_dag()
        self.assertEqual(caught.exception.diagnostic.code, code)
        return caught.exception

    def test_valid_static_graph_and_scalar_output(self):
        graph = self.valid_graph()
        graph.require_fixed_static_dag()

        scalar = GraphIR(source_format="onnx", source_name="scalar.onnx")
        scalar.add_tensor(tensor("x", shape=(), public_input=True))
        scalar.add_tensor(tensor("y", shape=(), public_output=True))
        scalar.inputs.append("x")
        scalar.add_node(OpNode.from_maps(
            "identity", "Identity", {"input": "x"}, {"out": "y"}))
        scalar.outputs.append("y")
        scalar.require_fixed_static_dag()

    def test_duplicate_tensor_definition_is_rejected(self):
        graph = GraphIR(source_format="onnx", source_name="duplicate.onnx")
        graph.add_tensor(tensor("x"))
        with self.assertRaises(ExporterError) as caught:
            graph.add_tensor(tensor("x"))
        self.assertEqual(caught.exception.diagnostic.code, "VXIR001")

    def test_missing_public_input_descriptor_is_rejected(self):
        graph = self.valid_graph()
        graph.tensors["x"].public_input = False
        self.assert_error_code(graph, "VXIR002")

    def test_control_flow_is_rejected_before_emission(self):
        graph = self.valid_graph()
        graph.nodes[0].op_type = "If"
        error = self.assert_error_code(graph, "VXIR003")
        self.assertEqual(error.diagnostic.source_node, "add")

    def test_non_topological_reference_is_rejected(self):
        graph = self.valid_graph()
        graph.nodes[0] = OpNode.from_maps(
            "add", "Add", {"a": "future", "b": "bias"}, {"out": "y"})
        self.assert_error_code(graph, "VXIR004")

    def test_output_redefinition_is_rejected(self):
        graph = self.valid_graph()
        graph.nodes[0] = OpNode.from_maps(
            "add", "Add", {"a": "x", "b": "bias"}, {"out": "x"})
        self.assert_error_code(graph, "VXIR005")

    def test_missing_output_descriptor_is_rejected(self):
        graph = self.valid_graph()
        del graph.tensors["y"]
        self.assert_error_code(graph, "VXIR006")

    def test_missing_public_output_is_rejected(self):
        graph = self.valid_graph()
        graph.outputs[:] = ["missing"]
        self.assert_error_code(graph, "VXIR007")

    def test_execution_dtype_and_static_shape_are_checked(self):
        graph = self.valid_graph()
        graph.tensors["x"].dtype = "int64"
        self.assert_error_code(graph, "VXIR008")

        graph = self.valid_graph()
        graph.tensors["y"].shape = (1, None)
        self.assert_error_code(graph, "VXIR009")

    def test_feature_provenance_and_abi_changes_are_preserved(self):
        graph = self.valid_graph()
        provenance = Provenance(
            source_format="onnx",
            source_name="source_add",
            source_op="Add",
            location="graph.node[0]",
        ).rewritten("canonicalize")
        graph.nodes[0].provenance = (provenance,)
        graph.nodes[0].feature = "fixed_adapter"
        graph.features.setdefault("fixed_adapter", []).append("add")
        graph.record_abi_change("input_dtype", "ids", "int64", "int32")

        self.assertEqual(graph.feature_nodes("fixed_adapter"), ("add",))
        self.assertEqual(tuple(graph.iter_provenance()), (provenance,))
        self.assertEqual(provenance.rewrites, ("canonicalize",))
        self.assertEqual(graph.abi_changes[0]["exported"], "int32")

    def test_source_ir_preserves_ordered_omitted_ports_and_symbolic_shapes(self):
        graph = GraphIR(source_format="onnx", source_name="optional.onnx")
        graph.add_tensor(tensor("x", shape=("batch", 4), public_input=True))
        graph.add_tensor(tensor("y", shape=("batch", 4), public_output=True))
        graph.inputs.append("x")
        graph.add_node(OpNode(
            name="clip",
            op_type="Clip",
            inputs=(
                ValuePort("input", "x", 0),
                ValuePort("min", None, 1),
                ValuePort("max", None, 2),
            ),
            outputs=(ValuePort("out", "y", 0),),
            domain="ai.onnx",
            version=13,
            attributes=(OpAttribute("source", "bytes", raw=b"exact"),),
        ))
        graph.outputs.append("y")
        graph.verify(IRDialect.SOURCE)
        self.assertIsNone(graph.nodes[0].inputs[1].value)
        with self.assertRaises(ExporterError) as caught:
            graph.require_fixed_static_dag()
        self.assertEqual(caught.exception.diagnostic.code, "VXIR009")

    def test_use_def_index_and_fingerprint_are_deterministic(self):
        graph = self.valid_graph()
        graph.verify(IRDialect.SOURCE)
        index = graph.use_def()
        self.assertEqual(index.producers["y"].node_index, 0)
        self.assertEqual(index.consumers["x"][0].port, "a")
        self.assertEqual(graph.fingerprint(), graph.clone().fingerprint())

    def test_runtime_affine_parameters_are_internal_tensor_references(self):
        graph = GraphIR(
            source_format="volvoxai", source_name="graph.json",
            dialect=IRDialect.RUNTIME,
        )
        graph.add_tensor(TensorValue(
            name="x", shape=(1, 4), dtype="int8", source_dtype="int8",
            public_input=True,
            quantization=AffineQuantization(
                "per_tensor", "__q.x.scale", "__q.x.zero"),
        ))
        graph.add_tensor(TensorValue(
            name="__q.x.scale", shape=(1,), dtype="float32",
            source_dtype="float32", initializer=True, raw_data=b"\x00" * 4,
        ))
        graph.add_tensor(TensorValue(
            name="__q.x.zero", shape=(1,), dtype="int8",
            source_dtype="int8", initializer=True, raw_data=b"\x00",
        ))
        graph.add_tensor(TensorValue(
            name="y", shape=(1, 4), dtype="int8", source_dtype="int8",
            public_output=True,
            quantization=AffineQuantization(
                "per_tensor", "__q.x.scale", "__q.x.zero"),
        ))
        graph.inputs.append("x")
        graph.add_node(OpNode.from_maps(
            "identity", "Identity", {"input": "x"}, {"out": "y"}))
        graph.outputs.append("y")
        graph.verify(IRDialect.RUNTIME)

        graph.tensors["__q.x.scale"].public_output = True
        with self.assertRaises(ExporterError) as caught:
            graph.verify(IRDialect.RUNTIME)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR026")

    def test_runtime_requires_outputs_and_rejects_hidden_affine_params(self):
        empty = self.valid_graph()
        empty.dialect = IRDialect.RUNTIME
        empty.outputs.clear()
        with self.assertRaises(ExporterError) as caught:
            empty.verify(IRDialect.RUNTIME)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR040")

        invalid = self.valid_graph()
        invalid.dialect = IRDialect.RUNTIME
        invalid.nodes[0].attributes = (OpAttribute(
            "params", "volvox.params",
            {"private": [{"affine": {"input_scale": 0.25}}]},
        ),)
        with self.assertRaises(ExporterError) as caught:
            invalid.verify(IRDialect.RUNTIME)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR042")
        self.assertIn(
            "params.private[0].affine.input_scale",
            caught.exception.diagnostic.message,
        )

        semantic_scale = self.valid_graph()
        semantic_scale.dialect = IRDialect.RUNTIME
        semantic_scale.nodes[0].attributes = (OpAttribute(
            "params", "volvox.params", {"scale": 0.5},
        ),)
        semantic_scale.verify(IRDialect.RUNTIME)

    def test_transaction_snapshot_restore(self):
        graph = self.valid_graph()
        before = graph.fingerprint()
        snapshot = graph.clone()
        graph.nodes[0].op_type = "Mul"
        graph.restore(snapshot)
        self.assertEqual(graph.fingerprint(), before)


if __name__ == "__main__":
    unittest.main()
