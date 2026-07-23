from __future__ import annotations

import unittest

import numpy as np

from tools.exporter.errors import ExporterError
from tools.exporter.ir import (
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    Provenance,
    TensorDataRef,
    TensorValue,
)
from tools.exporter.optimizer.typed_constant_folding import (
    RuntimeConstantFoldingPass,
)
from tools.exporter.optimizer.typed_silu_fusion import RuntimeSiluFusionPass
from tools.exporter.optimizer.typed_specialization import (
    InputHoistingSpec,
    RuntimeInputHoistingPass,
    RuntimeInputSpecializationPass,
)
from tools.exporter.pipeline import VerifiedPipeline
from tools.exporter.reference_executor import execute_reference
from tools.exporter.runtime_ir import export_runtime_package


def _tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: str = "float32",
    initializer: bool = False,
    public_input: bool = False,
    public_output: bool = False,
    source_name: str | None = None,
) -> TensorValue:
    metadata = {}
    if source_name is not None:
        metadata["runtime_input_fields"] = {"source_name": source_name}
    return TensorValue(
        name=name,
        shape=shape,
        dtype=dtype,
        source_dtype=dtype,
        initializer=initializer,
        public_input=public_input,
        public_output=public_output,
        data=TensorDataRef(name) if initializer else None,
        metadata=metadata,
    )


def _provenance(name: str, op: str) -> tuple[Provenance, ...]:
    return (Provenance(
        source_format="onnx",
        source_name=name,
        source_op=op,
        location=f"model.onnx:{name}",
    ),)


def _add_node(
    graph: GraphIR,
    name: str,
    op_type: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    *,
    params: dict | None = None,
) -> None:
    graph.add_node(OpNode.from_maps(
        name,
        op_type,
        inputs,
        outputs,
        attributes=(() if params is None else (
            OpAttribute("params", "volvox.params", params),
        )),
        provenance=_provenance(name, op_type),
    ))


def _silu_graph(*, shared_gate: bool = False) -> GraphIR:
    graph = GraphIR("volvoxai", "silu.json", dialect=IRDialect.RUNTIME)
    for tensor in (
        _tensor("x", (2, 4), public_input=True),
        _tensor("gate", (2, 4)),
        _tensor("y", (2, 4), public_output=True),
    ):
        graph.add_tensor(tensor)
    graph.inputs.append("x")
    _add_node(graph, "sigmoid", "Sigmoid", {"input": "x"}, {"out": "gate"})
    # Reverse operands to prove Mul commutativity is handled explicitly.
    _add_node(graph, "multiply", "Mul", {"a": "gate", "b": "x"}, {"out": "y"})
    graph.outputs.append("y")
    if shared_gate:
        graph.add_tensor(_tensor("side", (2, 4), public_output=True))
        _add_node(graph, "side-use", "Identity", {"input": "gate"}, {"out": "side"})
        graph.outputs.append("side")
    graph.verify(IRDialect.RUNTIME)
    return graph


class RuntimeSiluFusionTests(unittest.TestCase):
    def test_fuses_with_combined_typed_provenance_and_preserves_values(self):
        graph = _silu_graph()
        sample = np.asarray([
            [-20.0, -2.0, -0.0, 0.5],
            [1.0, 3.0, 10.0, 20.0],
        ], dtype=np.float32)
        expected = execute_reference(graph, {}, {"x": sample}).outputs["y"]

        report = VerifiedPipeline((RuntimeSiluFusionPass(),)).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual([node.op_type for node in graph.nodes], ["SiLU"])
        fused = graph.nodes[0]
        self.assertEqual(fused.name, "multiply")
        self.assertEqual(fused.input_map(), {"input": "x"})
        self.assertNotIn("gate", graph.tensors)
        self.assertEqual(
            [item.source_name for item in fused.provenance],
            ["sigmoid", "multiply"],
        )
        self.assertTrue(all(
            item.rewrites[-1] == "runtime-silu-fusion"
            for item in fused.provenance
        ))
        actual = execute_reference(graph, {}, {"x": sample}).outputs["y"]
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0.0)

    def test_refuses_shared_gate_without_mutation(self):
        graph = _silu_graph(shared_gate=True)
        before = graph.fingerprint()

        report = VerifiedPipeline((RuntimeSiluFusionPass(),)).run(graph)

        self.assertEqual(report.total_changes, 0)
        self.assertEqual(graph.fingerprint(), before)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["Sigmoid", "Mul", "Identity"],
        )


def _specialization_graph() -> GraphIR:
    graph = GraphIR("volvoxai", "specialize.json", dialect=IRDialect.RUNTIME)
    for tensor in (
        _tensor("x", (2,), public_input=True),
        _tensor(
            "input1", (2,), public_input=True, source_name="authored_bias",
        ),
        _tensor("y", (2,), public_output=True),
    ):
        graph.add_tensor(tensor)
    graph.inputs.extend(("x", "input1"))
    _add_node(graph, "add", "Add", {"a": "x", "b": "input1"}, {"out": "y"})
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


class RuntimeInputSpecializationTests(unittest.TestCase):
    def test_freezes_authored_alias_and_records_exact_input_abi(self):
        graph = _specialization_graph()
        tensors: dict[str, np.ndarray] = {}
        sample = np.asarray([0.25, -2.0], dtype=np.float32)
        bias = np.asarray([1.5, 0.75], dtype=np.float32)
        expected = execute_reference(
            graph, tensors, {"x": sample, "input1": bias},
        ).outputs["y"]

        report = VerifiedPipeline((RuntimeInputSpecializationPass(
            {"authored_bias": bias}, tensors,
        ),)).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual(graph.inputs, ["x"])
        frozen = graph.tensors["input1"]
        self.assertTrue(frozen.initializer)
        self.assertFalse(frozen.public_input)
        self.assertEqual(frozen.data, TensorDataRef("input1"))
        np.testing.assert_array_equal(tensors["input1"], bias)
        self.assertEqual(graph.abi_changes[-1]["kind"], "input-specialization")
        self.assertEqual(
            graph.abi_changes[-1]["source"]["source_name"], "authored_bias",
        )
        self.assertEqual(
            graph.abi_changes[-1]["exported"]["value_sha256"],
            frozen.metadata["optimizer_specialization"]["value_sha256"],
        )
        self.assertEqual(
            graph.nodes[0].provenance[0].rewrites[-1],
            "runtime-input-specialization",
        )
        actual = execute_reference(graph, tensors, {"x": sample}).outputs["y"]
        np.testing.assert_array_equal(actual, expected)
        document, published_tensors = export_runtime_package(graph, tensors)
        self.assertEqual(list(document["inputs"]), ["x"])
        self.assertEqual(
            document["source"]["abi_changes"][-1]["kind"],
            "input-specialization",
        )
        self.assertIn("input1", published_tensors)

    def test_unknown_request_rolls_back_all_valid_bindings(self):
        graph = _specialization_graph()
        tensors: dict[str, np.ndarray] = {}
        before = graph.fingerprint()
        pass_ = RuntimeInputSpecializationPass(
            {
                "authored_bias": np.ones((2,), dtype=np.float32),
                "missing": np.zeros((2,), dtype=np.float32),
            },
            tensors,
        )

        with self.assertRaises(ExporterError) as raised:
            VerifiedPipeline((pass_,)).run(graph)

        self.assertEqual(raised.exception.diagnostic.code, "VXTYPESPEC001")
        self.assertEqual(graph.fingerprint(), before)
        self.assertEqual(tensors, {})


def _hoisting_graph() -> GraphIR:
    graph = GraphIR("volvoxai", "hoist.json", dialect=IRDialect.RUNTIME)
    for tensor in (
        _tensor("x", (3,), public_input=True),
        _tensor("derived", (3,)),
        _tensor("y", (3,), public_output=True),
    ):
        graph.add_tensor(tensor)
    graph.inputs.append("x")
    _add_node(
        graph, "derive", "Identity", {"input": "x"}, {"out": "derived"},
    )
    _add_node(graph, "consume", "Add", {"a": "derived", "b": "x"}, {"out": "y"})
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph


class RuntimeInputHoistingTests(unittest.TestCase):
    def test_hoists_named_tensor_and_preserves_values_under_caller_obligation(self):
        graph = _hoisting_graph()
        sample = np.asarray([1.0, -0.5, 3.0], dtype=np.float32)
        expected = execute_reference(graph, {}, {"x": sample}).outputs["y"]

        report = VerifiedPipeline((RuntimeInputHoistingPass(
            (InputHoistingSpec("derived", "float32", tensor_name="derived"),),
        ),)).run(graph)

        self.assertEqual(report.total_changes, 1)
        self.assertEqual(graph.inputs, ["x", "derived"])
        self.assertEqual([node.name for node in graph.nodes], ["consume"])
        self.assertTrue(graph.tensors["derived"].public_input)
        self.assertEqual(graph.abi_changes[-1]["kind"], "input-hoisting")
        self.assertEqual(
            graph.abi_changes[-1]["exported"]["caller_obligation"],
            "supply-identical-derived-value",
        )
        self.assertEqual(
            graph.nodes[0].provenance[0].rewrites[-1],
            "runtime-input-hoisting",
        )
        actual = execute_reference(
            graph, {}, {"x": sample, "derived": sample},
        ).outputs["y"]
        np.testing.assert_array_equal(actual, expected)
        document, _ = export_runtime_package(graph, {})
        self.assertEqual(list(document["inputs"]), ["x", "derived"])
        self.assertEqual(
            document["source"]["abi_changes"][-1]["kind"],
            "input-hoisting",
        )

    def test_dtype_refusal_is_transactional(self):
        graph = _hoisting_graph()
        before = graph.fingerprint()

        with self.assertRaises(ExporterError) as raised:
            VerifiedPipeline((RuntimeInputHoistingPass(
                (InputHoistingSpec("derived", "int32", tensor_name="derived"),),
            ),)).run(graph)

        self.assertEqual(raised.exception.diagnostic.code, "VXTYPEHOIST002")
        self.assertEqual(graph.fingerprint(), before)


def _constant_matmul_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR("volvoxai", "constant.json", dialect=IRDialect.RUNTIME)
    for tensor in (
        _tensor("x", (1, 2, 3), public_input=True),
        _tensor("packed", (1, 4, 3), initializer=True),
        _tensor("weight", (1, 3, 4)),
        # Occupy the preferred view name to exercise deterministic suffixing.
        _tensor("weight.dense", (1,), initializer=True),
        _tensor("y", (1, 2, 4), public_output=True),
    ):
        graph.add_tensor(tensor)
    graph.inputs.append("x")
    _add_node(
        graph,
        "transpose-weight",
        "Transpose",
        {"input": "packed"},
        {"out": "weight"},
        params={"perm": [0, 2, 1]},
    )
    _add_node(
        graph, "project", "MatMul", {"a": "x", "b": "weight"}, {"out": "y"},
    )
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    tensors = {
        "packed": np.asarray([
            [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5],
             [-0.25, 2.0, 1.0], [0.125, -0.75, 0.5]],
        ], dtype=np.float32),
        "weight.dense": np.asarray([7.0], dtype=np.float32),
    }
    return graph, tensors


class RuntimeConstantFoldingTests(unittest.TestCase):
    def test_folds_storage_movement_demotes_matmul_and_preserves_values(self):
        graph, tensors = _constant_matmul_graph()
        sample = np.asarray([
            [[1.0, -2.0, 0.5], [-0.25, 0.75, 2.0]],
        ], dtype=np.float32)
        expected = execute_reference(graph, tensors, {"x": sample}).outputs["y"]

        folding = RuntimeConstantFoldingPass(tensors)
        report = VerifiedPipeline((folding,)).run(graph)

        self.assertEqual(report.total_changes, 2)
        self.assertEqual((folding.folded, folding.demoted), (1, 1))
        self.assertEqual([node.op_type for node in graph.nodes], ["Linear"])
        linear = graph.nodes[0]
        self.assertEqual(linear.name, "project")
        self.assertEqual(linear.input_map()["weight"], "weight.dense.2")
        self.assertEqual(
            linear.attributes[0].value, {"weight_layout": "IN_OUT"},
        )
        self.assertTrue(graph.tensors["weight"].initializer)
        self.assertEqual(
            graph.tensors["weight"].metadata["optimizer_constant_folding"][
                "source_node"
            ],
            "transpose-weight",
        )
        self.assertEqual(
            linear.provenance[0].rewrites[-1], "runtime-constant-folding",
        )
        actual = execute_reference(graph, tensors, {"x": sample}).outputs["y"]
        np.testing.assert_array_equal(actual, expected)

        second_graph, second_tensors = _constant_matmul_graph()
        VerifiedPipeline((RuntimeConstantFoldingPass(second_tensors),)).run(
            second_graph,
        )
        self.assertEqual(
            second_graph.nodes[0].input_map()["weight"], "weight.dense.2",
        )
        self.assertEqual(second_graph.fingerprint(), graph.fingerprint())

    def test_folds_in_range_gather_but_refuses_out_of_range_indices(self):
        for index, expected_changes in ((1, 2), (7, 0)):
            with self.subTest(index=index):
                graph = GraphIR(
                    "volvoxai", "gather.json", dialect=IRDialect.RUNTIME,
                )
                for tensor in (
                    _tensor("table", (3, 2), initializer=True),
                    _tensor("index", (1,), dtype="int32", initializer=True),
                    _tensor("selected", (1, 2)),
                    _tensor("expanded", (1, 1, 2)),
                    _tensor("y", (1, 1, 2), public_output=True),
                ):
                    graph.add_tensor(tensor)
                _add_node(
                    graph, "select", "Gather",
                    {"input": "table", "indices": "index"},
                    {"out": "selected"}, params={"axis": 0},
                )
                _add_node(
                    graph, "expand", "Unsqueeze",
                    {"input": "selected"}, {"out": "expanded"},
                    params={"axes": [0]},
                )
                _add_node(
                    graph, "publish", "Identity",
                    {"input": "expanded"}, {"out": "y"},
                )
                graph.outputs.append("y")
                graph.verify(IRDialect.RUNTIME)
                tensors = {
                    "table": np.arange(6, dtype=np.float32).reshape(3, 2),
                    "index": np.asarray([index], dtype=np.int32),
                }
                before = graph.fingerprint()

                report = VerifiedPipeline((
                    RuntimeConstantFoldingPass(tensors),
                )).run(graph)

                self.assertEqual(report.total_changes, expected_changes)
                if expected_changes:
                    np.testing.assert_array_equal(
                        tensors["selected"], tensors["table"][[index]],
                    )
                    np.testing.assert_array_equal(
                        tensors["expanded"], tensors["table"][[index]][None, ...],
                    )
                else:
                    self.assertEqual(graph.fingerprint(), before)
                    self.assertNotIn("selected", tensors)

    def test_payload_failure_rolls_back_prior_folds(self):
        class FailOnceDict(dict):
            failed = False

            def __setitem__(self, key, value):
                if key == "right.folded" and not self.failed:
                    self.failed = True
                    raise RuntimeError("injected tensor-store failure")
                super().__setitem__(key, value)

        graph = GraphIR("volvoxai", "rollback.json", dialect=IRDialect.RUNTIME)
        for tensor in (
            _tensor("left", (2,), initializer=True),
            _tensor("right", (2,), initializer=True),
            _tensor("left.folded", (2,)),
            _tensor("right.folded", (2,)),
            _tensor("y", (2,), public_output=True),
        ):
            graph.add_tensor(tensor)
        _add_node(
            graph, "fold-left", "Identity",
            {"input": "left"}, {"out": "left.folded"},
        )
        _add_node(
            graph, "fold-right", "Identity",
            {"input": "right"}, {"out": "right.folded"},
        )
        _add_node(
            graph, "add", "Add",
            {"a": "left.folded", "b": "right.folded"}, {"out": "y"},
        )
        graph.outputs.append("y")
        graph.verify(IRDialect.RUNTIME)
        tensors = FailOnceDict({
            "left": np.asarray([1.0, 2.0], dtype=np.float32),
            "right": np.asarray([3.0, 4.0], dtype=np.float32),
        })
        before_graph = graph.fingerprint()
        before_keys = tuple(tensors)

        with self.assertRaises(ExporterError) as raised:
            VerifiedPipeline((RuntimeConstantFoldingPass(tensors),)).run(graph)

        self.assertEqual(raised.exception.diagnostic.code, "VXOPT005")
        self.assertEqual(graph.fingerprint(), before_graph)
        self.assertEqual(tuple(tensors), before_keys)
        self.assertNotIn("left.folded", tensors)
        self.assertNotIn("right.folded", tensors)


if __name__ == "__main__":
    unittest.main()
