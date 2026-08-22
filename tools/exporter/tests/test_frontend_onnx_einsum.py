"""Contract tests for Einsum normalization and Pow shape-program folding.

Einsum is rewritten to Transpose/MatMul on the ONNX graph before shape
inference rather than lowered as an emitter. ONNX's own Einsum inference
invents a fresh anonymous batch symbol even for concrete operands, and every
downstream value inherits it, so rewriting first is what keeps the rest of the
frontend seeing one fully resolved graph.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import onnx
    import safetensors  # noqa: F401 -- importing is part of the dependency gate.
    from onnx import TensorProto, helper, numpy_helper
except ImportError as error:  # This module is the installed-export-dependencies suite.
    raise unittest.SkipTest(
        "ONNX frontend tests require installed numpy, onnx, and safetensors"
    ) from error

from tools.exporter.errors import ExporterError
from tools.exporter.frontend_onnx import OnnxCompiler

ONNX_OPSET = 20


def _save(directory: Path, name: str, *, nodes, inputs, outputs, initializers=()) -> Path:
    graph = helper.make_graph(
        list(nodes), name.removesuffix(".onnx"),
        list(inputs), list(outputs), list(initializers),
    )
    model = helper.make_model(
        graph, producer_name="volvoxai-exporter-tests",
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
    )
    onnx.checker.check_model(model)
    destination = directory / name
    onnx.save(model, destination)
    return destination


def _einsum_model(directory: Path, equation: str, *, left, right, output) -> Path:
    return _save(
        directory, "einsum.onnx",
        nodes=[helper.make_node("Einsum", ["a", "b"], ["out"],
                                name="contract", equation=equation)],
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, left),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, right),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, output)],
    )


class EinsumNormalizationTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def _lower(self, path: Path):
        return OnnxCompiler(str(path)).lower()

    def test_canonical_equation_needs_no_transpose(self):
        path = _einsum_model(
            self.root, "bkn,bnd->bkd",
            left=[2, 4, 6], right=[2, 6, 8], output=[2, 4, 8],
        )
        graph, _ = self._lower(path)
        operators = [node["opType"] for node in graph["nodes"]]
        self.assertEqual(operators, ["BatchMatMul"])
        self.assertEqual(
            graph["nodes"][0]["outputs"]["out"]["shape"], [2, 4, 8],
        )

    def test_transposed_right_operand_emits_one_transpose(self):
        path = _einsum_model(
            self.root, "bkd,bnd->bkn",
            left=[2, 4, 6], right=[2, 8, 6], output=[2, 4, 8],
        )
        graph, _ = self._lower(path)
        operators = [node["opType"] for node in graph["nodes"]]
        self.assertEqual(operators, ["Transpose", "BatchMatMul"])
        self.assertEqual(graph["nodes"][0]["params"]["perm"], [0, 2, 1])
        self.assertEqual(
            graph["nodes"][-1]["outputs"]["out"]["shape"], [2, 4, 8],
        )

    def test_downstream_shapes_resolve_concretely(self):
        # The regression this guards: ONNX stamps an anonymous symbol on the
        # Einsum output, and a consumer then fails its own broadcast check.
        path = _save(
            self.root, "einsum_then_add.onnx",
            nodes=[
                helper.make_node("Einsum", ["a", "b"], ["scores"],
                                 name="contract", equation="bkd,bnd->bkn"),
                helper.make_node("Add", ["scores", "bias"], ["out"], name="shift"),
            ],
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 4, 6]),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 8, 6]),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 4, 8])],
            initializers=[numpy_helper.from_array(
                np.zeros((2, 4, 8), dtype=np.float32), name="bias",
            )],
        )
        graph, _ = self._lower(path)
        for node in graph["nodes"]:
            shape = node["outputs"]["out"]["shape"]
            self.assertTrue(
                all(isinstance(extent, int) for extent in shape),
                f"{node['opType']} kept a symbolic extent {shape}",
            )

    def test_rank_four_batch_prefix_is_admitted(self):
        path = _einsum_model(
            self.root, "abkn,abnd->abkd",
            left=[2, 3, 4, 6], right=[2, 3, 6, 8], output=[2, 3, 4, 8],
        )
        graph, _ = self._lower(path)
        self.assertEqual([node["opType"] for node in graph["nodes"]], ["BatchMatMul"])


class EinsumAdmissionTests(unittest.TestCase):
    """Equation-level proof, independent of any particular operand storage.

    These exercise the admission rule directly. Driving them through a whole
    graph would test ONNX's own validator as much as this one -- an implicit
    output, for instance, is refused by ONNX shape inference before the
    rewrite is ever reached.
    """

    def _plan(self, equation):
        compiler = OnnxCompiler.__new__(OnnxCompiler)
        return compiler._einsum_batch_matmul_plan(equation, "contract")

    def _assert_rejected(self, equation, pattern):
        with self.assertRaises(ExporterError) as caught:
            self._plan(equation)
        self.assertEqual(caught.exception.diagnostic.code, "VXEINSUM_UNSUPPORTED")
        self.assertIn(pattern, caught.exception.diagnostic.message)

    def test_admits_the_canonical_and_transposed_forms(self):
        self.assertEqual(self._plan("bkn,bnd->bkd"), (1, False, False))
        self.assertEqual(self._plan("bkd,bnd->bkn"), (1, False, True))
        self.assertEqual(self._plan("bdk,bnd->bkn"), (1, True, True))
        self.assertEqual(self._plan("abkn,abnd->abkd"), (2, False, False))
        self.assertEqual(self._plan("kn,nd->kd"), (0, False, False))

    def test_rejects_an_ellipsis(self):
        self._assert_rejected("...kn,...nd->...kd", "ellipsis")

    def test_rejects_an_implicit_output(self):
        self._assert_rejected("bkn,bnd", "explicit")

    def test_rejects_a_single_operand(self):
        self._assert_rejected("bkn->bnk", "has 1 operands")

    def test_rejects_a_repeated_label(self):
        # A repeated label is a diagonal, not a contraction.
        self._assert_rejected("bkk,bkd->bkd", "repeats")

    def test_rejects_a_permuted_batch_order(self):
        self._assert_rejected("abkn,band->abkd", "batch")

    def test_rejects_a_summed_away_label(self):
        self._assert_rejected("bkne,bnd->bkd", "sums away")

    def test_rejects_multiple_contracted_labels(self):
        self._assert_rejected("bkne,bned->bkd", "contracts 2 labels")


class PowShapeProgramTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def _scale_model(self, name: str, exponent: float) -> Path:
        # The conventional attention scale: the feature extent is read off the
        # activation's own shape, raised to a power, and divided back in.
        return _save(
            self.root, name,
            nodes=[
                helper.make_node("Shape", ["x"], ["shape"], name="shape"),
                helper.make_node("Gather", ["shape", "axis"], ["extent"],
                                 name="pick", axis=0),
                helper.make_node("Cast", ["extent"], ["extent_f"],
                                 name="widen", to=TensorProto.FLOAT),
                helper.make_node("Pow", ["extent_f", "exponent"], ["scale"], name="scale"),
                helper.make_node("Mul", ["x", "scale"], ["out"], name="apply"),
            ],
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4, 16])],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 4, 16])],
            initializers=[
                numpy_helper.from_array(np.asarray(2, dtype=np.int64), name="axis"),
                numpy_helper.from_array(
                    np.asarray(exponent, dtype=np.float32), name="exponent",
                ),
            ],
        )

    def _bounded_scale_model(self, name: str, *, start: int) -> Path:
        return _save(
            self.root, name,
            nodes=[
                helper.make_node("Shape", ["x"], ["shape"], name="shape"),
                helper.make_node(
                    "Slice", ["shape", "start", "end", "axis"],
                    ["extent_vector"], name="slice_extent",
                ),
                helper.make_node(
                    "Squeeze", ["extent_vector", "axis"], ["extent"],
                    name="squeeze_extent",
                ),
                helper.make_node(
                    "Cast", ["extent"], ["extent_f"], name="widen",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Pow", ["extent_f", "exponent"], ["scale"], name="scale"
                ),
                helper.make_node("Mul", ["x", "scale"], ["out"], name="apply"),
            ],
            inputs=[helper.make_tensor_value_info(
                "x", TensorProto.FLOAT, ["B", 4, 16],
            )],
            outputs=[helper.make_tensor_value_info(
                "out", TensorProto.FLOAT, ["B", 4, 16],
            )],
            initializers=[
                numpy_helper.from_array(np.asarray([start], dtype=np.int64), name="start"),
                numpy_helper.from_array(
                    np.asarray([
                        np.iinfo(np.int64).max if start < 0 else start + 1
                    ], dtype=np.int64),
                    name="end",
                ),
                numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="axis"),
                numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name="exponent"),
            ],
        )

    def test_folds_the_attention_scale_into_a_constant(self):
        graph, weights = OnnxCompiler(str(self._scale_model("scale.onnx", -0.5))).lower()
        # The whole Shape/Gather/Cast/Pow program folds away, leaving only the
        # multiply against a published constant.
        self.assertEqual([node["opType"] for node in graph["nodes"]], ["Mul"])
        folded = np.asarray(weights[graph["nodes"][0]["inputs"]["b"]])
        self.assertTrue(
            np.allclose(folded, 16.0 ** -0.5),
            f"folded scale {folded.reshape(-1)[:4]} is not 16 ** -0.5",
        )

    def test_folds_a_fixed_suffix_of_a_bounded_dynamic_shape(self):
        compiler = OnnxCompiler(
            str(self._bounded_scale_model("bounded_suffix_scale.onnx", start=-1)),
            dimension_bounds={"B": {"min": 1, "max": 4}},
        )
        graph, weights = compiler.lower()

        self.assertEqual(
            [node["opType"] for node in graph["nodes"]], ["Expand", "Mul"]
        )
        folded = [
            np.asarray(weights[name])
            for node in graph["nodes"]
            for name in node["inputs"].values()
            if name in weights
        ]
        self.assertTrue(any(np.allclose(value, 4.0) for value in folded))
        eliminated = {
            compiler.model.graph.node[index].name
            for index in compiler.structural_shape_nodes
        }
        self.assertTrue({"shape", "slice_extent", "squeeze_extent"} <= eliminated)

    def test_rejects_a_scale_selected_from_the_dynamic_batch_extent(self):
        compiler = OnnxCompiler(
            str(self._bounded_scale_model("bounded_batch_scale.onnx", start=0)),
            dimension_bounds={"B": {"min": 1, "max": 4}},
        )
        with self.assertRaises(ExporterError) as caught:
            compiler.lower()
        self.assertEqual(caught.exception.diagnostic.code, "VXONNX_UNSUPPORTED")
        self.assertEqual(caught.exception.diagnostic.source_node, "shape")

    def test_a_non_finite_result_is_not_folded(self):
        # Overflow must not silently substitute an infinite scale.
        with self.assertRaises(ExporterError) as caught:
            OnnxCompiler(str(self._scale_model("overflow.onnx", 1e30))).lower()
        self.assertNotEqual(caught.exception.diagnostic.code, "")


if __name__ == "__main__":
    unittest.main()


class LinearBiasFusionTests(unittest.TestCase):
    """`MatMul -> Add(immutable [d_out])` must bind as one biased Linear.

    In F32 the fused and unfused forms are numerically identical, so this is
    not about the float result. An unbound bias is broadcast to the full
    activation shape, which hides its per-output-channel structure from bias
    folding and leaves the PTQ author quantizing the pre-bias accumulator and
    the bias separately instead of folding it into one I32 accumulator.
    """

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def _model(self, name, *, bias, extra_consumer=False, dynamic_right=False):
        nodes = [helper.make_node(
            "MatMul", ["x", "w" if not dynamic_right else "y"], ["product"], name="project",
        )]
        outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 3])]
        nodes.append(helper.make_node("Add", ["product", "bias"], ["out"], name="shift"))
        if extra_consumer:
            # The unbiased product stays live, so it cannot be replaced.
            nodes.append(helper.make_node("Sigmoid", ["product"], ["raw"], name="keep"))
            outputs.append(
                helper.make_tensor_value_info("raw", TensorProto.FLOAT, [1, 4, 3]),
            )
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2])]
        initializers = [numpy_helper.from_array(
            np.asarray(bias, dtype=np.float32), name="bias",
        )]
        if dynamic_right:
            inputs.append(helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3]))
        else:
            initializers.append(numpy_helper.from_array(
                np.arange(6, dtype=np.float32).reshape(2, 3) / 8, name="w",
            ))
        return _save(
            self.root, name, nodes=nodes, inputs=inputs,
            outputs=outputs, initializers=initializers,
        )

    def test_binds_a_per_channel_bias(self):
        path = self._model("biased.onnx", bias=[0.5, -0.25, 1.0])
        graph, weights = OnnxCompiler(str(path)).lower()
        self.assertEqual([node["opType"] for node in graph["nodes"]], ["Linear"])
        linear = graph["nodes"][0]
        self.assertIn("bias", linear["inputs"])
        self.assertEqual(
            list(np.asarray(weights[linear["inputs"]["bias"]]).shape), [3],
        )
        self.assertEqual(linear["outputs"]["out"]["tensor"], graph["outputs"][0])

    def test_keeps_a_separate_add_when_the_product_stays_live(self):
        path = self._model("live.onnx", bias=[0.5, -0.25, 1.0], extra_consumer=True)
        graph, _ = OnnxCompiler(str(path)).lower()
        operators = sorted(node["opType"] for node in graph["nodes"])
        self.assertIn("Add", operators)
        self.assertNotIn("bias", graph["nodes"][0]["inputs"])

    def test_does_not_fuse_a_dynamic_right_operand(self):
        # A dynamic RHS lowers to BatchMatMul, which has no bias port.
        path = self._model("dynamic.onnx", bias=[0.5, -0.25, 1.0], dynamic_right=True)
        graph, _ = OnnxCompiler(str(path)).lower()
        operators = [node["opType"] for node in graph["nodes"]]
        self.assertIn("BatchMatMul", operators)
        self.assertIn("Add", operators)

    def test_does_not_fuse_a_per_position_constant(self):
        # A [1, 4, 3] constant is a genuine per-position tensor, not a bias.
        path = self._model("per_position.onnx", bias=np.ones((1, 4, 3), dtype=np.float32))
        graph, _ = OnnxCompiler(str(path)).lower()
        operators = [node["opType"] for node in graph["nodes"]]
        self.assertIn("Add", operators)
