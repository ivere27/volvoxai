from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import tools.exporter.typed_ptq as typed_ptq
from tools.exporter.capabilities import validate_graph
from tools.exporter.differential import compare_tensor_maps
from tools.exporter.errors import ExporterError
from tools.exporter.ir import (
    AffineQuantization,
    GraphIR,
    IRDialect,
    OpAttribute,
    OpNode,
    TensorValue,
)
from tools.exporter.reference_executor import execute_reference
from tools.exporter.runtime_ir import export_runtime_package, import_runtime_package
from tools.exporter.shape_system import ShapeEnvironment
from tools.exporter.optimizer.safetensors_io import read_safetensors, write_safetensors
from tools.exporter.typed_ptq import (
    CalibrationTable,
    PTQConfig,
    calibration_profile_from_ranges,
    materialize_runtime_ptq,
    plan_runtime_ptq,
    required_ptq_observations,
)


def add_tensor(
    graph: GraphIR,
    name: str,
    shape: tuple[int | str, ...],
    *,
    initializer: bool = False,
    public_input: bool = False,
    public_output: bool = False,
    dtype: str = "float32",
) -> None:
    graph.add_tensor(TensorValue(
        name=name,
        shape=shape,
        dtype=dtype,
        source_dtype=dtype,
        initializer=initializer,
        public_input=public_input,
        public_output=public_output,
    ))


def linear_graph(
    *,
    layout: str = "din_dout",
    bias: bool = True,
) -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR(
        source_format="volvoxai",
        source_name="linear.json",
        dialect=IRDialect.RUNTIME,
    )
    add_tensor(graph, "x", (2, 3), public_input=True)
    weight_shape = (3, 2) if layout == "din_dout" else (2, 3)
    add_tensor(graph, "weight", weight_shape, initializer=True)
    if bias:
        add_tensor(graph, "bias", (2,), initializer=True)
    add_tensor(graph, "y", (2, 2), public_output=True)
    graph.inputs.append("x")
    inputs = {"input": "x", "weight": "weight"}
    if bias:
        inputs["bias"] = "bias"
    graph.add_node(OpNode.from_maps(
        name="dense",
        op_type="Linear",
        inputs=inputs,
        outputs={"out": "y"},
        attributes=(OpAttribute(
            "params", "volvox.params", {"weight_layout": layout},
        ),),
    ))
    graph.outputs.append("y")
    out_in = np.asarray([
        [0.5, 1.0, -0.5],
        [-0.25, 0.75, 0.25],
    ], dtype=np.float32)
    tensors = {
        "weight": np.ascontiguousarray(out_in.T if layout == "din_dout" else out_in),
    }
    if bias:
        tensors["bias"] = np.asarray([0.125, -0.25], dtype=np.float32)
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def chain_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR(
        source_format="volvoxai",
        source_name="chain.json",
        dialect=IRDialect.RUNTIME,
    )
    add_tensor(graph, "x", (2, 3), public_input=True)
    add_tensor(graph, "w1", (2, 3), initializer=True)
    add_tensor(graph, "b1", (2,), initializer=True)
    add_tensor(graph, "hidden", (2, 2))
    add_tensor(graph, "w2", (2, 2), initializer=True)
    add_tensor(graph, "b2", (2,), initializer=True)
    add_tensor(graph, "y", (2, 2), public_output=True)
    graph.inputs.append("x")
    for name, input_name, weight, bias, output in (
        ("dense1", "x", "w1", "b1", "hidden"),
        ("dense2", "hidden", "w2", "b2", "y"),
    ):
        graph.add_node(OpNode.from_maps(
            name=name,
            op_type="Linear",
            inputs={"input": input_name, "weight": weight, "bias": bias},
            outputs={"out": output},
            attributes=(OpAttribute(
                "params", "volvox.params", {"weight_layout": "dout_din"},
            ),),
        ))
    graph.outputs.append("y")
    tensors = {
        "w1": np.asarray([[0.5, 1.0, -0.5], [-0.25, 0.75, 0.25]], dtype=np.float32),
        "b1": np.asarray([0.125, -0.25], dtype=np.float32),
        "w2": np.asarray([[0.75, -0.25], [0.5, 1.0]], dtype=np.float32),
        "b2": np.asarray([0.0, 0.125], dtype=np.float32),
    }
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def mixed_chain_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph, tensors = chain_graph()
    graph.source_name = "mixed-chain.json"
    add_tensor(graph, "activated", (2, 2))
    dense1, dense2 = graph.nodes
    gelu = OpNode.from_maps(
        "gelu", "GELU", {"input": "hidden"}, {"out": "activated"},
        attributes=(OpAttribute(
            "params", "volvox.params", {"approximate": "none"},
        ),),
    )
    dense2 = OpNode.from_maps(
        name=dense2.name,
        op_type=dense2.op_type,
        inputs={
            "input": "activated",
            "weight": dense2.input_map()["weight"],
            "bias": dense2.input_map()["bias"],
        },
        outputs=dense2.output_map(),
        attributes=dense2.attributes,
    )
    graph.nodes[:] = [dense1, gelu, dense2]
    graph.invalidate_analyses()
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def matmul_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR(
        source_format="volvoxai",
        source_name="matmul.json",
        dialect=IRDialect.RUNTIME,
    )
    add_tensor(graph, "a", (1, 2), public_input=True)
    add_tensor(graph, "b", (2, 3), initializer=True)
    add_tensor(graph, "y", (1, 3), public_output=True)
    graph.inputs.append("a")
    graph.add_node(OpNode.from_maps(
        "matmul", "MatMul", {"a": "a", "b": "b"}, {"out": "y"},
    ))
    graph.outputs.append("y")
    tensors = {
        "b": np.asarray([[0.5, -0.25, 1.0], [1.0, 0.75, -0.5]], dtype=np.float32),
    }
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def pointwise_norm_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR(
        source_format="volvoxai",
        source_name="pointwise-norm.json",
        dialect=IRDialect.RUNTIME,
    )
    add_tensor(graph, "x", (2, 4), public_input=True)
    add_tensor(graph, "residual", (2, 4), public_input=True)
    add_tensor(graph, "added", (2, 4))
    add_tensor(graph, "gamma", (4,), initializer=True)
    add_tensor(graph, "beta", (4,), initializer=True)
    add_tensor(graph, "normalized", (2, 4))
    add_tensor(graph, "gelu", (2, 4))
    add_tensor(graph, "y", (2, 4), public_output=True)
    graph.inputs.extend(("x", "residual"))
    graph.add_node(OpNode.from_maps(
        "add", "Add", {"a": "x", "b": "residual"}, {"out": "added"},
    ))
    graph.add_node(OpNode.from_maps(
        "norm", "LayerNorm",
        {"input": "added", "weight": "gamma", "bias": "beta"},
        {"out": "normalized"},
        attributes=(OpAttribute(
            "params", "volvox.params", {"d_model": 4, "eps": 1e-5},
        ),),
    ))
    graph.add_node(OpNode.from_maps(
        "gelu-op", "GELU", {"input": "normalized"}, {"out": "gelu"},
        attributes=(OpAttribute(
            "params", "volvox.params", {"approximate": "none"},
        ),),
    ))
    graph.add_node(OpNode.from_maps(
        "silu", "SiLU", {"input": "gelu"}, {"out": "y"},
    ))
    graph.outputs.append("y")
    tensors = {
        "gamma": np.asarray([1.0, 0.75, 1.25, 0.5], dtype=np.float32),
        "beta": np.asarray([0.0, 0.125, -0.25, 0.25], dtype=np.float32),
    }
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def groupnorm_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR(
        source_format="volvoxai",
        source_name="groupnorm.json",
        dialect=IRDialect.RUNTIME,
    )
    add_tensor(graph, "x", (1, 2, 2, 4), public_input=True)
    add_tensor(graph, "gamma", (4,), initializer=True)
    add_tensor(graph, "beta", (4,), initializer=True)
    add_tensor(graph, "y", (1, 2, 2, 4), public_output=True)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "group-norm", "GroupNorm",
        {"input": "x", "weight": "gamma", "bias": "beta"},
        {"out": "y"},
        attributes=(OpAttribute(
            "params", "volvox.params",
            {"num_groups": 2, "eps": 1e-5, "data_layout": "NHWC"},
        ),),
    ))
    graph.outputs.append("y")
    tensors = {
        "gamma": np.asarray([1.0, 0.5, 1.25, 0.75], dtype=np.float32),
        "beta": np.asarray([0.0, 0.125, -0.25, 0.25], dtype=np.float32),
    }
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def constant_add_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR(
        source_format="volvoxai",
        source_name="constant-add.json",
        dialect=IRDialect.RUNTIME,
    )
    add_tensor(graph, "x", (2, 4), public_input=True)
    add_tensor(graph, "offset", (4,), initializer=True)
    add_tensor(graph, "y", (2, 4), public_output=True)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "add", "Add", {"a": "x", "b": "offset"}, {"out": "y"},
    ))
    graph.outputs.append("y")
    tensors = {
        "offset": np.asarray([0.25, -0.5, 0.75, -1.0], dtype=np.float32),
    }
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def dynamic_broadcast_add_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR(
        source_format="volvoxai",
        source_name="dynamic-broadcast-add.json",
        dialect=IRDialect.RUNTIME,
    )
    add_tensor(graph, "x", (1, 2, 4), public_input=True)
    add_tensor(graph, "route", (1, 1, 4), public_input=True)
    add_tensor(graph, "sum", (1, 2, 4))
    add_tensor(graph, "y", (1, 2, 4), public_output=True)
    graph.inputs.extend(("x", "route"))
    graph.add_node(OpNode.from_maps(
        "broadcast-add", "Add", {"a": "x", "b": "route"}, {"out": "sum"},
    ))
    graph.add_node(OpNode.from_maps(
        "gelu", "GELU", {"input": "sum"}, {"out": "y"},
        attributes=(OpAttribute(
            "params", "volvox.params", {"approximate": "none"},
        ),),
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph, {}


def batch_matmul_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR("volvoxai", "batch-matmul.json", IRDialect.RUNTIME)
    add_tensor(graph, "a", (2, 2, 3), public_input=True)
    add_tensor(graph, "b", (1, 3, 2), public_input=True)
    add_tensor(graph, "y", (2, 2, 2), public_output=True)
    graph.inputs.extend(("a", "b"))
    graph.add_node(OpNode.from_maps(
        "batch", "BatchMatMul", {"a": "a", "b": "b"}, {"out": "y"},
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph, {}


def conv_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR("volvoxai", "conv.json", IRDialect.RUNTIME)
    add_tensor(graph, "x", (1, 3, 3, 2), public_input=True)
    add_tensor(graph, "weight", (2, 2, 2, 1), initializer=True)
    add_tensor(graph, "y", (1, 2, 2, 2), public_output=True)
    graph.inputs.append("x")
    graph.add_node(OpNode.from_maps(
        "conv", "Conv2D", {"input": "x", "weight": "weight"}, {"out": "y"},
        attributes=(OpAttribute("params", "volvox.params", {
            "stride": [1, 1], "dilation": [1, 1], "groups": 2,
            "pads": [0, 0, 0, 0], "data_layout": "NHWC",
            "weight_layout": "OHWI", "relu": 0,
        }),),
    ))
    graph.outputs.append("y")
    tensors = {"weight": np.asarray([
        [[[0.5], [-0.25]], [[0.75], [0.125]]],
        [[[-0.5], [0.25]], [[0.125], [0.875]]],
    ], dtype=np.float32)}
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def embedding_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR("volvoxai", "embedding.json", IRDialect.RUNTIME)
    add_tensor(graph, "ids", (2, 2), public_input=True, dtype="int32")
    add_tensor(graph, "table", (5, 4), initializer=True)
    add_tensor(graph, "y", (2, 2, 4), public_output=True)
    graph.inputs.append("ids")
    graph.add_node(OpNode.from_maps(
        "embedding", "Embedding",
        {"input": "ids", "weight": "table"}, {"out": "y"},
    ))
    graph.outputs.append("y")
    tensors = {"table": np.asarray([
        [-1.0, -0.5, 0.0, 0.5], [0.25, 0.75, 1.0, -0.25],
        [1.25, -1.25, 0.5, -0.75], [0.0, 0.125, -0.125, 0.25],
        [3.0, -2.5, 2.0, -1.5],
    ], dtype=np.float32)}
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def clipped_embedding_graph(
    *, maximum: int = 4,
) -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph, tensors = embedding_graph()
    add_tensor(graph, "bounded_ids", (2, 2), dtype="int32")
    embedding = graph.nodes[0]
    graph.nodes[:] = [
        OpNode.from_maps(
            "bound_ids", "Clip", {"input": "ids"}, {"out": "bounded_ids"},
            attributes=(OpAttribute(
                "params", "volvox.params", {"min": 0, "max": maximum},
            ),),
        ),
        OpNode.from_maps(
            embedding.name, embedding.op_type,
            {"input": "bounded_ids", "weight": "table"},
            embedding.output_map(),
        ),
    ]
    graph.invalidate_analyses()
    graph.verify(IRDialect.RUNTIME)
    return graph, tensors


def attention_graph() -> tuple[GraphIR, dict[str, np.ndarray]]:
    graph = GraphIR("volvoxai", "attention.json", IRDialect.RUNTIME)
    add_tensor(graph, "q", (1, 2, 4), public_input=True)
    add_tensor(graph, "k", (1, 3, 4), public_input=True)
    add_tensor(graph, "v", (1, 3, 4), public_input=True)
    add_tensor(graph, "mask", (1, 3), public_input=True, dtype="int32")
    add_tensor(graph, "y", (1, 2, 4), public_output=True)
    graph.inputs.extend(("q", "k", "v", "mask"))
    graph.add_node(OpNode.from_maps(
        "attention", "CrossSDPA",
        {"q": "q", "k": "k", "v": "v", "mask": "mask"}, {"out": "y"},
        attributes=(OpAttribute("params", "volvox.params", {
            "heads": 1, "causal": False, "scale": 0.5,
        }),),
    ))
    graph.outputs.append("y")
    graph.verify(IRDialect.RUNTIME)
    return graph, {}


def calibrate(
    graph: GraphIR,
    tensors: dict[str, np.ndarray],
    samples: tuple[np.ndarray, ...],
):
    table = CalibrationTable(graph)
    input_name = graph.inputs[0]
    executions = []
    for sample in samples:
        execution = execute_reference(graph, tensors, {input_name: sample})
        table.observe_reference(execution)
        executions.append(execution)
    return table.profile(), executions


def calibrate_inputs(
    graph: GraphIR,
    tensors: dict[str, np.ndarray],
    samples: tuple[dict[str, np.ndarray], ...],
    *,
    config: PTQConfig = PTQConfig(),
):
    table = CalibrationTable(graph, config=config)
    executions = []
    for sample in samples:
        execution = execute_reference(graph, tensors, sample)
        table.observe_reference(execution)
        executions.append(execution)
    return table.profile(), executions


class TypedPTQTests(unittest.TestCase):
    def test_activation_scheme_defaults_are_explicit_and_validated(self):
        self.assertEqual(PTQConfig().activation_scheme, "symmetric")
        self.assertEqual(PTQConfig("uint8").activation_scheme, "asymmetric")
        self.assertEqual(
            PTQConfig(
                activation_dtype="int8", activation_scheme="asymmetric",
            ).activation_scheme,
            "asymmetric",
        )
        self.assertEqual(
            PTQConfig(
                activation_dtype="uint8", activation_scheme="symmetric",
            ).activation_scheme,
            "symmetric",
        )
        with self.assertRaisesRegex(ValueError, "activation_scheme"):
            PTQConfig(activation_scheme="unknown")

    def test_reduce_range_is_off_by_default_and_bounds_authored_weights(self):
        """A backend can only prove VPMADDUBSW is saturation-free when every
        weight satisfies |w| <= 64, since 255 * 64 * 2 stays inside I16.  The
        option that guarantees it costs a bit of weight precision, so the
        default has to stay the full signed range."""
        self.assertFalse(PTQConfig().reduce_range)
        self.assertTrue(PTQConfig(reduce_range=True).reduce_range)
        for value in (0, 1, None, "false", np.bool_(True)):
            with self.subTest(invalid_reduce_range=value):
                with self.assertRaisesRegex(TypeError, "reduce_range"):
                    PTQConfig(reduce_range=value)

        def authored_weight(reduce_range):
            graph, tensors = linear_graph()
            table = CalibrationTable(graph)
            table.observe({
                "x": np.asarray(
                    [[-1.0, 0.0, 3.0], [0.0, 1.0, 2.0]], dtype=np.float32,
                ),
                "y": np.asarray(
                    [[-1.0, 0.0], [1.0, 3.0]], dtype=np.float32,
                ),
            })
            plan = plan_runtime_ptq(
                graph, tensors, table.profile(),
                config=PTQConfig(reduce_range=reduce_range),
            )
            weight_name = plan.nodes[0].quantized_weight.quantized_tensor
            materialize_runtime_ptq(graph, tensors, plan)
            return np.asarray(tensors[weight_name])

        full = authored_weight(False)
        reduced = authored_weight(True)
        self.assertEqual(int(np.abs(full.astype(np.int16)).max()), 127)
        self.assertEqual(int(np.abs(reduced.astype(np.int16)).max()), 64)

    def test_reduce_range_materializes_conv_weights_including_subnormals(self):
        cases = (
            (None, np.linspace(-1.0, 1.0, 18, dtype=np.float32)),
            (
                np.float32(1.0229478789571165e-43),
                np.linspace(-1e30, 1e30, 18, dtype=np.float32),
            ),
        )
        for weight_value, sample in cases:
            with self.subTest(weight_value=weight_value):
                graph, tensors = conv_graph()
                if weight_value is not None:
                    tensors["weight"].fill(weight_value)
                calibration, _ = calibrate(
                    graph, tensors, (sample.reshape(1, 3, 3, 2),),
                )
                plan = plan_runtime_ptq(
                    graph, tensors, calibration,
                    config=PTQConfig(reduce_range=True),
                )
                weight_plan = plan.conv_nodes[0].quantized_weight
                if weight_value is not None:
                    self.assertGreater(weight_plan.saturation_count, 0)
                materialize_runtime_ptq(graph, tensors, plan)
                packed = np.asarray(tensors[weight_plan.quantized_tensor])
                self.assertEqual(
                    int(np.abs(packed.astype(np.int16)).max()), 64,
                )
                descriptor = graph.tensors[weight_plan.quantized_tensor]
                self.assertEqual(descriptor.quantization.axis, 0)
                np.testing.assert_array_equal(
                    tensors[weight_plan.zero_point_tensor],
                    np.zeros(len(weight_plan.zero_points), dtype=np.int8),
                )

    def test_reduce_range_bounds_embedding_weights(self):
        graph, tensors = embedding_graph()
        plan = plan_runtime_ptq(
            graph, tensors, CalibrationTable(graph).profile(),
            config=PTQConfig(reduce_range=True),
        )
        weight_plan = plan.embedding_nodes[0].quantized_weight
        materialize_runtime_ptq(graph, tensors, plan)
        packed = np.asarray(tensors[weight_plan.quantized_tensor])
        self.assertEqual(int(np.abs(packed.astype(np.int16)).max()), 64)
        descriptor = graph.tensors[weight_plan.quantized_tensor]
        self.assertEqual(descriptor.quantization.axis, 0)
        np.testing.assert_array_equal(
            tensors[weight_plan.zero_point_tensor],
            np.zeros(len(weight_plan.zero_points), dtype=np.int8),
        )

    def test_int8_asymmetric_uses_full_signed_affine_domain(self):
        graph, tensors = linear_graph()
        table = CalibrationTable(graph)
        table.observe({
            "x": np.asarray(
                [[-1.0, 0.0, 3.0], [0.0, 1.0, 2.0]], dtype=np.float32,
            ),
            "y": np.asarray(
                [[-1.0, 0.0], [1.0, 3.0]], dtype=np.float32,
            ),
        })
        config = PTQConfig(
            activation_dtype="int8", activation_scheme="asymmetric",
        )
        plan = plan_runtime_ptq(
            graph, tensors, table.profile(), config=config,
        )
        activation = plan.activation("x")
        self.assertEqual(activation.dtype, "int8")
        self.assertEqual(activation.zero_point, -64)
        self.assertAlmostEqual(activation.scale, 4.0 / 255.0, places=7)

        materialize_runtime_ptq(graph, tensors, plan)
        document, packaged = export_runtime_package(graph, tensors)
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        quantized_input = graph.tensors[activation.quantized_tensor]
        zero = packaged[quantized_input.quantization.zero_point]
        np.testing.assert_array_equal(zero, np.asarray([-64], dtype=np.int8))

    def test_ptq_preserves_opaque_public_input_without_coverage_policy(self):
        graph, tensors = linear_graph()
        add_tensor(
            graph,
            "route_ids",
            (2,),
            public_input=True,
            dtype="int32",
        )
        graph.inputs.append("route_ids")
        graph.verify(IRDialect.RUNTIME)
        before_document, _ = export_runtime_package(graph, tensors)

        table = CalibrationTable(graph)
        table.observe({
            "x": np.asarray(
                [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]],
                dtype=np.float32,
            ),
            "y": np.asarray(
                [[1.625, -1.875], [-0.125, 0.875]],
                dtype=np.float32,
            ),
        })
        materialize_runtime_ptq(
            graph,
            tensors,
            plan_runtime_ptq(graph, tensors, table.profile()),
        )
        after_document, _ = export_runtime_package(graph, tensors)

        self.assertEqual(after_document["inputs"], before_document["inputs"])
        self.assertEqual(after_document["outputs"], before_document["outputs"])
        self.assertEqual(graph.inputs, ["x", "route_ids"])
        self.assertEqual(graph.tensors["route_ids"].dtype, "int32")

    def test_aggregate_range_profile_is_graph_bound_counted_and_alias_strict(self):
        graph, _ = linear_graph()
        digest = "a" * 64
        profile = calibration_profile_from_ranges(
            graph,
            {
                "observed_x": {"min": -2.0, "max": 1.0},
                "y": {"min": -1.0, "max": 2.0},
            },
            sample_count=3,
            sample_digest=digest,
            aliases={"x": "observed_x"},
        )
        self.assertEqual(profile.graph_fingerprint, graph.fingerprint())
        self.assertEqual(profile.sample_count, 3)
        self.assertEqual(profile.sample_digest, digest)
        self.assertEqual(profile.observation("x").samples, 3)
        self.assertEqual(profile.observation("x").elements, 18)
        with self.assertRaises(ExporterError) as caught:
            calibration_profile_from_ranges(
                graph,
                {"x": {"min": -1.0, "max": 1.0}},
                sample_count=1,
                sample_digest=digest,
            )
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ094")

    def test_symbolic_ptq_preserves_bounds_across_weighted_and_byte_islands(self):
        cases = []

        graph, tensors = linear_graph()
        graph.shape_environment = ShapeEnvironment((
            {"name": "B", "min": 1, "max": 4},
        ))
        graph.tensors["x"].shape = ("B", 3)
        graph.tensors["y"].shape = ("B", 2)
        cases.append(("linear", graph, tensors, {"B": 2}))

        graph, tensors = conv_graph()
        graph.shape_environment = ShapeEnvironment((
            {"name": "B", "min": 1, "max": 4},
        ))
        graph.tensors["x"].shape = ("B", 3, 3, 2)
        graph.tensors["y"].shape = ("B", 2, 2, 2)
        cases.append(("conv", graph, tensors, {"B": 2}))

        graph, tensors = embedding_graph()
        graph.shape_environment = ShapeEnvironment((
            {"name": "B", "min": 1, "max": 4},
            {"name": "S", "min": 1, "max": 8},
        ))
        graph.tensors["ids"].shape = ("B", "S")
        graph.tensors["y"].shape = ("B", "S", 4)
        cases.append(("embedding", graph, tensors, {"B": 2, "S": 3}))

        graph, tensors = pointwise_norm_graph()
        graph.shape_environment = ShapeEnvironment((
            {"name": "B", "min": 1, "max": 4},
            {"name": "S", "min": 1, "max": 8},
        ))
        for name in ("x", "residual", "added", "normalized", "gelu", "y"):
            graph.tensors[name].shape = ("B", "S", 4)
        cases.append(("pointwise", graph, tensors, {"B": 2, "S": 3}))

        graph, tensors = batch_matmul_graph()
        graph.shape_environment = ShapeEnvironment((
            {"name": "B", "min": 1, "max": 4},
            {"name": "S", "min": 1, "max": 8},
        ))
        graph.tensors["a"].shape = ("B", "S", 3)
        graph.tensors["b"].shape = (1, 3, 2)
        graph.tensors["y"].shape = ("B", "S", 2)
        cases.append(("batch-matmul", graph, tensors, {"B": 2, "S": 3}))

        graph, tensors = attention_graph()
        graph.shape_environment = ShapeEnvironment((
            {"name": "B", "min": 1, "max": 4},
            {"name": "Q", "min": 1, "max": 8},
            {"name": "K", "min": 1, "max": 12},
        ))
        graph.tensors["q"].shape = ("B", "Q", 4)
        graph.tensors["k"].shape = ("B", "K", 4)
        graph.tensors["v"].shape = ("B", "K", 4)
        graph.tensors["mask"].shape = ("B", "K")
        graph.tensors["y"].shape = ("B", "Q", 4)
        cases.append(("attention", graph, tensors, {"B": 2, "Q": 3, "K": 5}))

        for label, graph, tensors, profile in cases:
            with self.subTest(label=label):
                graph.verify(IRDialect.RUNTIME)
                logical_shapes = {
                    name: tensor.shape for name, tensor in graph.tensors.items()
                }
                ranges = {}
                for name in required_ptq_observations(graph):
                    shape = tuple(
                        profile[dimension] if isinstance(dimension, str) else dimension
                        for dimension in graph.tensors[name].shape
                    )
                    ranges[name] = {
                        "min": -1.0,
                        "max": 1.0,
                        "samples": 1,
                        "elements": int(np.prod(shape)),
                    }
                calibration = calibration_profile_from_ranges(
                    graph,
                    ranges,
                    sample_count=1,
                    sample_digest="b" * 64,
                )
                plan = plan_runtime_ptq(graph, tensors, calibration)
                materialize_runtime_ptq(graph, tensors, plan)
                graph.verify(IRDialect.RUNTIME)
                self.assertTrue(graph.shape_environment.dimensions)
                for name in graph.inputs + graph.outputs:
                    self.assertEqual(graph.tensors[name].shape, logical_shapes[name])
                self.assertTrue(any(
                    node.op_type.startswith("Q") for node in graph.nodes
                ))

        graph, _ = linear_graph()
        graph.shape_environment = ShapeEnvironment((
            {"name": "B", "min": 1, "max": 4},
        ))
        graph.tensors["x"].shape = ("B", 3)
        graph.tensors["y"].shape = ("B", 2)
        graph.verify(IRDialect.RUNTIME)
        with self.assertRaises(ExporterError) as caught:
            calibration_profile_from_ranges(
                graph,
                {name: {"min": -1.0, "max": 1.0}
                 for name in required_ptq_observations(graph)},
                sample_count=1,
                sample_digest="c" * 64,
            )
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ098")

    def test_symbolic_aggregate_element_counts_must_fit_bounded_totals(self):
        graph, _ = linear_graph()
        graph.shape_environment = ShapeEnvironment((
            {"name": "B", "min": 1, "max": 4},
        ))
        graph.tensors["x"].shape = ("B", 3)
        graph.tensors["y"].shape = ("B", 2)
        graph.verify(IRDialect.RUNTIME)
        valid = {
            "x": {
                "min": -1.0, "max": 1.0, "samples": 2, "elements": 12,
            },
            "y": {
                "min": -1.0, "max": 1.0, "samples": 2, "elements": 8,
            },
        }

        for label, elements in (("below-minimum", 5), ("above-maximum", 25)):
            with self.subTest(label=label):
                ranges = {
                    name: dict(bounds) for name, bounds in valid.items()
                }
                ranges["x"]["elements"] = elements
                with self.assertRaises(ExporterError) as caught:
                    calibration_profile_from_ranges(
                        graph,
                        ranges,
                        sample_count=2,
                        sample_digest="e" * 64,
                    )
                self.assertEqual(caught.exception.diagnostic.code, "VXPTQ098")
                self.assertIn(
                    "outside bounded total [6, 24]", str(caught.exception),
                )

    def test_default_calibration_requests_only_exact_ptq_demand(self):
        graph, _ = linear_graph()
        add_tensor(graph, "attention_mask", (2, 2), public_input=True)
        graph.inputs.append("attention_mask")
        graph.verify(IRDialect.RUNTIME)

        table = CalibrationTable(graph)
        self.assertEqual(table.tensor_names, ("x", "y"))
        table.observe({
            "x": np.ones((2, 3), dtype=np.float32),
            "y": np.zeros((2, 2), dtype=np.float32),
            "attention_mask": np.asarray(
                [[0.0, -np.inf], [0.0, 0.0]], dtype=np.float32,
            ),
        })
        self.assertEqual(
            tuple(name for name, _ in table.profile().observations),
            ("x", "y"),
        )

        explicit = CalibrationTable(graph, ("attention_mask",))
        with self.assertRaises(ExporterError) as caught:
            explicit.observe({
                "attention_mask": np.asarray(
                    [[0.0, -np.inf], [0.0, 0.0]], dtype=np.float32,
                ),
            })
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ007")

    def test_calibration_accumulates_deterministically_and_rejects_nonfinite(self):
        graph, tensors = linear_graph()
        first = np.asarray([[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32)
        second = np.asarray([[-3.0, 0.0, 1.0], [1.5, -1.0, 0.25]], dtype=np.float32)
        profile, _ = calibrate(graph, tensors, (first, second))
        self.assertEqual(profile.sample_count, 2)
        self.assertEqual(profile.observation("x").minimum, -3.0)
        self.assertEqual(profile.observation("x").maximum, 2.0)
        self.assertEqual(profile.observation("x").elements, 12)

        again, _ = calibrate(graph, tensors, (first, second))
        self.assertEqual(profile, again)

        table = CalibrationTable(graph, ("x", "y"))
        valid = execute_reference(graph, tensors, {"x": first})
        table.observe_reference(valid)
        broken = dict(valid.tensors)
        broken["x"] = first.copy()
        broken["x"][0, 0] = np.nan
        with self.assertRaises(ExporterError) as caught:
            table.observe(broken)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ007")
        self.assertEqual(table.profile().sample_count, 1)

    def test_linear_materializes_only_central_safetensor_refs_and_matches_oracle(self):
        graph, tensors = linear_graph(layout="din_dout", bias=True)
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        calibration, _ = calibrate(graph, tensors, (sample, -sample))
        heldout = np.multiply(sample, np.float32(0.5), dtype=np.float32)
        expected = execute_reference(graph, tensors, {"x": heldout})
        plan = plan_runtime_ptq(graph, tensors, calibration)
        duplicate = plan_runtime_ptq(graph.clone(), dict(tensors), calibration)
        self.assertEqual(plan, duplicate)
        self.assertEqual(plan.nodes[0].source_layout, "din_dout")
        self.assertEqual(plan.nodes[0].quantized_weight.saturation_count, 0)

        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(report.nodes_quantized, 1)
        self.assertEqual(report.quantize_boundaries, 1)
        self.assertEqual(report.dequantize_boundaries, 1)
        self.assertNotIn("weight", tensors)
        self.assertNotIn("bias", tensors)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QLinear", "DequantizeLinear"],
        )
        qlinear = graph.nodes[1]
        self.assertEqual(set(qlinear.input_map()), {"input", "weight", "bias"})
        self.assertFalse(qlinear.attributes)
        qweight = graph.tensors[qlinear.input_map()["weight"]]
        self.assertEqual(qweight.shape, (2, 3))
        self.assertEqual(qweight.dtype, "int8")
        self.assertEqual(qweight.quantization.axis, 0)
        self.assertEqual(tensors[qlinear.input_map()["bias"]].dtype, np.int32)

        document, packaged = export_runtime_package(graph, tensors)
        self.assertEqual(document["format"], "volvox-graph/v1")
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        self.assertNotIn("weights_quantization", document)
        self.assertTrue(all("outputs_quantization" not in node
                            for node in document["nodes"]))
        for descriptor in document["quantization"]["tensors"].values():
            self.assertIsInstance(descriptor["scale_tensor"], str)
            self.assertIsInstance(descriptor["zero_point_tensor"], str)
            self.assertIn(descriptor["scale_tensor"], packaged)
            self.assertIn(descriptor["zero_point_tensor"], packaged)
            self.assertNotIn("scales", descriptor)
            self.assertNotIn("zero_points", descriptor)

        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "model.safetensors"
            write_safetensors(weights_path, packaged)
            on_disk = read_safetensors(weights_path)
            for descriptor in document["quantization"]["tensors"].values():
                self.assertEqual(
                    on_disk[descriptor["scale_tensor"]].dtype, np.float32,
                )
                self.assertIn(
                    on_disk[descriptor["zero_point_tensor"]].dtype,
                    (np.dtype(np.int8), np.dtype(np.uint8)),
                )

        restored = import_runtime_package(document, packaged)
        restored.verify(IRDialect.RUNTIME)
        actual = execute_reference(graph, tensors, {"x": heldout})
        comparison = compare_tensor_maps(
            expected.outputs,
            actual.outputs,
            tensor_order=("y",),
            atol=0.04,
            rtol=0.0,
        )
        comparison.raise_for_divergence()

    def test_adjacent_dense_nodes_form_one_byte_island(self):
        graph, tensors = chain_graph()
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        calibration, executions = calibrate(graph, tensors, (sample,))
        expected = executions[0]
        plan = plan_runtime_ptq(graph, tensors, calibration)
        report = materialize_runtime_ptq(graph, tensors, plan)

        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QLinear", "QLinear", "DequantizeLinear"],
        )
        self.assertEqual(report.quantize_boundaries, 1)
        self.assertEqual(report.dequantize_boundaries, 1)
        self.assertEqual(report.byte_edges_reused, 1)
        self.assertNotIn("hidden", graph.tensors)
        self.assertEqual(
            graph.nodes[1].output_map()["out"],
            graph.nodes[2].input_map()["input"],
        )

        actual = execute_reference(graph, tensors, {"x": sample})
        compare_tensor_maps(
            expected.outputs, actual.outputs,
            tensor_order=("y",), atol=0.06, rtol=0.0,
        ).raise_for_divergence()

    def test_dense_and_pointwise_nodes_share_the_same_byte_island(self):
        graph, tensors = mixed_chain_graph()
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        calibration, executions = calibrate(graph, tensors, (sample, -sample))
        plan = plan_runtime_ptq(graph, tensors, calibration)
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            [
                "QuantizeLinear", "QLinear", "QGELU", "QLinear",
                "DequantizeLinear",
            ],
        )
        self.assertEqual(report.quantize_boundaries, 1)
        self.assertEqual(report.dequantize_boundaries, 1)
        self.assertEqual(report.byte_edges_reused, 2)
        first, gelu, second = (
            next(node for node in graph.nodes if node.name == name)
            for name in ("dense1", "gelu", "dense2")
        )
        self.assertEqual(first.output_map()["out"], gelu.input_map()["input"])
        self.assertEqual(gelu.output_map()["out"], second.input_map()["input"])
        actual = execute_reference(graph, tensors, {"x": sample})
        compare_tensor_maps(
            executions[0].outputs, actual.outputs,
            tensor_order=("y",), atol=0.08, rtol=0.0,
        ).raise_for_divergence()

    def test_biasless_out_in_linear_and_static_rhs_matmul_synthesize_i32_bias(self):
        cases = [
            (*linear_graph(layout="dout_din", bias=False), "x", PTQConfig("uint8")),
            (*matmul_graph(), "a", PTQConfig("int8")),
        ]
        samples = {
            "x": np.asarray(
                [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
            ),
            "a": np.asarray([[1.0, -2.0]], dtype=np.float32),
        }
        for graph, tensors, input_name, config in cases:
            with self.subTest(source=graph.source_name):
                calibration, executions = calibrate(
                    graph, tensors, (samples[input_name],),
                )
                plan = plan_runtime_ptq(
                    graph, tensors, calibration, config=config,
                )
                self.assertIsNone(plan.nodes[0].source_bias)
                materialize_runtime_ptq(graph, tensors, plan)
                qlinear = next(node for node in graph.nodes if node.op_type == "QLinear")
                self.assertEqual(
                    graph.tensors[qlinear.input_map()["input"]].dtype,
                    config.activation_dtype,
                )
                bias = tensors[qlinear.input_map()["bias"]]
                np.testing.assert_array_equal(
                    bias, np.zeros((bias.shape[0],), dtype=np.int32),
                )
                actual = execute_reference(
                    graph, tensors, {input_name: samples[input_name]},
                )
                compare_tensor_maps(
                    executions[0].outputs,
                    actual.outputs,
                    atol=0.04,
                    rtol=0.0,
                ).raise_for_divergence()

    def test_pointwise_and_layernorm_share_one_calibrated_byte_island(self):
        graph, tensors = pointwise_norm_graph()
        first = {
            "x": np.asarray(
                [[-1.5, 0.25, 1.0, 2.0], [0.5, -0.75, 1.5, -2.0]],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                [[0.25, -0.5, 0.75, -1.0], [-0.25, 0.5, -0.75, 1.0]],
                dtype=np.float32,
            ),
        }
        second = {
            name: np.negative(value, dtype=np.float32)
            for name, value in first.items()
        }
        calibration, executions = calibrate_inputs(
            graph, tensors, (first, second),
        )
        expected = executions[0]
        self.assertEqual(
            tuple(name for name, _ in calibration.observations),
            ("x", "residual", "added", "normalized", "gelu", "y"),
        )

        plan = plan_runtime_ptq(graph, tensors, calibration)
        self.assertEqual(
            tuple(node.quantized_op for node in plan.byte_nodes),
            ("QAdd", "QLayerNorm", "QGELU", "QSiLU"),
        )
        self.assertEqual(plan.nodes, ())
        self.assertEqual(plan, plan_runtime_ptq(
            graph.clone(), dict(tensors), calibration,
        ))
        self.assertIsInstance(plan.byte_nodes[0].params, tuple)

        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            [
                "QuantizeLinear", "QuantizeLinear", "QAdd", "QLayerNorm",
                "QGELU", "QSiLU", "DequantizeLinear",
            ],
        )
        self.assertEqual(report.nodes_quantized, 4)
        self.assertEqual(report.quantize_boundaries, 2)
        self.assertEqual(report.dequantize_boundaries, 1)
        self.assertEqual(report.byte_edges_reused, 3)
        qadd = next(node for node in graph.nodes if node.op_type == "QAdd")
        qnorm = next(node for node in graph.nodes if node.op_type == "QLayerNorm")
        qgelu = next(node for node in graph.nodes if node.op_type == "QGELU")
        qsilu = next(node for node in graph.nodes if node.op_type == "QSiLU")
        self.assertEqual(qadd.output_map()["out"], qnorm.input_map()["input"])
        self.assertEqual(qnorm.output_map()["out"], qgelu.input_map()["input"])
        self.assertEqual(qgelu.output_map()["out"], qsilu.input_map()["input"])
        self.assertNotIn("added", graph.tensors)
        self.assertNotIn("normalized", graph.tensors)
        self.assertNotIn("gelu", graph.tensors)
        self.assertIn("gamma", tensors)
        self.assertIn("beta", tensors)

        document, packaged = export_runtime_package(graph, tensors)
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        actual = execute_reference(graph, tensors, first)
        compare_tensor_maps(
            expected.outputs, actual.outputs,
            tensor_order=("y",), atol=0.12, rtol=0.0,
        ).raise_for_divergence()

    def test_groupnorm_materialization_is_differential_and_package_valid(self):
        graph, tensors = groupnorm_graph()
        config = PTQConfig("uint8")
        sample = {
            "x": np.asarray(
                [[[[0.0, 0.5, -1.0, 1.5], [1.0, -0.5, 0.25, -1.5]],
                  [[-0.75, 1.25, 0.75, -0.25], [1.5, -1.0, 0.5, 0.0]]]],
                dtype=np.float32,
            ),
        }
        calibration, executions = calibrate_inputs(
            graph, tensors, (sample,), config=config,
        )
        plan = plan_runtime_ptq(graph, tensors, calibration, config=config)
        self.assertEqual(plan.byte_nodes[0].quantized_op, "QGroupNorm")
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QGroupNorm", "DequantizeLinear"],
        )
        qgroupnorm = graph.nodes[1]
        self.assertEqual(
            graph.tensors[qgroupnorm.output_map()["out"]].dtype, "uint8",
        )
        self.assertEqual(report.nodes_quantized, 1)
        document, packaged = export_runtime_package(graph, tensors)
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        actual = execute_reference(graph, tensors, sample)
        compare_tensor_maps(
            executions[0].outputs, actual.outputs,
            tensor_order=("y",), atol=0.08, rtol=0.0,
        ).raise_for_divergence()

    def test_add_constant_is_broadcast_once_and_reuses_output_affine(self):
        graph, tensors = constant_add_graph()
        sample = np.asarray(
            [[-1.0, 0.0, 1.0, 2.0], [0.5, -0.5, 1.5, -1.5]],
            dtype=np.float32,
        )
        calibration, executions = calibrate(graph, tensors, (sample,))
        self.assertEqual(
            tuple(name for name, _ in calibration.observations),
            ("x", "y"),
        )
        plan = plan_runtime_ptq(graph, tensors, calibration)
        self.assertEqual(len(plan.byte_nodes[0].constant_inputs), 1)
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QAdd", "DequantizeLinear"],
        )
        qadd = graph.nodes[1]
        constant_name = qadd.input_map()["b"]
        output_name = qadd.output_map()["out"]
        self.assertEqual(graph.tensors[constant_name].shape, (2, 4))
        self.assertTrue(graph.tensors[constant_name].initializer)
        self.assertEqual(
            graph.tensors[constant_name].quantization,
            graph.tensors[output_name].quantization,
        )
        self.assertNotIn("offset", tensors)
        self.assertEqual(report.source_initializers_removed, 1)
        document, packaged = export_runtime_package(graph, tensors)
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        actual = execute_reference(graph, tensors, {"x": sample})
        compare_tensor_maps(
            executions[0].outputs, actual.outputs,
            tensor_order=("y",), atol=0.04, rtol=0.0,
        ).raise_for_divergence()

    def test_dynamic_broadcast_add_lowers_to_expand_plus_exact_qadd(self):
        graph, tensors = dynamic_broadcast_add_graph()
        sample = {
            "x": np.asarray(
                [[[-1.0, 0.0, 1.0, 2.0], [0.5, -0.5, 1.5, -1.5]]],
                dtype=np.float32,
            ),
            "route": np.asarray([[[0.25, -0.25, 0.5, -0.5]]], dtype=np.float32),
        }
        calibration, executions = calibrate_inputs(graph, tensors, (sample,))
        self.assertEqual(
            tuple(name for name, _ in calibration.observations),
            ("x", "route", "sum", "y"),
        )
        plan = plan_runtime_ptq(graph, tensors, calibration)
        self.assertEqual(plan.retained_nodes, ())
        self.assertEqual(len(plan.byte_nodes[0].broadcast_inputs), 1)
        self.assertEqual(plan.byte_nodes[0].broadcast_inputs[0].port, "b")
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            [
                "QuantizeLinear", "QuantizeLinear", "Expand", "QAdd",
                "QGELU", "DequantizeLinear",
            ],
        )
        self.assertEqual(report.broadcast_expands, 1)
        self.assertEqual(report.quantize_boundaries, 2)
        self.assertEqual(report.retained_nodes, ())
        expand = graph.nodes[2]
        qadd = graph.nodes[3]
        self.assertEqual(qadd.input_map()["b"], expand.output_map()["out"])
        self.assertEqual(
            graph.tensors[expand.input_map()["input"]].quantization,
            graph.tensors[expand.output_map()["out"]].quantization,
        )
        document, packaged = export_runtime_package(graph, tensors)
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        actual = execute_reference(graph, tensors, sample)
        compare_tensor_maps(
            executions[0].outputs, actual.outputs,
            tensor_order=("y",), atol=0.04, rtol=0.0,
        ).raise_for_divergence()

        strict_graph, _ = dynamic_broadcast_add_graph()
        self.assertEqual(
            CalibrationTable(
                strict_graph, selected_nodes=("broadcast-add",),
            ).tensor_names,
            ("x", "route", "sum"),
        )

    def test_unrepresentable_implicit_broadcast_cannot_be_republished(self):
        graph = GraphIR(
            "volvoxai", "rank-nine-add.json", IRDialect.RUNTIME,
        )
        output_shape = (1, 1, 1, 1, 1, 1, 1, 2, 4)
        add_tensor(graph, "x", output_shape, public_input=True)
        add_tensor(graph, "route", (4,), public_input=True)
        add_tensor(graph, "y", output_shape, public_output=True)
        graph.inputs.extend(("x", "route"))
        graph.add_node(OpNode.from_maps(
            "broadcast-add", "Add", {"a": "x", "b": "route"}, {"out": "y"},
        ))
        graph.outputs.append("y")
        graph.verify(IRDialect.RUNTIME)
        table = CalibrationTable(graph)
        self.assertEqual(table.tensor_names, ())
        plan = plan_runtime_ptq(graph, {}, table.profile())
        self.assertEqual(
            tuple(item.diagnostic_code for item in plan.retained_nodes),
            ("VXPTQ096",),
        )
        before = graph.fingerprint()
        with self.assertRaises(ExporterError) as caught:
            materialize_runtime_ptq(graph, {}, plan)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR035")
        self.assertIn("broadcasting is explicit", caught.exception.diagnostic.message)
        self.assertEqual(graph.fingerprint(), before)

        strict = graph.clone()
        with self.assertRaisesRegex(ExporterError, "rank-1..8 byte Expand"):
            CalibrationTable(strict, selected_nodes=("broadcast-add",))

    def test_generic_float_op_exclusion_controls_calibration_and_island_boundary(self):
        graph, tensors = pointwise_norm_graph()
        config = PTQConfig(float_ops=frozenset({"Add", "FutureBackendOp"}))
        table = CalibrationTable(graph, config=config)
        self.assertEqual(
            table.tensor_names,
            ("added", "normalized", "gelu", "y"),
        )
        sample = {
            "x": np.asarray(
                [[-1.0, 0.0, 1.0, 2.0], [0.5, -0.5, 1.5, -1.5]],
                dtype=np.float32,
            ),
            "residual": np.asarray(
                [[0.25, 0.5, -0.25, -0.5], [-0.25, 0.25, -0.5, 0.5]],
                dtype=np.float32,
            ),
        }
        table.observe_reference(execute_reference(graph, tensors, sample))
        plan = plan_runtime_ptq(
            graph, tensors, table.profile(),
            selected_nodes=("add", "norm", "gelu-op", "silu"),
            config=config,
        )
        self.assertEqual(plan.config.float_ops, frozenset({"Add", "FutureBackendOp"}))
        self.assertEqual(
            tuple(node.source_op for node in plan.byte_nodes),
            ("LayerNorm", "GELU", "SiLU"),
        )
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            [
                "Add", "QuantizeLinear", "QLayerNorm", "QGELU", "QSiLU",
                "DequantizeLinear",
            ],
        )
        self.assertEqual(report.quantize_boundaries, 1)
        self.assertEqual(report.byte_edges_reused, 2)

    def test_byte_island_malformed_forms_refuse_before_mutation(self):
        graph, _ = pointwise_norm_graph()
        graph.tensors["residual"].shape = (3, 4)
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ058")

        graph, _ = pointwise_norm_graph()
        gelu = next(node for node in graph.nodes if node.op_type == "GELU")
        gelu.attributes = (OpAttribute(
            "params", "volvox.params", {"approximate": "tanh"},
        ),)
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ061")

        graph, _ = pointwise_norm_graph()
        silu = next(node for node in graph.nodes if node.op_type == "SiLU")
        silu.attributes = (OpAttribute(
            "params", "volvox.params", {"alpha": 1.0},
        ),)
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ062")

        graph, _ = pointwise_norm_graph()
        norm = next(node for node in graph.nodes if node.op_type == "LayerNorm")
        norm.inputs = tuple(port for port in norm.inputs if port.name != "bias")
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ063")

        graph, _ = groupnorm_graph()
        group = graph.nodes[0]
        group.attributes = (OpAttribute(
            "params", "volvox.params",
            {"num_groups": 2, "eps": 1e-5, "data_layout": "NCHW"},
        ),)
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ071")

        graph, tensors = groupnorm_graph()
        sample = {"x": np.ones((1, 2, 2, 4), dtype=np.float32)}
        calibration, _ = calibrate_inputs(graph, tensors, (sample,))
        tensors["gamma"][0] = np.nan
        fingerprint = graph.fingerprint()
        with self.assertRaises(ExporterError) as caught:
            plan_runtime_ptq(graph, tensors, calibration)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ072")
        self.assertEqual(graph.fingerprint(), fingerprint)

    def test_norm_payload_binding_and_second_ptq_run_are_transactional_noop(self):
        graph, tensors = groupnorm_graph()
        sample = {
            "x": np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(1, 2, 2, 4),
        }
        calibration, _ = calibrate_inputs(graph, tensors, (sample,))
        plan = plan_runtime_ptq(graph, tensors, calibration)
        before = graph.fingerprint()
        tensors["gamma"][0] += np.float32(0.25)
        with self.assertRaises(ExporterError) as caught:
            materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ051")
        self.assertEqual(graph.fingerprint(), before)
        tensors["gamma"][0] -= np.float32(0.25)

        materialize_runtime_ptq(graph, tensors, plan)
        quantized_fingerprint = graph.fingerprint()
        tensor_snapshot = {name: np.array(value, copy=True) for name, value in tensors.items()}
        empty_table = CalibrationTable(graph)
        self.assertEqual(empty_table.tensor_names, ())
        second_plan = plan_runtime_ptq(graph, tensors, empty_table.profile())
        self.assertEqual(second_plan.nodes, ())
        self.assertEqual(second_plan.byte_nodes, ())
        second_report = materialize_runtime_ptq(graph, tensors, second_plan)
        self.assertEqual(second_report.nodes_quantized, 0)
        self.assertEqual(graph.fingerprint(), quantized_fingerprint)
        self.assertEqual(set(tensors), set(tensor_snapshot))
        for name, expected in tensor_snapshot.items():
            np.testing.assert_array_equal(tensors[name], expected)

    def test_batch_matmul_lowers_with_broadcast_and_exact_byte_boundaries(self):
        graph, tensors = batch_matmul_graph()
        sample = {
            "a": np.asarray([
                [[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]],
                [[-0.25, 0.75, 1.0], [0.5, -0.75, 0.25]],
            ], dtype=np.float32),
            "b": np.asarray([[
                [0.5, -0.25], [1.0, 0.75], [-0.5, 0.25],
            ]], dtype=np.float32),
        }
        calibration, executions = calibrate_inputs(graph, tensors, (sample,))
        plan = plan_runtime_ptq(graph, tensors, calibration)
        self.assertEqual(plan.byte_nodes[0].quantized_op, "QBatchMatMul")
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QuantizeLinear", "QBatchMatMul", "DequantizeLinear"],
        )
        self.assertEqual(report.quantize_boundaries, 2)
        document, packaged = export_runtime_package(graph, tensors)
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        actual = execute_reference(graph, tensors, sample)
        compare_tensor_maps(
            executions[0].outputs, actual.outputs,
            tensor_order=("y",), atol=0.04, rtol=0.0,
        ).raise_for_divergence()

    def test_conv2d_lowers_to_canonical_ohwi_and_synthesizes_zero_bias(self):
        graph, tensors = conv_graph()
        sample = np.linspace(-1.0, 1.0, 18, dtype=np.float32).reshape(1, 3, 3, 2)
        calibration, executions = calibrate(graph, tensors, (sample,))
        plan = plan_runtime_ptq(graph, tensors, calibration)
        self.assertEqual(plan.conv_nodes[0].weight_shape, (2, 2, 2, 1))
        self.assertIsNone(plan.conv_nodes[0].source_bias)
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QConv2D", "DequantizeLinear"],
        )
        qconv = graph.nodes[1]
        self.assertEqual(graph.tensors[qconv.input_map()["weight"]].shape, (2, 2, 2, 1))
        self.assertEqual(tensors[qconv.input_map()["bias"]].dtype, np.int32)
        self.assertEqual(report.source_initializers_removed, 1)
        document, packaged = export_runtime_package(graph, tensors)
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        actual = execute_reference(graph, tensors, {"x": sample})
        compare_tensor_maps(
            executions[0].outputs, actual.outputs,
            tensor_order=("y",), atol=0.05, rtol=0.0,
        ).raise_for_divergence()

    def test_embedding_uses_complete_table_affine_and_starts_shared_byte_island(self):
        graph, tensors = embedding_graph()
        graph.tensors["y"].public_output = False
        graph.outputs.clear()
        add_tensor(graph, "z", (2, 2, 4), public_output=True)
        graph.add_node(OpNode.from_maps(
            "gelu", "GELU", {"input": "y"}, {"out": "z"},
            attributes=(OpAttribute(
                "params", "volvox.params", {"approximate": "none"},
            ),),
        ))
        graph.outputs.append("z")
        graph.verify(IRDialect.RUNTIME)
        ids = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        expected = execute_reference(graph, tensors, {"ids": ids})
        table = CalibrationTable(graph)
        self.assertEqual(table.tensor_names, ("z",))
        table.observe_reference(expected)
        plan = plan_runtime_ptq(graph, tensors, table.profile())
        embedding_activation = plan.activation("y")
        self.assertEqual(
            embedding_activation.affine_source,
            "producer:embedding:complete-table",
        )
        self.assertEqual(plan.embedding_nodes[0].quantized_weight.zero_points, (0,) * 5)
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QEmbedding", "QGELU", "DequantizeLinear"],
        )
        self.assertEqual(report.quantize_boundaries, 0)
        self.assertEqual(report.byte_edges_reused, 1)
        document, packaged = export_runtime_package(graph, tensors)
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        actual = execute_reference(graph, tensors, {"ids": ids})
        compare_tensor_maps(
            expected.outputs, actual.outputs,
            tensor_order=("z",), atol=0.08, rtol=0.0,
        ).raise_for_divergence()

    def test_embedding_accepts_only_vocabulary_bounded_internal_clip_ids(self):
        graph, tensors = clipped_embedding_graph()
        calibration = CalibrationTable(graph).profile()
        plan = plan_runtime_ptq(graph, tensors, calibration)
        self.assertEqual(len(plan.embedding_nodes), 1)
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["Clip", "QEmbedding", "DequantizeLinear"],
        )
        self.assertEqual(
            graph.nodes[1].input_map()["input"], "bounded_ids",
        )
        self.assertEqual(report.nodes_quantized, 1)

        unbounded, unbounded_tensors = clipped_embedding_graph(maximum=5)
        unbounded_profile = CalibrationTable(unbounded).profile()
        unbounded_plan = plan_runtime_ptq(
            unbounded, unbounded_tensors, unbounded_profile,
        )
        self.assertEqual(
            tuple(item.diagnostic_code for item in unbounded_plan.retained_nodes),
            ("VXPTQ082",),
        )
        unbounded_report = materialize_runtime_ptq(
            unbounded, unbounded_tensors, unbounded_plan,
        )
        self.assertEqual(unbounded_report.nodes_quantized, 0)
        self.assertEqual(
            [node.op_type for node in unbounded.nodes], ["Clip", "Embedding"],
        )
        self.assertEqual(set(unbounded_tensors), {"table"})
        strict, _ = clipped_embedding_graph(maximum=5)
        with self.assertRaisesRegex(
            ExporterError,
            "public I32 IDs or a canonical I32 Clip",
        ):
            CalibrationTable(strict, selected_nodes=("embedding",))

    def test_cross_attention_lowers_to_qsdpa_with_keep_mask_contract(self):
        graph, tensors = attention_graph()
        sample = {
            "q": np.asarray([[[0.5, -0.25, 0.75, 0.0],
                              [0.25, 0.5, -0.5, 1.0]]], dtype=np.float32),
            "k": np.asarray([[[0.25, -0.5, 0.5, 0.75],
                              [1.0, 0.0, -0.25, 0.5],
                              [-0.5, 0.75, 0.25, -0.25]]], dtype=np.float32),
            "v": np.asarray([[[0.5, 0.25, -0.5, 0.75],
                              [1.0, -0.25, 0.0, 0.5],
                              [-0.75, 0.5, 0.25, -0.5]]], dtype=np.float32),
            "mask": np.asarray([[1, 1, 0]], dtype=np.int32),
        }
        calibration, executions = calibrate_inputs(graph, tensors, (sample,))
        self.assertNotIn("mask", calibration.observation_map)
        plan = plan_runtime_ptq(graph, tensors, calibration)
        self.assertEqual(plan.byte_nodes[0].quantized_op, "QSDPA")
        report = materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["QuantizeLinear", "QuantizeLinear", "QuantizeLinear", "QSDPA", "DequantizeLinear"],
        )
        self.assertEqual(report.quantize_boundaries, 3)
        document, packaged = export_runtime_package(graph, tensors)
        validation = validate_graph(document, ["portable"], weights=packaged)
        self.assertTrue(validation.supported, validation.diagnostics)
        actual = execute_reference(graph, tensors, sample)
        compare_tensor_maps(
            executions[0].outputs, actual.outputs,
            tensor_order=("y",), atol=0.05, rtol=0.0,
        ).raise_for_divergence()

    def test_unrepresentable_packed_sdpa_falls_back_but_malformed_forms_fail(self):
        graph = GraphIR("volvoxai", "packed-sdpa.json", IRDialect.RUNTIME)
        add_tensor(graph, "qkv", (1, 2, 12), public_input=True)
        add_tensor(graph, "y", (1, 2, 4), public_output=True)
        graph.inputs.append("qkv")
        graph.add_node(OpNode.from_maps(
            "sdpa", "SDPA", {"qkv": "qkv"}, {"out": "y"},
            attributes=(OpAttribute("params", "volvox.params", {
                "heads": 1, "causal": False,
            }),),
        ))
        graph.outputs.append("y")
        graph.verify(IRDialect.RUNTIME)
        self.assertEqual(required_ptq_observations(graph), ())
        plan = plan_runtime_ptq(
            graph, {}, CalibrationTable(graph).profile(),
        )
        self.assertEqual(
            tuple(item.diagnostic_code for item in plan.retained_nodes),
            ("VXPTQ088",),
        )
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph, selected_nodes=("sdpa",))
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ088")

        graph, _ = conv_graph()
        graph.nodes[0].attributes = (OpAttribute("params", "volvox.params", {
            "weight_layout": "HWIO", "data_layout": "NHWC", "groups": 2,
        }),)
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ079")

        graph, _ = attention_graph()
        graph.nodes[0].attributes = (OpAttribute("params", "volvox.params", {
            "heads": 1, "causal": False, "mask_encoding": "additive",
        }),)
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ087")

    def test_unsupported_or_incomplete_topology_fails_closed(self):
        graph, tensors = matmul_graph()
        graph.tensors["b"].initializer = False
        graph.tensors["b"].public_input = True
        graph.inputs.append("b")
        del tensors["b"]
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ029")

        graph, tensors = matmul_graph()
        graph.nodes[0].attributes = (
            OpAttribute("params", "volvox.params", {"transB": True}),
        )
        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ027")

        graph, tensors = linear_graph()
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        execution = execute_reference(graph, tensors, {"x": sample})
        incomplete = CalibrationTable(graph, ("x",))
        incomplete.observe_reference(execution)
        with self.assertRaises(ExporterError) as caught:
            plan_runtime_ptq(graph, tensors, incomplete.profile())
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ011")

    def test_ptq_entry_rejects_hidden_affine_params(self):
        graph, _ = linear_graph()
        graph.nodes[0].attributes = (OpAttribute(
            "params",
            "volvox.params",
            {
                "weight_layout": "din_dout",
                "private": [{"affine": {"zero_point": 0}}],
            },
        ),)

        with self.assertRaises(ExporterError) as caught:
            required_ptq_observations(graph)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR042")
        self.assertIn(
            "params.private[0].affine.zero_point",
            caught.exception.diagnostic.message,
        )

    def test_stale_plan_and_failed_working_copy_leave_both_states_unchanged(self):
        graph, tensors = linear_graph()
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        calibration, _ = calibrate(graph, tensors, (sample,))
        plan = plan_runtime_ptq(graph, tensors, calibration)
        graph_before = graph.fingerprint()
        tensors_before = {name: value.copy() for name, value in tensors.items()}

        tensors["weight"][0, 0] += np.float32(1.0)
        with self.assertRaises(ExporterError) as caught:
            materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ051")
        self.assertEqual(graph.fingerprint(), graph_before)
        tensors["weight"] = tensors_before["weight"].copy()

        with mock.patch(
            "tools.exporter.typed_ptq.export_runtime_package",
            side_effect=RuntimeError("qualification failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "qualification failure"):
                materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(graph.fingerprint(), graph_before)
        self.assertEqual(set(tensors), set(tensors_before))
        for name, expected in tensors_before.items():
            np.testing.assert_array_equal(tensors[name], expected)

    def test_materializer_refuses_an_implementation_that_changes_public_abi(self):
        graph, tensors = linear_graph()
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        calibration, _ = calibrate(graph, tensors, (sample,))
        plan = plan_runtime_ptq(graph, tensors, calibration)
        graph_before = graph.fingerprint()
        tensors_before = {
            name: np.array(value, copy=True) for name, value in tensors.items()
        }
        apply_plan = typed_ptq._apply_plan

        def corrupt_public_abi(working_graph, working_tensors, current_plan):
            report = apply_plan(working_graph, working_tensors, current_plan)
            name = working_graph.inputs.pop()
            working_graph.tensors[name].public_input = False
            return report

        with mock.patch.object(
            typed_ptq, "_apply_plan", side_effect=corrupt_public_abi,
        ):
            with self.assertRaises(ExporterError) as caught:
                materialize_runtime_ptq(graph, tensors, plan)
        self.assertEqual(caught.exception.diagnostic.code, "VXPTQ095")
        self.assertEqual(graph.fingerprint(), graph_before)
        self.assertEqual(set(tensors), set(tensors_before))
        for name, expected in tensors_before.items():
            np.testing.assert_array_equal(tensors[name], expected)

    def test_runtime_ir_rejects_noncanonical_qlinear(self):
        graph, tensors = linear_graph()
        sample = np.asarray(
            [[1.0, -2.0, 0.5], [-0.75, 0.25, 2.0]], dtype=np.float32,
        )
        calibration, _ = calibrate(graph, tensors, (sample,))
        materialize_runtime_ptq(
            graph, tensors, plan_runtime_ptq(graph, tensors, calibration),
        )
        qlinear = next(node for node in graph.nodes if node.op_type == "QLinear")
        qlinear.attributes = (
            OpAttribute("params", "volvox.params", {"weight_layout": "dout_din"}),
        )
        with self.assertRaises(ExporterError) as caught:
            graph.verify(IRDialect.RUNTIME)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR035")

        qlinear.attributes = ()
        weight = graph.tensors[qlinear.input_map()["weight"]]
        descriptor = weight.quantization
        assert descriptor is not None
        weight.quantization = AffineQuantization(
            "per_axis", descriptor.scale, descriptor.zero_point, axis=-2,
        )
        with self.assertRaises(ExporterError) as caught:
            graph.verify(IRDialect.RUNTIME)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR036")


if __name__ == "__main__":
    unittest.main()
