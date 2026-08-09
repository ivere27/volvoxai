from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import onnx
    import safetensors  # noqa: F401 -- importing is part of the dependency gate.
    from onnx import TensorProto, helper, numpy_helper
    from safetensors.numpy import safe_open
except ImportError as error:  # This module is the installed-export-dependencies suite.
    raise unittest.SkipTest(
        "ONNX frontend tests require installed numpy, onnx, and safetensors"
    ) from error

from tools.exporter.capabilities import classify_package, validate_graph
from tools.exporter.errors import ExporterError
from tools.exporter.frontend_onnx import OnnxCompiler
from tools.exporter.quantization_storage import validate_external_quantization


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPORTER_CLI = REPOSITORY_ROOT / "tools" / "export_safetensors.py"
ONNX_OPSET = 20


def _resolved_quantization(graph, weights, name):
    descriptor = validate_external_quantization(graph, weights)[name]
    if descriptor.scheme == "per_axis":
        return {
            "scheme": "per_axis",
            "axis": descriptor.axis,
            "scales": descriptor.scales.tolist(),
            "zero_points": descriptor.zero_points.tolist(),
        }
    return {
        "scheme": "per_tensor",
        "scale": float(descriptor.scales[0]),
        "zero_point": int(descriptor.zero_points[0]),
    }


def _initializer(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def _value(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def _save_model(
    directory: Path,
    filename: str,
    *,
    nodes,
    inputs,
    outputs,
    initializers=(),
    value_info=(),
    extra_opsets=(),
    opset=ONNX_OPSET,
) -> Path:
    graph = helper.make_graph(
        list(nodes),
        filename.removesuffix(".onnx"),
        list(inputs),
        list(outputs),
        list(initializers),
        value_info=list(value_info),
    )
    model = helper.make_model(
        graph,
        producer_name="volvoxai-exporter-tests",
        opset_imports=[
            helper.make_opsetid("", opset),
            *[helper.make_opsetid(domain, version) for domain, version in extra_opsets],
        ],
    )
    onnx.checker.check_model(model)
    destination = directory / filename
    onnx.save(model, destination)
    return destination


def _lora_model(directory: Path, *, dynamic_scale: bool = False):
    x = np.asarray(
        [[[1.0, 2.0, -1.0], [0.5, -0.25, 3.0]]],
        dtype=np.float32,
    )
    base_weight = np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0
    a_weight = np.asarray(
        [[1.0, 0.5], [-1.0, 2.0], [0.25, -0.5]],
        dtype=np.float32,
    )
    b_weight = np.asarray(
        [[0.2, -0.4, 0.6, 0.8], [1.0, -0.5, 0.25, -0.75]],
        dtype=np.float32,
    )
    scale = np.float32(0.25)
    inputs = [_value("tokens", TensorProto.FLOAT, [1, 2, 3])]
    initializers = [
        _initializer("base_weight", base_weight),
        _initializer("lora_a_weight", a_weight),
        _initializer("lora_b_weight", b_weight),
    ]
    if dynamic_scale:
        inputs.append(_value("lora_scale", TensorProto.FLOAT, []))
    else:
        initializers.append(_initializer("lora_scale", scale))
    nodes = [
        helper.make_node(
            "MatMul", ["tokens", "base_weight"], ["base"], name="base_projection"
        ),
        helper.make_node(
            "MatMul", ["tokens", "lora_a_weight"], ["low_rank"], name="lora_a"
        ),
        helper.make_node(
            "MatMul", ["low_rank", "lora_b_weight"], ["delta_raw"], name="lora_b"
        ),
        helper.make_node(
            "Mul", ["delta_raw", "lora_scale"], ["delta"], name="lora_scale_mul"
        ),
        helper.make_node("Add", ["base", "delta"], ["result"], name="lora_residual"),
    ]
    path = _save_model(
        directory,
        "dynamic_lora.onnx" if dynamic_scale else "explicit_lora_f32.onnx",
        nodes=nodes,
        inputs=inputs,
        outputs=[_value("result", TensorProto.FLOAT, [1, 2, 4])],
        initializers=initializers,
    )
    return path, x, base_weight, a_weight, b_weight, scale


def _argmax_model(directory: Path, *, select_last_index: int, filename: str) -> Path:
    node = helper.make_node(
        "ArgMax",
        ["logits"],
        ["route"],
        name="route_argmax",
        axis=1,
        keepdims=0,
        select_last_index=select_last_index,
    )
    return _save_model(
        directory,
        filename,
        nodes=[node],
        inputs=[_value("logits", TensorProto.FLOAT, [1, 3])],
        outputs=[_value("route", TensorProto.INT64, [1])],
    )


def _qdq_conv_model(
    directory: Path,
    filename: str,
    *,
    input_shape=(1, 2, 3, 4),
    raw_weight=None,
    activation_dtype=np.uint8,
    weight_dtype=np.int8,
    weight_axis: int = 0,
    weight_scale_dtype=np.float32,
    input_scale=0.25,
    weight_scales=None,
    output_scale=0.125,
    weight_zero_points=None,
    strides=(1, 1),
    dilations=(1, 1),
    pads=(1, 1, 1, 1),
    group: int = 1,
    include_bias: bool = False,
    float_fanout: bool = False,
) -> Path:
    input_shape = tuple(input_shape)
    if raw_weight is None:
        raw_weight = (
            np.arange(3 * input_shape[1] * 3 * 3, dtype=np.int16)
            .reshape(3, input_shape[1], 3, 3)
            % 13 - 6
        ).astype(weight_dtype)
    raw_weight = np.asarray(raw_weight, dtype=weight_dtype)
    output_channels, _input_per_group, kernel_height, kernel_width = (
        raw_weight.shape
    )
    output_height = (
        input_shape[2] + pads[0] + pads[2]
        - dilations[0] * (kernel_height - 1) - 1
    ) // strides[0] + 1
    output_width = (
        input_shape[3] + pads[1] + pads[3]
        - dilations[1] * (kernel_width - 1) - 1
    ) // strides[1] + 1
    output_shape = [
        input_shape[0], output_channels, output_height, output_width,
    ]
    activation_type = (
        TensorProto.UINT8
        if np.dtype(activation_dtype) == np.dtype(np.uint8)
        else TensorProto.INT8
    )
    activation_zero = (
        np.asarray(128, dtype=np.uint8)
        if activation_type == TensorProto.UINT8
        else np.asarray(0, dtype=np.int8)
    )
    if weight_scales is None:
        weight_scales = np.linspace(
            0.25, 0.5, output_channels, dtype=np.float32
        )
    if weight_zero_points is None:
        weight_zero_points = np.zeros(output_channels, dtype=weight_dtype)
    nodes = [
        helper.make_node(
            "DequantizeLinear",
            ["raw_input", "input_scale", "input_zero"],
            ["input"],
            name="input_dequantize",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["raw_weight", "weight_scales", "weight_zero"],
            ["weight"],
            name="weight_dequantize",
            axis=weight_axis,
        ),
    ]
    conv_inputs = ["input", "weight"]
    initializers = [
        _initializer("input_scale", np.asarray(input_scale, dtype=np.float32)),
        _initializer("input_zero", activation_zero),
        _initializer("raw_weight", raw_weight),
        _initializer(
            "weight_scales",
            np.asarray(weight_scales, dtype=weight_scale_dtype),
        ),
        _initializer(
            "weight_zero",
            np.asarray(weight_zero_points, dtype=weight_dtype),
        ),
        _initializer("output_scale", np.asarray(output_scale, dtype=np.float32)),
        _initializer("output_zero", np.asarray(120, dtype=np.uint8)),
    ]
    if include_bias:
        conv_inputs.append("bias")
        initializers.append(
            _initializer(
                "bias", np.zeros(output_channels, dtype=np.float32)
            )
        )
    nodes.extend([
        helper.make_node(
            "Conv",
            conv_inputs,
            ["convolution"],
            name="convolution",
            auto_pad="NOTSET",
            pads=list(pads),
            strides=list(strides),
            dilations=list(dilations),
            group=group,
        ),
        helper.make_node(
            "QuantizeLinear",
            ["convolution", "output_scale", "output_zero"],
            ["result"],
            name="output_quantize",
        ),
    ])
    outputs = [_value("result", TensorProto.UINT8, output_shape)]
    if float_fanout:
        nodes.insert(
            2,
            helper.make_node(
                "Identity", ["input"], ["float_copy"], name="float_consumer"
            ),
        )
        outputs.append(
            _value("float_copy", TensorProto.FLOAT, list(input_shape))
        )
    return _save_model(
        directory,
        filename,
        nodes=nodes,
        inputs=[_value("raw_input", activation_type, list(input_shape))],
        outputs=outputs,
        initializers=initializers,
    )


def _evaluate_preserved_lora(graph, weights, tokens: np.ndarray) -> np.ndarray:
    values = {"input0": tokens, **weights}
    for node in graph["nodes"]:
        inputs = node["inputs"]
        op_type = node["opType"]
        if op_type == "Linear":
            weight = values[inputs["weight"]]
            if node["params"]["weight_layout"] == "dout_din":
                weight = weight.T
            result = np.matmul(values[inputs["input"]], weight)
            if "bias" in inputs:
                result = result + values[inputs["bias"]]
        elif op_type == "Mul":
            result = values[inputs["a"]] * values[inputs["b"]]
        elif op_type == "Add":
            result = values[inputs["a"]] + values[inputs["b"]]
        else:  # Keep this evaluator deliberately scoped to the preserved region.
            raise AssertionError(f"unexpected LoRA node {op_type!r}")
        values[node["outputs"]["out"]["tensor"]] = result
    return values[graph["outputs"][0]]


def _output_descriptor(node, port: str = "out") -> dict:
    return node["outputs"][port]


def _output_tensor(node, port: str = "out") -> str:
    return _output_descriptor(node, port)["tensor"]


def _nodes_by_source(compiler: OnnxCompiler, graph: dict) -> dict[str, dict]:
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    return {
        item["source_node"]: nodes_by_id[item["id"]]
        for item in compiler.publication_report["node_sources"]
        if item["id"] in nodes_by_id
    }


class InstalledOnnxFrontendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="volvox-onnx-frontend-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, str(EXPORTER_CLI), *map(str, arguments)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def assert_diagnostic(self, code: str, callback) -> ExporterError:
        with self.assertRaises(ExporterError) as caught:
            callback()
        self.assertEqual(caught.exception.diagnostic.code, code)
        return caught.exception

    def assert_no_stage_directories(self, directory: Path) -> None:
        self.assertEqual(list(directory.glob(".model.export-*")), [])

    def _dynamic_concat_model(
        self,
        filename: str,
        *,
        first_axis: int | str,
        second_axis: int | str,
        output_axis: str,
    ) -> Path:
        return _save_model(
            self.root,
            filename,
            nodes=[helper.make_node(
                "Concat",
                ["first", "second"],
                ["memory"],
                name="memory_concat",
                axis=1,
            )],
            inputs=[
                _value("first", TensorProto.FLOAT, ["B", first_axis, 320]),
                _value("second", TensorProto.FLOAT, ["B", second_axis, 320]),
            ],
            outputs=[
                _value("memory", TensorProto.FLOAT, ["B", output_axis, 320])
            ],
        )

    def test_dynamic_concat_uses_exact_canonical_output_only_symbol(self):
        path = self._dynamic_concat_model(
            "dynamic_concat.onnx",
            first_axis=210,
            second_axis="Q",
            output_axis="M",
        )
        bounds = {
            "B": {"min": 1, "max": 1},
            "Q": {"min": 1, "max": 192},
            "M": {"min": 211, "max": 402},
        }
        graph, weights = OnnxCompiler(
            str(path), dimension_bounds=bounds
        ).lower()

        self.assertEqual(graph["dimensions"], bounds)
        self.assertEqual(graph["nodes"], [{
            "id": "node_0",
            "opType": "Concat",
            "inputs": {"input0": "input0", "input1": "input1"},
            "outputs": {"out": {
                "tensor": "output0",
                "dtype": "float32",
                "shape": ["B", "M", 320],
            }},
            "params": {"axis": 1},
        }])
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

    def test_dynamic_concat_reuses_one_affine_symbol_for_memory_and_mask(self):
        path = _save_model(
            self.root,
            "dynamic_memory_and_mask_concat.onnx",
            nodes=[
                helper.make_node(
                    "Concat",
                    ["image", "question"],
                    ["memory"],
                    name="memory_concat",
                    axis=1,
                ),
                helper.make_node(
                    "Concat",
                    ["image_mask", "question_mask"],
                    ["memory_mask"],
                    name="mask_concat",
                    axis=1,
                ),
            ],
            inputs=[
                _value("image", TensorProto.FLOAT, ["B", 210, 320]),
                _value("question", TensorProto.FLOAT, ["B", "Q", 320]),
                _value("image_mask", TensorProto.INT32, ["B", 210]),
                _value("question_mask", TensorProto.INT32, ["B", "Q"]),
            ],
            outputs=[
                _value("memory", TensorProto.FLOAT, ["B", "M", 320]),
                _value("memory_mask", TensorProto.INT32, ["B", "M"]),
            ],
        )
        graph, weights = OnnxCompiler(
            str(path),
            dimension_bounds={
                "B": {"min": 1, "max": 1},
                "Q": {"min": 1, "max": 192},
                "M": {"min": 211, "max": 402},
            },
        ).lower()

        self.assertEqual(
            [node["outputs"]["out"]["shape"] for node in graph["nodes"]],
            [["B", "M", 320], ["B", "M"]],
        )
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

    def test_kv_cache_concat_proves_positive_seed_affine_domain(self):
        path = _save_model(
            self.root,
            "kv_cache_concat.onnx",
            nodes=[helper.make_node(
                "Concat",
                ["past_k", "current_k"],
                ["present_k"],
                name="append_key_cache",
                axis=2,
            )],
            inputs=[
                _value("past_k", TensorProto.FLOAT, ["B", 8, "P", 40]),
                _value("current_k", TensorProto.FLOAT, ["B", 8, 1, 40]),
            ],
            outputs=[
                _value("present_k", TensorProto.FLOAT, ["B", 8, "R", 40])
            ],
        )
        bounds = {
            "B": {"min": 1, "max": 1},
            # P=1 is the representable masked-zero seed.  Zero extents remain
            # outside the bounded runtime contract.
            "P": {"min": 1, "max": 191},
            "R": {"min": 2, "max": 192},
        }

        graph, weights = OnnxCompiler(
            str(path), dimension_bounds=bounds
        ).lower()

        self.assertEqual(graph["dimensions"], bounds)
        self.assertEqual(len(graph["nodes"]), 1)
        self.assertEqual(graph["nodes"][0]["opType"], "Concat")
        self.assertEqual(
            graph["nodes"][0]["outputs"]["out"]["shape"],
            ["B", 8, "R", 40],
        )
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )
        zero_seed_error = self.assert_diagnostic(
            "VXIR053",
            lambda: OnnxCompiler(
                str(path),
                dimension_bounds={
                    "B": {"min": 1, "max": 1},
                    "P": {"min": 0, "max": 191},
                    "R": {"min": 1, "max": 192},
                },
            ),
        )
        self.assertIn("positive safe integer", zero_seed_error.diagnostic.message)

    def test_symbolic_shape_program_lowers_cached_projection_structurally(self):
        path = _save_model(
            self.root,
            "cached_projection_shape_program.onnx",
            nodes=[
                helper.make_node(
                    "Shape", ["token_state"], ["state_shape"], name="state_shape"
                ),
                helper.make_node(
                    "Gather",
                    ["state_shape", "batch_axis"],
                    ["batch_extent"],
                    name="gather_batch",
                    axis=0,
                ),
                helper.make_node(
                    "Gather",
                    ["state_shape", "step_axis"],
                    ["step_extent"],
                    name="gather_step",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["batch_extent", "step_extent", "heads", "head_width"],
                    ["projection_shape"],
                    name="assemble_projection_shape",
                    axis=0,
                ),
                helper.make_node(
                    "Reshape",
                    ["token_state", "projection_shape"],
                    ["current_bshd"],
                    name="reshape_projection",
                ),
                helper.make_node(
                    "Transpose",
                    ["current_bshd"],
                    ["current_k"],
                    name="transpose_projection",
                    perm=[0, 2, 1, 3],
                ),
                helper.make_node(
                    "Concat",
                    ["past_k", "current_k"],
                    ["present_k"],
                    name="append_key_cache",
                    axis=2,
                ),
            ],
            inputs=[
                _value("token_state", TensorProto.FLOAT, ["B", 1, 320]),
                _value("past_k", TensorProto.FLOAT, ["B", 8, "P", 40]),
            ],
            outputs=[
                _value("present_k", TensorProto.FLOAT, ["B", 8, "R", 40])
            ],
            initializers=[
                _initializer("batch_axis", np.asarray([0], dtype=np.int64)),
                _initializer("step_axis", np.asarray([1], dtype=np.int64)),
                _initializer("heads", np.asarray([8], dtype=np.int64)),
                _initializer("head_width", np.asarray([40], dtype=np.int64)),
            ],
            value_info=[
                _value("current_bshd", TensorProto.FLOAT, ["B", 1, 8, 40]),
                _value("current_k", TensorProto.FLOAT, ["B", 8, 1, 40]),
            ],
            opset=18,
        )
        compiler = OnnxCompiler(
            str(path),
            dimension_bounds={
                "B": {"min": 1, "max": 1},
                "P": {"min": 1, "max": 191},
                "R": {"min": 2, "max": 192},
            },
        )
        graph, weights = compiler.lower()

        structural_sources = {
            compiler.model.graph.node[index].name
            for index in compiler.structural_shape_nodes
        }
        self.assertEqual(structural_sources, {
            "state_shape",
            "gather_batch",
            "gather_step",
            "assemble_projection_shape",
        })
        self.assertEqual(
            [node["opType"] for node in graph["nodes"]],
            ["Reshape", "Concat"],
        )
        self.assertEqual(
            graph["nodes"][-1]["outputs"]["out"]["shape"],
            ["B", 8, "R", 40],
        )
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

    def test_symbolic_shape_program_is_not_erased_when_shape_value_escapes(self):
        path = _save_model(
            self.root,
            "escaping_shape_program.onnx",
            nodes=[
                helper.make_node(
                    "Shape", ["tokens"], ["runtime_shape"], name="runtime_shape"
                ),
                helper.make_node(
                    "Reshape",
                    ["tokens", "runtime_shape"],
                    ["reshaped"],
                    name="reshape_tokens",
                ),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, ["B", 4])],
            outputs=[
                _value("reshaped", TensorProto.FLOAT, ["B", 4]),
                _value("runtime_shape", TensorProto.INT64, [2]),
            ],
        )
        compiler = OnnxCompiler(
            str(path),
            dimension_bounds={"B": {"min": 1, "max": 8}},
            output_dtypes={"runtime_shape": "int32"},
        )

        self.assertEqual(compiler.structural_shape_nodes, set())
        error = self.assert_diagnostic("VXONNX_UNSUPPORTED", compiler.lower)
        self.assertEqual(error.diagnostic.source_node, "runtime_shape")
        self.assertEqual(error.diagnostic.source_op, "Shape")

    def test_symbolic_shape_program_rejects_non_identity_cast_semantics(self):
        path = _save_model(
            self.root,
            "shape_program_bool_cast.onnx",
            nodes=[
                helper.make_node(
                    "Shape", ["tokens"], ["runtime_shape"], name="runtime_shape"
                ),
                helper.make_node(
                    "Cast",
                    ["runtime_shape"],
                    ["truth_shape"],
                    name="shape_to_bool",
                    to=TensorProto.BOOL,
                ),
                helper.make_node(
                    "Cast",
                    ["truth_shape"],
                    ["integer_shape"],
                    name="bool_to_shape",
                    to=TensorProto.INT64,
                ),
                helper.make_node(
                    "Reshape",
                    ["tokens", "integer_shape"],
                    ["reshaped"],
                    name="reshape_tokens",
                ),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, ["B", 4])],
            outputs=[_value("reshaped", TensorProto.FLOAT, ["B", 4])],
        )
        compiler = OnnxCompiler(
            str(path), dimension_bounds={"B": {"min": 1, "max": 8}}
        )

        # For B > 1 the source Casts produce [1, 1], not [B, 4].  Treating
        # either Cast as metadata-only would silently change ONNX semantics.
        self.assertEqual(compiler.structural_shape_nodes, set())
        error = self.assert_diagnostic("VXONNX_UNSUPPORTED", compiler.lower)
        self.assertEqual(error.diagnostic.source_node, "runtime_shape")
        self.assertEqual(error.diagnostic.source_op, "Shape")

    def test_kv_cache_concat_recovers_unique_opaque_qdq_axis(self):
        path = _save_model(
            self.root,
            "kv_cache_concat_qdq_opaque_axis.onnx",
            nodes=[
                helper.make_node(
                    "Concat",
                    ["past_v", "current_v"],
                    ["present_v_float"],
                    name="append_value_cache",
                    axis=2,
                ),
                helper.make_node(
                    "QuantizeLinear",
                    ["present_v_float", "scale", "zero"],
                    ["present_v_byte"],
                    name="quantize_present_v",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    ["present_v_byte", "scale", "zero"],
                    ["present_v"],
                    name="dequantize_present_v",
                ),
            ],
            inputs=[
                _value("past_v", TensorProto.FLOAT, ["B", 8, "P", 40]),
                _value("current_v", TensorProto.FLOAT, ["B", 8, 1, 40]),
            ],
            outputs=[
                _value("present_v", TensorProto.FLOAT, ["B", 8, "R", 40])
            ],
            initializers=[
                _initializer("scale", np.asarray(0.125, dtype=np.float32)),
                _initializer("zero", np.asarray(0, dtype=np.uint8)),
            ],
            value_info=[
                _value(
                    "present_v_float",
                    TensorProto.FLOAT,
                    ["B", 8, "unk__7", 40],
                ),
                _value(
                    "present_v_byte",
                    TensorProto.UINT8,
                    ["B", 8, "unk__7", 40],
                ),
            ],
        )
        graph, weights = OnnxCompiler(
            str(path),
            dimension_bounds={
                "B": {"min": 1, "max": 1},
                "P": {"min": 1, "max": 191},
                "R": {"min": 2, "max": 192},
            },
        ).lower()

        self.assertEqual(
            [node["outputs"]["out"]["shape"] for node in graph["nodes"]],
            [["B", 8, "R", 40]] * 3,
        )
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

    def test_static_cache_reshape_refines_only_inference_opaque_axis(self):
        path = _save_model(
            self.root,
            "cache_static_reshape_opaque_batch.onnx",
            nodes=[
                helper.make_node(
                    "Reshape",
                    ["cache_heads", "flattened_shape"],
                    ["flattened_opaque"],
                    name="flatten_cache_heads",
                ),
                helper.make_node(
                    "Identity",
                    ["flattened_opaque"],
                    ["flattened"],
                    name="publish_flattened_cache",
                ),
            ],
            inputs=[
                _value("cache_heads", TensorProto.FLOAT, ["B", 8, 1, 40])
            ],
            outputs=[
                _value("flattened", TensorProto.FLOAT, ["B", 1, 320])
            ],
            initializers=[
                _initializer(
                    "flattened_shape", np.asarray([-1, 1, 320], dtype=np.int64)
                )
            ],
            value_info=[
                _value(
                    "flattened_opaque",
                    TensorProto.FLOAT,
                    ["unk__9", 1, 320],
                )
            ],
        )
        compiler = OnnxCompiler(
            str(path),
            dimension_bounds={"B": {"min": 1, "max": 8}},
        )
        graph, weights = compiler.lower()

        self.assertEqual(compiler.shape_of("flattened_opaque"), ["B", 1, 320])
        self.assertEqual(
            graph["nodes"][0]["outputs"]["out"]["shape"], ["B", 1, 320]
        )
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

    def test_source_bool_identity_cast_preserves_int32_mask_representation(self):
        path = _save_model(
            self.root,
            "bool_identity_cast_mask.onnx",
            nodes=[
                helper.make_node(
                    "Cast",
                    ["blocked_mask"],
                    ["blocked_again"],
                    name="source_bool_identity",
                    to=TensorProto.BOOL,
                ),
                helper.make_node(
                    "Where",
                    ["blocked_again", "blocked_score", "scores"],
                    ["masked_scores"],
                    name="apply_blocked_mask",
                ),
            ],
            inputs=[
                _value("blocked_mask", TensorProto.BOOL, ["B", "R"]),
                _value("scores", TensorProto.FLOAT, ["B", "R"]),
            ],
            outputs=[
                _value("masked_scores", TensorProto.FLOAT, ["B", "R"])
            ],
            initializers=[
                _initializer("blocked_score", np.asarray(-1.0e4, dtype=np.float32))
            ],
        )
        compiler = OnnxCompiler(
            str(path),
            dimension_bounds={
                "B": {"min": 1, "max": 1},
                "R": {"min": 2, "max": 192},
            },
            input_dtypes={"blocked_mask": "int32"},
        )
        graph, weights = compiler.lower()

        self.assertEqual(compiler.identity_bool_casts, {0})
        self.assertNotIn("Cast", [node["opType"] for node in graph["nodes"]])
        where = next(node for node in graph["nodes"] if node["opType"] == "Where")
        self.assertEqual(where["inputs"]["condition"], "input0")
        self.assertIn({
            "kind": "input-dtype",
            "name": "blocked_mask",
            "source": "bool",
            "exported": "int32",
        }, compiler.publication_report["abi_changes"])
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

        unsafe_path = _save_model(
            self.root,
            "int32_to_bool_cast.onnx",
            nodes=[helper.make_node(
                "Cast",
                ["values"],
                ["condition"],
                name="nonzero_to_bool",
                to=TensorProto.BOOL,
            )],
            inputs=[_value("values", TensorProto.INT32, [1, 2])],
            outputs=[_value("condition", TensorProto.BOOL, [1, 2])],
        )
        self.assert_diagnostic(
            "VXCAST_DTYPE",
            lambda: OnnxCompiler(
                str(unsafe_path), output_dtypes={"condition": "int32"}
            ).lower(),
        )

    def test_dynamic_concat_rejects_unsound_bounds_and_multiple_terms(self):
        cases = (
            (
                "concat_bad_bounds.onnx",
                210,
                "Q",
                {"M": {"min": 211, "max": 401}},
                "declared output 'out'.shape[1]",
            ),
            (
                "concat_two_dynamic.onnx",
                "Q",
                "T",
                {"T": {"min": 1, "max": 192}, "M": {"min": 2, "max": 384}},
                "more than one dynamic axis term",
            ),
        )
        for filename, first_axis, second_axis, extra_bounds, detail in cases:
            with self.subTest(filename=filename):
                path = self._dynamic_concat_model(
                    filename,
                    first_axis=first_axis,
                    second_axis=second_axis,
                    output_axis="M",
                )
                bounds = {
                    "B": {"min": 1, "max": 1},
                    "Q": {"min": 1, "max": 192},
                    **extra_bounds,
                }
                error = self.assert_diagnostic(
                    "VXDYNAMIC_CONCAT_AXIS",
                    lambda: OnnxCompiler(
                        str(path), dimension_bounds=bounds
                    ).lower(),
                )
                self.assertIn("UNPROVABLE_DYNAMIC_SHAPE_FORMULA", error.diagnostic.message)
                self.assertIn(detail, error.diagnostic.message)

    def test_dynamic_concat_does_not_trust_onnx_dimension_expressions(self):
        path = self._dynamic_concat_model(
            "concat_expression.onnx",
            first_axis=210,
            second_axis="Q",
            output_axis="Q + 210",
        )
        error = self.assert_diagnostic(
            "VXCONCAT_SHAPE",
            lambda: OnnxCompiler(
                str(path),
                dimension_bounds={
                    "B": {"min": 1, "max": 1},
                    "Q": {"min": 1, "max": 192},
                    "M": {"min": 211, "max": 402},
                },
            ).lower(),
        )
        self.assertIn("INVALID_DOMAIN", error.diagnostic.message)
        self.assertIn("valid symbol name", error.diagnostic.message)

    def test_explicit_fp32_lora_is_preserved_and_numerically_structural(self):
        path, tokens, base_weight, a_weight, b_weight, scale = _lora_model(self.root)

        compiler = OnnxCompiler(
            str(path), weight_dtype="float32", output_names=["answer"]
        )
        graph, weights = compiler.lower()
        report = compiler.publication_report

        feature = report["features"]["lora"]
        self.assertEqual(len(feature), 1)
        self.assertEqual(feature[0]["rank"], 2)
        self.assertEqual(feature[0]["scale"], float(scale))
        self.assertEqual(report["package_class"], "fp32")
        self.assertNotIn("source", graph)
        self.assertEqual(report["source_ir"]["dialect"], "source")
        self.assertEqual(report["source_ir"]["nodes"], 5)
        self.assertEqual(
            report["typed_optimizer"]["runs"][0]["input_dialect"],
            "runtime",
        )
        self.assertEqual(
            [node["opType"] for node in graph["nodes"]],
            ["Linear", "Linear", "Linear", "Mul", "Add"],
        )
        self.assertTrue(all(value.dtype == np.float32 for value in weights.values()))

        nodes_by_source = _nodes_by_source(compiler, graph)
        np.testing.assert_array_equal(
            weights[nodes_by_source["base_projection"]["inputs"]["weight"]],
            base_weight,
        )
        np.testing.assert_array_equal(
            weights[nodes_by_source["lora_a"]["inputs"]["weight"]],
            a_weight,
        )
        np.testing.assert_array_equal(
            weights[nodes_by_source["lora_b"]["inputs"]["weight"]],
            b_weight,
        )
        actual = _evaluate_preserved_lora(graph, weights, tokens)
        expected = tokens @ base_weight + ((tokens @ a_weight) @ b_weight) * scale
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_source_lowering_can_defer_storage_layout_rewrites(self):
        path = _save_model(
            self.root,
            "singleton_transpose.onnx",
            nodes=[helper.make_node(
                "Transpose", ["tokens"], ["result"],
                name="move_batch_axis", perm=[1, 0, 2],
            )],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 3, 4])],
            outputs=[_value("result", TensorProto.FLOAT, [3, 1, 4])],
        )

        optimized_compiler = OnnxCompiler(str(path))
        optimized, _ = optimized_compiler.lower()
        deferred_compiler = OnnxCompiler(str(path))
        deferred, _ = deferred_compiler.lower(
            enable_static_qdq_layout_optimization=False,
        )

        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]], ["Reshape"],
        )
        self.assertEqual(
            [node["opType"] for node in deferred["nodes"]], ["Transpose"],
        )
        optimized_passes = {
            run["pass"] for run in optimized_compiler.publication_report["typed_optimizer"]["runs"]
        }
        deferred_passes = {
            run["pass"] for run in deferred_compiler.publication_report["typed_optimizer"]["runs"]
        }
        self.assertIn("runtime-singleton-transpose", optimized_passes)
        self.assertNotIn("runtime-singleton-transpose", deferred_passes)

    def test_silu_fusion_requires_the_narrow_numerical_migration_opt_in(self):
        path = _save_model(
            self.root,
            "decomposed_silu.onnx",
            nodes=[
                helper.make_node(
                    "Sigmoid", ["tokens"], ["gate"], name="silu_gate"
                ),
                helper.make_node(
                    "Mul", ["tokens", "gate"], ["result"], name="silu_mul"
                ),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, ["B", 2, 4])],
            outputs=[_value("result", TensorProto.FLOAT, ["B", 2, 4])],
        )

        bounds = {"B": {"min": 1, "max": 4}}
        default_compiler = OnnxCompiler(str(path), dimension_bounds=bounds)
        default_graph, _ = default_compiler.lower()
        fused_compiler = OnnxCompiler(str(path), dimension_bounds=bounds)
        fused_graph, _ = fused_compiler.lower(
            allow_silu_numerical_migration=True,
        )

        self.assertEqual(
            [node["opType"] for node in default_graph["nodes"]],
            ["Sigmoid", "Mul"],
        )
        self.assertEqual(
            [node["opType"] for node in fused_graph["nodes"]],
            ["SiLU"],
        )
        fused_report = fused_compiler.publication_report["typed_optimizer"]
        pipeline = fused_report["pipeline"]["recipe"]
        self.assertEqual(pipeline["selection_features"], ["silu-fusion"])
        self.assertEqual(
            sum(
                run["changes"]
                for run in fused_report["runs"]
                if run["pass"] == "runtime-silu-fusion"
            ),
            1,
        )
        for unrelated in (
            "runtime-bias-folding",
            "runtime-grouped-projection-split",
            "runtime-sequence-layout",
        ):
            self.assertNotIn(unrelated, pipeline["passes"])

    def test_silu_opt_in_is_a_zero_rewrite_for_a_silu_free_decoder_graph(self):
        path = _save_model(
            self.root,
            "decoder_without_silu.onnx",
            nodes=[helper.make_node(
                "Add", ["tokens", "tokens"], ["result"], name="decoder_add"
            )],
            inputs=[_value("tokens", TensorProto.FLOAT, ["B", 1, 4])],
            outputs=[_value("result", TensorProto.FLOAT, ["B", 1, 4])],
        )
        compiler = OnnxCompiler(
            str(path), dimension_bounds={"B": {"min": 1, "max": 1}}
        )

        graph, _ = compiler.lower(allow_silu_numerical_migration=True)

        self.assertEqual([node["opType"] for node in graph["nodes"]], ["Add"])
        report = compiler.publication_report["typed_optimizer"]
        silu_runs = [
            run for run in report["runs"]
            if run["pass"] == "runtime-silu-fusion"
        ]
        self.assertEqual(len(silu_runs), 1)
        self.assertEqual(silu_runs[0]["changes"], 0)

    def test_fixed_static_adapter_is_recognized_and_preserved(self):
        down_weight = np.arange(8, dtype=np.float32).reshape(4, 2) / 10.0
        up_weight = np.arange(8, dtype=np.float32).reshape(2, 4) / 20.0
        nodes = [
            helper.make_node(
                "MatMul", ["tokens", "down_weight"], ["down"], name="adapter_down"
            ),
            helper.make_node("Gelu", ["down"], ["activated"], name="adapter_gelu"),
            helper.make_node(
                "MatMul", ["activated", "up_weight"], ["up"], name="adapter_up"
            ),
            helper.make_node(
                "Add", ["tokens", "up"], ["result"], name="adapter_residual"
            ),
        ]
        path = _save_model(
            self.root,
            "static_adapter.onnx",
            nodes=nodes,
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 2, 4])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 2, 4])],
            initializers=[
                _initializer("down_weight", down_weight),
                _initializer("up_weight", up_weight),
            ],
        )

        compiler = OnnxCompiler(str(path), weight_dtype="float32")
        graph, weights = compiler.lower()

        features = compiler.publication_report["features"]["static_adapter"]
        self.assertEqual(len(features), 1)
        self.assertEqual(
            {key: features[0][key] for key in ("source_node", "input", "down", "up")},
            {
                "source_node": "adapter_residual",
                "input": "tokens",
                "down": "adapter_down",
                "up": "adapter_up",
            },
        )
        self.assertIsNone(features[0].get("down_bias"))
        self.assertIsNone(features[0].get("up_bias"))
        self.assertEqual(
            [node["opType"] for node in graph["nodes"]],
            ["Linear", "GELU", "Linear", "Add"],
        )
        self.assertEqual({value.dtype for value in weights.values()}, {np.dtype(np.float32)})

    def test_fp32_router_is_portable_and_publishes_atomically(self):
        nodes = [
            helper.make_node(
                "Unsqueeze", ["keep_mask", "axis_two"], ["keep_mask_3d"], name="expand_mask"
            ),
            helper.make_node(
                "Mul", ["tokens", "keep_mask_3d"], ["masked_tokens"], name="mask_tokens"
            ),
            helper.make_node(
                "ReduceSum",
                ["masked_tokens", "axis_one"],
                ["token_sum"],
                name="token_sum",
                keepdims=0,
            ),
            helper.make_node(
                "ReduceSum",
                ["keep_mask", "axis_one"],
                ["keep_count"],
                name="keep_count",
                keepdims=1,
            ),
            helper.make_node(
                "Clip", ["keep_count", "one"], ["safe_keep_count"],
                name="clamp_keep_count",
            ),
            helper.make_node(
                "Div", ["token_sum", "safe_keep_count"], ["pooled"], name="masked_mean"
            ),
            helper.make_node(
                "MatMul", ["pooled", "router_weight"], ["logits"], name="router_head"
            ),
            helper.make_node(
                "ArgMax",
                ["logits"],
                ["route"],
                name="route_argmax",
                axis=1,
                keepdims=0,
                select_last_index=0,
            ),
        ]
        path = _save_model(
            self.root,
            "fp32_router.onnx",
            nodes=nodes,
            inputs=[
                _value("tokens", TensorProto.FLOAT, [1, 3, 4]),
                _value("keep_mask", TensorProto.FLOAT, [1, 3]),
            ],
            outputs=[_value("route", TensorProto.INT64, [1])],
            initializers=[
                _initializer("axis_two", np.asarray([2], dtype=np.int64)),
                _initializer("axis_one", np.asarray([1], dtype=np.int64)),
                _initializer("one", np.asarray(1.0, dtype=np.float32)),
                _initializer(
                    "router_weight", np.arange(8, dtype=np.float32).reshape(4, 2) / 10.0
                ),
            ],
        )

        compiler = OnnxCompiler(
            str(path), weight_dtype="float32", output_dtypes={"route": "int32"}
        )
        graph, weights = compiler.lower()
        self.assertEqual(compiler.publication_report["package_class"], "fp32")
        self.assertEqual(len(compiler.publication_report["features"]["router"]), 1)
        argmax = next(node for node in graph["nodes"] if node["opType"] == "ArgMax")
        self.assertEqual(_output_descriptor(argmax)["dtype"], "int32")
        self.assertTrue(validate_graph(graph, ["browser"], weights=weights).supported)
        portable = validate_graph(graph, ["portable"], weights=weights)
        self.assertTrue(portable.supported)

        browser_directory = self.root / "browser"
        browser_directory.mkdir()
        browser_weights = browser_directory / "model.safetensors"
        browser_run = self.run_cli(
            "--model",
            path,
            "--out",
            browser_weights,
            "--target",
            "browser",
            "--weight-dtype",
            "float32",
            "--output-dtype",
            "route=int32",
        )
        self.assertEqual(browser_run.returncode, 0, browser_run.stderr)
        self.assertTrue(browser_weights.is_file())
        browser_graph = json.loads((browser_directory / "graph.json").read_text(encoding="utf-8"))
        self.assertNotIn("source", browser_graph)
        self.assertEqual(classify_package(browser_graph), "fp32")
        with safe_open(str(browser_weights), framework="numpy", device="cpu") as tensors:
            self.assertGreater(len(tensors.keys()), 0)

        portable_directory = self.root / "portable"
        portable_directory.mkdir()
        portable_weights = portable_directory / "model.safetensors"
        portable_graph = portable_directory / "graph.json"
        portable_run = self.run_cli(
            "--model",
            path,
            "--out",
            portable_weights,
            "--target",
            "portable",
            "--output-dtype",
            "route=int32",
        )
        self.assertEqual(portable_run.returncode, 0, portable_run.stderr)
        self.assertTrue(portable_weights.is_file())
        self.assertEqual(
            classify_package(json.loads(portable_graph.read_text(encoding="utf-8"))),
            "fp32",
        )

        unsafe_model = onnx.load(path)
        kept_nodes = [node for node in unsafe_model.graph.node if node.name != "clamp_keep_count"]
        del unsafe_model.graph.node[:]
        unsafe_model.graph.node.extend(kept_nodes)
        for node in unsafe_model.graph.node:
            if node.name == "masked_mean":
                node.input[1] = "keep_count"
        unsafe_path = self.root / "fp32_router_unclamped.onnx"
        onnx.checker.check_model(unsafe_model)
        onnx.save(unsafe_model, unsafe_path)
        unsafe_compiler = OnnxCompiler(
            str(unsafe_path), output_dtypes={"route": "int32"}
        )
        unsafe_graph, _ = unsafe_compiler.lower()
        self.assertNotIn("router", unsafe_compiler.publication_report["features"])
        self.assertNotIn("source", unsafe_graph)
        self.assert_no_stage_directories(portable_directory)

    def test_specialized_gather_removes_selector_and_family_bank(self):
        family_bank = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        path = _save_model(
            self.root,
            "specialized_family.onnx",
            nodes=[
                helper.make_node(
                    "Gather",
                    ["family_bank", "family_selector"],
                    ["selected_weight"],
                    name="select_family",
                    axis=0,
                ),
                helper.make_node(
                    "MatMul", ["tokens", "selected_weight"], ["result"], name="family_linear"
                ),
            ],
            inputs=[
                _value("tokens", TensorProto.FLOAT, [1, 3]),
                _value("family_selector", TensorProto.INT64, []),
            ],
            outputs=[_value("result", TensorProto.FLOAT, [1, 4])],
            initializers=[_initializer("family_bank", family_bank)],
        )

        compiler = OnnxCompiler(
            str(path), specialize_inputs={"family_selector": 1}
        )
        graph, weights = compiler.lower()

        self.assertEqual(list(graph["inputs"]), ["input0"])
        self.assertEqual(graph["inputs"]["input0"], {
            "shape": [1, 3], "dtype": "float32",
        })
        self.assertNotIn("Gather", [node["opType"] for node in graph["nodes"]])
        self.assertEqual(len(graph["nodes"]), 1)
        linear = graph["nodes"][0]
        selected = weights[linear["inputs"]["weight"]]
        np.testing.assert_array_equal(selected, family_bank[1])
        self.assertFalse(any(value.ndim == 3 for value in weights.values()))
        self.assertEqual(
            compiler.publication_report["abi_changes"],
            [
                {
                    "kind": "specialize-input",
                    "name": "family_selector",
                    "source": {"dtype": "int64", "shape": []},
                    "exported": {"removed": True, "value": 1},
                }
            ],
        )

    def test_scalar_gather_preserves_onnx_rank_via_slice_and_squeeze(self):
        path = _save_model(
            self.root,
            "scalar_gather.onnx",
            nodes=[
                helper.make_node(
                    "Gather",
                    ["values", "row"],
                    ["selected"],
                    name="select_row",
                    axis=0,
                ),
            ],
            inputs=[_value("values", TensorProto.FLOAT, [2, 3, 4])],
            outputs=[_value("selected", TensorProto.FLOAT, [3, 4])],
            initializers=[_initializer("row", np.asarray(1, dtype=np.int64))],
        )

        graph, weights = OnnxCompiler(str(path), weight_dtype="float32").lower()

        self.assertEqual(
            [node["opType"] for node in graph["nodes"]],
            ["Slice", "Squeeze"],
        )
        self.assertEqual(_output_descriptor(graph["nodes"][0])["shape"], [1, 3, 4])
        self.assertEqual(graph["nodes"][0]["params"], {
            "starts": [1],
            "ends": [2],
            "axes": [0],
            "steps": [1],
        })
        self.assertEqual(_output_descriptor(graph["nodes"][1])["shape"], [3, 4])
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

    def test_opset18_static_shape_value_chain_folds_from_bound_runtime_input(self):
        path = _save_model(
            self.root,
            "static_shape_value_chain.onnx",
            nodes=[
                helper.make_node(
                    "Shape", ["tokens"], ["full_shape"], name="runtime_shape"
                ),
                helper.make_node(
                    "Gather",
                    ["full_shape", "batch_index"],
                    ["batch_vector"],
                    name="gather_batch",
                    axis=0,
                ),
                helper.make_node(
                    "Squeeze",
                    ["batch_vector", "axis_zero"],
                    ["batch_scalar"],
                    name="squeeze_batch",
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["batch_scalar", "axis_zero"],
                    ["batch_again"],
                    name="unsqueeze_batch",
                ),
                helper.make_node(
                    "Slice",
                    [
                        "full_shape",
                        "slice_starts",
                        "slice_ends",
                        "axis_zero",
                        "slice_steps",
                    ],
                    ["shape_tail"],
                    name="slice_shape_tail",
                ),
                helper.make_node(
                    "Concat",
                    ["batch_again", "shape_tail"],
                    ["assembled_shape"],
                    name="concat_shape",
                    axis=0,
                ),
                helper.make_node(
                    "Reshape",
                    ["assembled_shape", "shape_vector_shape"],
                    ["target_shape"],
                    name="reshape_shape_value",
                ),
                helper.make_node(
                    "Reshape",
                    ["tokens", "target_shape"],
                    ["reshaped_once"],
                    name="reshape_runtime_data",
                ),
                helper.make_node(
                    "Shape",
                    ["reshaped_once"],
                    ["propagated_shape"],
                    name="shape_after_runtime_reshape",
                ),
                helper.make_node(
                    "Reshape",
                    ["reshaped_once", "propagated_shape"],
                    ["result"],
                    name="reshape_runtime_again",
                ),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, ["batch", "sequence", 4])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 3, 4])],
            initializers=[
                _initializer("batch_index", np.asarray([0], dtype=np.int64)),
                _initializer("axis_zero", np.asarray([0], dtype=np.int64)),
                _initializer("slice_starts", np.asarray([1], dtype=np.int64)),
                _initializer("slice_ends", np.asarray([3], dtype=np.int64)),
                _initializer("slice_steps", np.asarray([1], dtype=np.int64)),
                _initializer("shape_vector_shape", np.asarray([3], dtype=np.int64)),
            ],
            opset=18,
        )

        compiler = OnnxCompiler(
            str(path),
            weight_dtype="float32",
            input_shapes={"tokens": [1, 3, 4]},
        )
        graph, weights = compiler.lower()

        folded_nodes = {
            compiler.model.graph.node[index].name for index in compiler.folded
        }
        self.assertTrue(
            {
                "runtime_shape",
                "gather_batch",
                "squeeze_batch",
                "unsqueeze_batch",
                "slice_shape_tail",
                "concat_shape",
                "reshape_shape_value",
                "shape_after_runtime_reshape",
            }.issubset(folded_nodes)
        )
        np.testing.assert_array_equal(
            compiler.arrays["target_shape"],
            np.asarray([1, 3, 4], dtype=np.int64),
        )
        self.assertEqual(graph["inputs"]["input0"]["shape"], [1, 3, 4])
        self.assertEqual(
            [(node["opType"], source) for source, node in _nodes_by_source(compiler, graph).items()],
            [("Reshape", "reshape_runtime_again")],
        )
        self.assertEqual(graph["nodes"][0]["inputs"], {"input": "input0"})
        self.assertGreater(compiler.publication_report["typed_optimizer"]["total_changes"], 0)
        self.assertEqual(weights, {})

    def test_opset18_pytorch_association_of_exact_erf_gelu_is_recognized(self):
        path = _save_model(
            self.root,
            "pytorch_exact_erf_gelu.onnx",
            nodes=[
                helper.make_node(
                    "Add",
                    ["tokens", "bias"],
                    ["activation_input"],
                    name="activation_input",
                ),
                helper.make_node(
                    "Div",
                    ["activation_input", "sqrt_two"],
                    ["normalized"],
                    name="gelu_div",
                ),
                helper.make_node(
                    "Erf", ["normalized"], ["erf"], name="gelu_erf"
                ),
                helper.make_node(
                    "Add", ["erf", "one"], ["erf_plus_one"], name="gelu_add"
                ),
                helper.make_node(
                    "Mul",
                    ["half", "erf_plus_one"],
                    ["scaled_erf"],
                    name="gelu_scale",
                ),
                helper.make_node(
                    "Mul",
                    ["activation_input", "scaled_erf"],
                    ["result"],
                    name="gelu",
                ),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 2, 4])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 2, 4])],
            initializers=[
                _initializer("bias", np.zeros(4, dtype=np.float32)),
                _initializer("sqrt_two", np.asarray(np.sqrt(2.0), dtype=np.float32)),
                _initializer("one", np.asarray(1.0, dtype=np.float32)),
                _initializer("half", np.asarray(0.5, dtype=np.float32)),
            ],
            opset=18,
        )

        compiler = OnnxCompiler(
            str(path), weight_dtype="float32"
        )
        graph, weights = compiler.lower()

        self.assertEqual(
            compiler.publication_report["features"]["gelu"],
            [
                {
                    "source_node": "gelu",
                    "approximate": "none",
                    "source_pattern": "erf",
                }
            ],
        )
        self.assertEqual(
            [
                (node["opType"], node.get("params", {}))
                for node in graph["nodes"]
            ],
            [("Add", {}), ("GELU", {"approximate": "none"})],
        )
        self.assertEqual(len(weights), 1)

    def test_weight_only_qdq_is_materialized_as_canonical_fp32_linear(self):
        raw_weight = np.asarray(
            [[1, -2, 3, 4], [5, 6, -7, 8], [-9, 10, 11, -12]],
            dtype=np.int8,
        )
        scales = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        path = _save_model(
            self.root,
            "weight_only_qdq.onnx",
            nodes=[
                helper.make_node(
                    "DequantizeLinear",
                    ["raw_weight", "weight_scales", "weight_zero_points"],
                    ["weight_f32"],
                    name="weight_dequantize",
                    axis=1,
                ),
                helper.make_node(
                    "MatMul", ["tokens", "weight_f32"], ["result"], name="quantized_linear"
                ),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 3])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 4])],
            initializers=[
                _initializer("raw_weight", raw_weight),
                _initializer("weight_scales", scales),
                _initializer("weight_zero_points", np.zeros(4, dtype=np.int8)),
            ],
        )

        compiler = OnnxCompiler(str(path), weight_dtype="float32")
        graph, weights = compiler.lower()

        self.assertEqual(compiler.publication_report["package_class"], "fp32")
        self.assertEqual(len(graph["nodes"]), 1)
        linear = graph["nodes"][0]
        self.assertEqual(linear["opType"], "Linear")
        self.assertEqual(linear["params"]["weight_layout"], "din_dout")
        self.assertEqual(set(linear["inputs"]), {"input", "weight"})
        np.testing.assert_allclose(
            weights[linear["inputs"]["weight"]],
            raw_weight.astype(np.float32) * scales.reshape(1, -1),
        )
        self.assertEqual(
            compiler.publication_report["features"]["dequantized_weight_linear"][0]["source_node"],
            "quantized_linear",
        )

    def test_dynamic_lora_scale_is_rejected(self):
        path, *_ = _lora_model(self.root, dynamic_scale=True)

        error = self.assert_diagnostic(
            "VXLORA_SCALE", lambda: OnnxCompiler(str(path), weight_dtype="float32")
        )

        self.assertEqual(error.diagnostic.stage, "recognize-lora")
        self.assertIn("dynamic scale", error.diagnostic.message)

    def test_custom_domain_cannot_bypass_lora_residual_semantics(self):
        # A domain name cannot authorize producer-defined MatMul semantics.  A
        # real custom lowering registry must exist before such a graph reaches
        # feature recognition or residual-shape checks.
        custom_domain = "volvox.test.broadcast_lora"
        path = _save_model(
            self.root,
            "broadcast_lora.onnx",
            nodes=[
                helper.make_node(
                    "MatMul",
                    ["tokens", "base_weight"],
                    ["base"],
                    name="base_projection",
                    domain=custom_domain,
                ),
                helper.make_node(
                    "MatMul",
                    ["tokens", "lora_a_weight"],
                    ["low_rank"],
                    name="lora_a",
                    domain=custom_domain,
                ),
                helper.make_node(
                    "MatMul",
                    ["low_rank", "lora_b_weight"],
                    ["delta_raw"],
                    name="lora_b",
                    domain=custom_domain,
                ),
                helper.make_node(
                    "Mul", ["delta_raw", "scale"], ["delta"], name="lora_scale_mul"
                ),
                helper.make_node(
                    "Add", ["base", "delta"], ["result"], name="lora_residual"
                ),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 2, 3])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 2, 4])],
            initializers=[
                _initializer("base_weight", np.ones((3, 4), dtype=np.float32)),
                _initializer("lora_a_weight", np.ones((3, 2), dtype=np.float32)),
                _initializer("lora_b_weight", np.ones((2, 4), dtype=np.float32)),
                _initializer("scale", np.asarray(0.5, dtype=np.float32)),
            ],
            value_info=[
                _value("base", TensorProto.FLOAT, [1, 2, 4]),
                _value("low_rank", TensorProto.FLOAT, [1, 2, 2]),
                _value("delta_raw", TensorProto.FLOAT, [1, 1, 4]),
            ],
            extra_opsets=[(custom_domain, 1)],
        )

        error = self.assert_diagnostic(
            "VXONNX_DOMAIN",
            lambda: OnnxCompiler(
                str(path),
                weight_dtype="float32",
                registered_domains={custom_domain},
            ),
        )

        self.assertIn("standard ai.onnx", error.diagnostic.constraint)

    def test_training_dropout_is_rejected(self):
        path = _save_model(
            self.root,
            "training_dropout.onnx",
            nodes=[
                helper.make_node(
                    "Dropout",
                    ["tokens", "ratio", "training_mode"],
                    ["result"],
                    name="training_dropout",
                )
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, [1, 2, 4])],
            outputs=[_value("result", TensorProto.FLOAT, [1, 2, 4])],
            initializers=[
                _initializer("ratio", np.asarray(0.25, dtype=np.float32)),
                _initializer("training_mode", np.asarray(True, dtype=np.bool_)),
            ],
        )

        error = self.assert_diagnostic(
            "VXDROPOUT_TRAINING", lambda: OnnxCompiler(str(path)).lower()
        )

        self.assertEqual(error.diagnostic.source_node, "training_dropout")

    def test_argmax_select_last_index_is_rejected(self):
        path = _argmax_model(
            self.root, select_last_index=1, filename="argmax_select_last.onnx"
        )

        error = self.assert_diagnostic(
            "VXARGMAX_TIE",
            lambda: OnnxCompiler(
                str(path), output_dtypes={"route": "int32"}
            ).lower(),
        )

        self.assertEqual(error.diagnostic.constraint, "first-index ties")

    def test_public_i64_argmax_requires_explicit_int32_binding(self):
        path = _argmax_model(
            self.root, select_last_index=0, filename="public_i64_argmax.onnx"
        )

        error = self.assert_diagnostic(
            "VXARGMAX_PUBLIC_I64", lambda: OnnxCompiler(str(path)).lower()
        )
        self.assertEqual(error.diagnostic.constraint, "--output-dtype NAME=int32")

        graph, _weights = OnnxCompiler(
            str(path), output_dtypes={"route": "int32"}
        ).lower()
        self.assertEqual(_output_descriptor(graph["nodes"][0])["dtype"], "int32")

    def test_qdq_byte_logits_lower_to_canonical_qargmax(self):
        path = _save_model(
            self.root,
            "qdq_qargmax.onnx",
            nodes=[
                helper.make_node(
                    "DequantizeLinear",
                    ["raw_logits", "logits_scale", "logits_zero"],
                    ["logits"],
                    name="logits_dequantize",
                ),
                helper.make_node(
                    "ArgMax",
                    ["logits"],
                    ["route"],
                    name="route_argmax",
                    axis=1,
                    keepdims=0,
                    select_last_index=0,
                ),
            ],
            inputs=[_value("raw_logits", TensorProto.UINT8, [2, 3])],
            outputs=[_value("route", TensorProto.INT64, [2])],
            initializers=[
                _initializer("logits_scale", np.asarray(0.125, dtype=np.float32)),
                _initializer("logits_zero", np.asarray(117, dtype=np.uint8)),
            ],
        )

        error = self.assert_diagnostic(
            "VXARGMAX_PUBLIC_I64", lambda: OnnxCompiler(str(path)).lower()
        )
        self.assertEqual(error.diagnostic.constraint, "--output-dtype NAME=int32")

        compiler = OnnxCompiler(
            str(path), output_dtypes={"route": "int32"}
        )
        graph, weights = compiler.lower()

        self.assertEqual(len(weights), 2)
        self.assertEqual(compiler.publication_report["package_class"], "w8a8-v1")
        self.assertNotIn("source", graph)
        self.assertEqual(
            _resolved_quantization(graph, weights, "input0"),
            {"scheme": "per_tensor", "scale": 0.125, "zero_point": 117},
        )
        self.assertEqual(len(graph["nodes"]), 1)
        node = graph["nodes"][0]
        self.assertEqual(node["opType"], "QArgMax")
        self.assertEqual(node["inputs"], {"input": "input0"})
        self.assertEqual(node["params"], {"axis": 1})
        self.assertEqual(_output_descriptor(node)["shape"], [2])
        self.assertEqual(_output_descriptor(node)["dtype"], "int32")
        self.assertNotIn("outputs_quantization", node)
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

    def test_qdq_linear_with_exact_bias_lowers_to_canonical_qlinear(self):
        raw_weight = np.asarray(
            [[1, -2, 3, 4], [5, 6, -7, 8], [-9, 10, 11, -12]],
            dtype=np.int8,
        )
        weight_scales = np.asarray([0.5, 0.25, 1.0, 2.0], dtype=np.float32)
        bias_i32 = np.asarray([1, -2, 3, -4], dtype=np.int32)
        bias = np.multiply(
            bias_i32.astype(np.float32),
            np.multiply(np.float32(0.25), weight_scales, dtype=np.float32),
            dtype=np.float32,
        )
        path = _save_model(
            self.root,
            "qdq_qlinear.onnx",
            nodes=[
                helper.make_node(
                    "DequantizeLinear",
                    ["raw_tokens", "input_scale", "input_zero"],
                    ["tokens"],
                    name="input_dequantize",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    ["raw_weight", "weight_scales", "weight_zero_points"],
                    ["weight"],
                    name="weight_dequantize",
                    axis=1,
                ),
                helper.make_node(
                    "MatMul", ["tokens", "weight"], ["projection"], name="projection"
                ),
                helper.make_node(
                    "Add", ["projection", "bias"], ["biased"], name="projection_bias"
                ),
                helper.make_node(
                    "QuantizeLinear",
                    ["biased", "output_scale", "output_zero"],
                    ["raw_result"],
                    name="output_quantize",
                ),
            ],
            inputs=[_value("raw_tokens", TensorProto.UINT8, [1, 2, 3])],
            outputs=[_value("raw_result", TensorProto.UINT8, [1, 2, 4])],
            initializers=[
                _initializer("input_scale", np.asarray(0.25, dtype=np.float32)),
                _initializer("input_zero", np.asarray(128, dtype=np.uint8)),
                _initializer("raw_weight", raw_weight),
                _initializer("weight_scales", weight_scales),
                _initializer("weight_zero_points", np.zeros(4, dtype=np.int8)),
                _initializer("bias", bias),
                _initializer("output_scale", np.asarray(0.125, dtype=np.float32)),
                _initializer("output_zero", np.asarray(120, dtype=np.uint8)),
            ],
        )

        compiler = OnnxCompiler(str(path), weight_dtype="float32")
        graph, weights = compiler.lower()

        self.assertEqual(compiler.publication_report["package_class"], "w8a8-v1")
        self.assertNotIn("source", graph)
        self.assertEqual(
            _resolved_quantization(graph, weights, "input0"),
            {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
        )
        self.assertEqual(len(graph["nodes"]), 1)
        node = graph["nodes"][0]
        self.assertEqual(node["opType"], "QLinear")
        self.assertEqual(node["params"], {})
        self.assertEqual(_output_descriptor(node)["dtype"], "uint8")
        self.assertEqual(
            _resolved_quantization(graph, weights, _output_tensor(node)),
            {"scheme": "per_tensor", "scale": 0.125, "zero_point": 120},
        )
        preserved_weight = weights[node["inputs"]["weight"]]
        preserved_bias = weights[node["inputs"]["bias"]]
        np.testing.assert_array_equal(preserved_weight, raw_weight.T)
        np.testing.assert_array_equal(preserved_bias, bias_i32)
        self.assertEqual(
            _resolved_quantization(graph, weights, node["inputs"]["weight"]),
            {
                "scheme": "per_axis",
                "axis": 0,
                "scales": [0.5, 0.25, 1.0, 2.0],
                "zero_points": [0, 0, 0, 0],
            },
        )
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

        raw_tokens = np.asarray([[[128, 129, 127], [132, 124, 130]]], dtype=np.uint8)
        centered_input = raw_tokens.astype(np.int32) - 128
        centered_weight = raw_weight.astype(np.int32)
        accumulators = centered_input @ centered_weight + bias_i32
        multipliers = np.multiply(
            np.float32(0.25), weight_scales, dtype=np.float32
        ) / np.float32(0.125)
        expected = np.clip(
            np.rint(accumulators.astype(np.float32) * multipliers + np.float32(120)),
            0,
            255,
        ).astype(np.uint8)
        from onnx.reference import ReferenceEvaluator

        source_result = ReferenceEvaluator(str(path)).run(
            None, {"raw_tokens": raw_tokens}
        )[0]
        np.testing.assert_array_equal(expected, source_result)

        inexact_model = onnx.load(path)
        for initializer in inexact_model.graph.initializer:
            if initializer.name == "bias":
                initializer.CopyFrom(_initializer("bias", bias + np.float32(0.01)))
        inexact_path = self.root / "qdq_qlinear_inexact_bias.onnx"
        onnx.checker.check_model(inexact_model)
        onnx.save(inexact_model, inexact_path)
        hybrid_compiler = OnnxCompiler(str(inexact_path), weight_dtype="float32")
        hybrid_graph, _ = hybrid_compiler.lower()
        self.assertEqual(hybrid_compiler.publication_report["package_class"], "hybrid")
        self.assertNotIn("QLinear", [node["opType"] for node in hybrid_graph["nodes"]])

        overflow_model = onnx.load(path)
        overflow_bias = np.full(4, np.float32(2_147_483_008), dtype=np.float32)
        for initializer in overflow_model.graph.initializer:
            if initializer.name == "input_scale":
                initializer.CopyFrom(_initializer("input_scale", np.asarray(1.0, dtype=np.float32)))
            elif initializer.name == "weight_scales":
                initializer.CopyFrom(_initializer("weight_scales", np.ones(4, dtype=np.float32)))
            elif initializer.name == "bias":
                initializer.CopyFrom(_initializer("bias", overflow_bias))
        overflow_path = self.root / "qdq_qlinear_accumulator_overflow.onnx"
        onnx.checker.check_model(overflow_model)
        onnx.save(overflow_model, overflow_path)
        overflow_compiler = OnnxCompiler(str(overflow_path), weight_dtype="float32")
        overflow_graph, _ = overflow_compiler.lower()
        self.assertEqual(overflow_compiler.publication_report["package_class"], "hybrid")
        self.assertNotIn("QLinear", [node["opType"] for node in overflow_graph["nodes"]])

    def test_qdq_metadata_propagates_after_static_input_binding(self):
        path = _save_model(
            self.root,
            "qdq_bound_shape.onnx",
            nodes=[
                helper.make_node(
                    "DequantizeLinear",
                    ["raw", "input_scale", "input_zero"],
                    ["decoded"],
                    name="decode",
                ),
                helper.make_node("Relu", ["decoded"], ["activated"], name="relu"),
                helper.make_node(
                    "QuantizeLinear",
                    ["activated", "output_scale", "output_zero"],
                    ["result"],
                    name="encode",
                ),
            ],
            inputs=[_value("raw", TensorProto.UINT8, ["batch", 3])],
            outputs=[_value("result", TensorProto.UINT8, ["batch", 3])],
            initializers=[
                _initializer("input_scale", np.asarray(0.25, dtype=np.float32)),
                _initializer("input_zero", np.asarray(128, dtype=np.uint8)),
                _initializer("output_scale", np.asarray(0.5, dtype=np.float32)),
                _initializer("output_zero", np.asarray(120, dtype=np.uint8)),
            ],
        )

        compiler = OnnxCompiler(str(path), input_shapes={"raw": [2, 3]})
        self.assertEqual(compiler.shape_of("decoded"), [2, 3])
        self.assertEqual(compiler.dtype_of("decoded"), "float32")
        self.assertEqual(compiler.shape_of("result"), [2, 3])
        self.assertEqual(compiler.dtype_of("result"), "uint8")

        graph, weights = compiler.lower()
        self.assertEqual(
            [_output_descriptor(node)["shape"] for node in graph["nodes"]],
            [[2, 3], [2, 3], [2, 3]],
        )
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

    def test_shared_activation_dq_collapses_all_safe_qlinear_consumers(self):
        nodes = [
            helper.make_node(
                "DequantizeLinear",
                ["raw_tokens", "input_scale", "input_zero"],
                ["tokens"],
                name="input_dequantize",
            ),
        ]
        initializers = [
            _initializer("input_scale", np.asarray(0.25, dtype=np.float32)),
            _initializer("input_zero", np.asarray(128, dtype=np.uint8)),
        ]
        for index in range(2):
            nodes.extend([
                helper.make_node(
                    "DequantizeLinear",
                    [f"raw_weight_{index}", f"weight_scale_{index}", f"weight_zero_{index}"],
                    [f"weight_{index}"],
                    name=f"weight_dequantize_{index}",
                    axis=1,
                ),
                helper.make_node(
                    "MatMul",
                    ["tokens", f"weight_{index}"],
                    [f"projection_{index}"],
                    name=f"projection_{index}",
                ),
                helper.make_node(
                    "QuantizeLinear",
                    [f"projection_{index}", f"output_scale_{index}", f"output_zero_{index}"],
                    [f"result_{index}"],
                    name=f"output_quantize_{index}",
                ),
            ])
            initializers.extend([
                _initializer(
                    f"raw_weight_{index}",
                    np.asarray(
                        [[1 + index, -2], [3, 4 + index], [-5, 6]],
                        dtype=np.int8,
                    ),
                ),
                _initializer(
                    f"weight_scale_{index}",
                    np.asarray([0.5, 0.25], dtype=np.float32),
                ),
                _initializer(
                    f"weight_zero_{index}", np.zeros(2, dtype=np.int8)
                ),
                _initializer(
                    f"output_scale_{index}", np.asarray(0.125, dtype=np.float32)
                ),
                _initializer(
                    f"output_zero_{index}", np.asarray(120, dtype=np.uint8)
                ),
            ])
        path = _save_model(
            self.root,
            "shared_qlinear_activation.onnx",
            nodes=nodes,
            inputs=[_value("raw_tokens", TensorProto.UINT8, [1, 2, 3])],
            outputs=[
                _value("result_0", TensorProto.UINT8, [1, 2, 2]),
                _value("result_1", TensorProto.UINT8, [1, 2, 2]),
            ],
            initializers=initializers,
        )

        compiler = OnnxCompiler(str(path))
        graph, weights = compiler.lower()

        self.assertEqual(
            [node["opType"] for node in graph["nodes"]],
            ["QLinear", "QLinear"],
        )
        self.assertEqual(
            len(compiler.publication_report["features"]["w8a8_qlinear"]), 2
        )
        self.assertEqual(compiler.publication_report["package_class"], "w8a8-v1")
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

        unsafe = onnx.load(path)
        unsafe.graph.node.insert(
            1,
            helper.make_node(
                "Identity", ["tokens"], ["decoded_copy"], name="float_consumer"
            ),
        )
        unsafe.graph.output.extend([
            _value("decoded_copy", TensorProto.FLOAT, [1, 2, 3])
        ])
        unsafe_path = self.root / "shared_qlinear_with_float_consumer.onnx"
        onnx.checker.check_model(unsafe)
        onnx.save(unsafe, unsafe_path)

        hybrid_compiler = OnnxCompiler(str(unsafe_path))
        hybrid, hybrid_weights = hybrid_compiler.lower()
        hybrid_ops = [node["opType"] for node in hybrid["nodes"]]
        self.assertEqual(hybrid_ops.count("QLinear"), 2)
        self.assertEqual(hybrid_ops.count("DequantizeLinear"), 1)
        self.assertEqual(hybrid_ops.count("Identity"), 1)
        self.assertLess(
            hybrid_ops.index("DequantizeLinear"), hybrid_ops.index("Identity")
        )
        self.assertEqual(hybrid_compiler.publication_report["package_class"], "hybrid")
        self.assertTrue(
            validate_graph(hybrid, ["portable"], weights=hybrid_weights).supported
        )

    def test_qdq_gemm_lowers_raw_axis_zero_weight_and_bias_to_qgemm(self):
        raw_weight = np.asarray(
            [[1, -2, 3], [4, 5, -6], [-7, 8, 9]], dtype=np.int8
        )
        weight_scales = np.asarray([0.5, 0.25, 1.0], dtype=np.float32)
        input_scale = np.asarray(0.125, dtype=np.float32)
        bias_i32 = np.asarray([7, -8, 9], dtype=np.int32)
        bias_scales = np.multiply(input_scale, weight_scales, dtype=np.float32)

        def save(
            filename: str,
            *,
            trans_a: int = 0,
            trans_b: int = 1,
            alpha: float = 1.0,
            beta: float = 1.0,
            activation_dtype=np.uint8,
            weight_dtype=np.int8,
            weight_axis: int = 0,
            weight_scale_dtype=np.float32,
            bias_axis: int = 0,
            bias_scale_values=bias_scales,
            bias_zero_values=None,
            bias_values=bias_i32,
        ) -> Path:
            activation_type = (
                TensorProto.UINT8
                if np.dtype(activation_dtype) == np.dtype(np.uint8)
                else TensorProto.INT8
            )
            activation_zero = (
                np.asarray(128, dtype=np.uint8)
                if activation_type == TensorProto.UINT8
                else np.asarray(0, dtype=np.int8)
            )
            weight_values = raw_weight.astype(weight_dtype)
            weight_zero = np.zeros(3, dtype=weight_dtype)
            bias_zero = (
                np.zeros(3, dtype=np.int32)
                if bias_zero_values is None
                else np.asarray(bias_zero_values, dtype=np.int32)
            )
            return _save_model(
                self.root,
                filename,
                nodes=[
                    helper.make_node(
                        "DequantizeLinear",
                        ["raw_input", "input_scale", "input_zero"],
                        ["input"],
                        name="input_dequantize",
                    ),
                    helper.make_node(
                        "DequantizeLinear",
                        ["raw_weight", "weight_scales", "weight_zero"],
                        ["weight"],
                        name="weight_dequantize",
                        axis=weight_axis,
                    ),
                    helper.make_node(
                        "DequantizeLinear",
                        ["raw_bias", "bias_scales", "bias_zero"],
                        ["bias"],
                        name="bias_dequantize",
                        axis=bias_axis,
                    ),
                    helper.make_node(
                        "Gemm",
                        ["input", "weight", "bias"],
                        ["projection"],
                        name="projection",
                        transA=trans_a,
                        transB=trans_b,
                        alpha=alpha,
                        beta=beta,
                    ),
                    helper.make_node(
                        "QuantizeLinear",
                        ["projection", "output_scale", "output_zero"],
                        ["result"],
                        name="output_quantize",
                    ),
                ],
                inputs=[_value("raw_input", activation_type, [3, 3])],
                outputs=[_value("result", TensorProto.UINT8, [3, 3])],
                initializers=[
                    _initializer("input_scale", input_scale),
                    _initializer("input_zero", activation_zero),
                    _initializer("raw_weight", weight_values),
                    _initializer(
                        "weight_scales",
                        weight_scales.astype(weight_scale_dtype),
                    ),
                    _initializer("weight_zero", weight_zero),
                    _initializer("raw_bias", np.asarray(bias_values, dtype=np.int32)),
                    _initializer(
                        "bias_scales",
                        np.asarray(bias_scale_values, dtype=np.float32),
                    ),
                    _initializer("bias_zero", bias_zero),
                    _initializer(
                        "output_scale", np.asarray(0.25, dtype=np.float32)
                    ),
                    _initializer(
                        "output_zero", np.asarray(120, dtype=np.uint8)
                    ),
                ],
            )

        path = save("qdq_qgemm.onnx")
        compiler = OnnxCompiler(str(path))
        graph, weights = compiler.lower()

        self.assertEqual([node["opType"] for node in graph["nodes"]], ["QGemm"])
        node = graph["nodes"][0]
        np.testing.assert_array_equal(weights[node["inputs"]["weight"]], raw_weight)
        np.testing.assert_array_equal(weights[node["inputs"]["bias"]], bias_i32)
        self.assertEqual(
            _resolved_quantization(
                graph, weights, node["inputs"]["weight"]
            )["axis"], 0
        )
        self.assertEqual(len(compiler.publication_report["features"]["w8a8_qgemm"]), 1)
        self.assertEqual(compiler.publication_report["package_class"], "w8a8-v1")
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

        negative_axis_graph, negative_axis_weights = OnnxCompiler(str(save(
            "qdq_qgemm_negative_axes.onnx",
            weight_axis=-2,
            bias_axis=-1,
        ))).lower()
        self.assertEqual(
            [node["opType"] for node in negative_axis_graph["nodes"]],
            ["QGemm"],
        )
        self.assertTrue(validate_graph(
            negative_axis_graph,
            ["portable"],
            weights=negative_axis_weights,
        ).supported)

        invalid_paths = [
            save("qdq_qgemm_transa.onnx", trans_a=1),
            save("qdq_qgemm_transb.onnx", trans_b=0),
            save("qdq_qgemm_alpha.onnx", alpha=0.5),
            save("qdq_qgemm_beta.onnx", beta=0.5),
            save("qdq_qgemm_weight_axis.onnx", weight_axis=1),
            save(
                "qdq_qgemm_activation_dtype.onnx",
                activation_dtype=np.int8,
            ),
            save("qdq_qgemm_weight_dtype.onnx", weight_dtype=np.uint8),
            save(
                "qdq_qgemm_scale_dtype.onnx",
                weight_scale_dtype=np.float64,
            ),
            save(
                "qdq_qgemm_bias_scale.onnx",
                bias_scale_values=bias_scales + np.float32(0.01),
            ),
            save(
                "qdq_qgemm_bias_zero.onnx",
                bias_zero_values=np.asarray([0, 1, 0], dtype=np.int32),
            ),
            save(
                "qdq_qgemm_overflow.onnx",
                bias_values=np.full(3, 2**31 - 1, dtype=np.int32),
            ),
        ]
        for invalid_path in invalid_paths:
            with self.subTest(model=invalid_path.name):
                compiler = OnnxCompiler(str(invalid_path))
                self.assertNotIn(
                    "QGemm",
                    [region.op_type for region in compiler.qlinear_replacements.values()],
                )

    def test_dynamic_qdq_matmul_lowers_to_broadcast_qbatch_matmul(self):
        def save(
            filename: str,
            *,
            left_shape=(2, 1, 3, 4),
            right_shape=(1, 5, 4, 6),
            left_scale=0.25,
            right_scale=0.5,
            output_scale=0.125,
            add_bias: bool = False,
        ) -> Path:
            output_shape = list(
                np.broadcast_shapes(left_shape[:-2], right_shape[:-2])
            ) + [left_shape[-2], right_shape[-1]]
            nodes = [
                helper.make_node(
                    "DequantizeLinear",
                    ["raw_a", "a_scale", "a_zero"],
                    ["a"],
                    name="a_dequantize",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    ["raw_b", "b_scale", "b_zero"],
                    ["b"],
                    name="b_dequantize",
                ),
                helper.make_node("MatMul", ["a", "b"], ["product"], name="product"),
            ]
            quantized_input = "product"
            initializers = [
                _initializer("a_scale", np.asarray(left_scale, dtype=np.float32)),
                _initializer("a_zero", np.asarray(128, dtype=np.uint8)),
                _initializer("b_scale", np.asarray(right_scale, dtype=np.float32)),
                _initializer("b_zero", np.asarray(127, dtype=np.uint8)),
                _initializer(
                    "output_scale", np.asarray(output_scale, dtype=np.float32)
                ),
                _initializer("output_zero", np.asarray(120, dtype=np.uint8)),
            ]
            if add_bias:
                nodes.append(
                    helper.make_node(
                        "Add", ["product", "bias"], ["biased"], name="bias_add"
                    )
                )
                initializers.append(
                    _initializer(
                        "bias",
                        np.zeros(output_shape[-1], dtype=np.float32),
                    )
                )
                quantized_input = "biased"
            nodes.append(
                helper.make_node(
                    "QuantizeLinear",
                    [quantized_input, "output_scale", "output_zero"],
                    ["result"],
                    name="output_quantize",
                )
            )
            return _save_model(
                self.root,
                filename,
                nodes=nodes,
                inputs=[
                    _value("raw_a", TensorProto.UINT8, list(left_shape)),
                    _value("raw_b", TensorProto.UINT8, list(right_shape)),
                ],
                outputs=[_value("result", TensorProto.UINT8, output_shape)],
                initializers=initializers,
            )

        path = save("qdq_qbatch_matmul.onnx")
        compiler = OnnxCompiler(str(path))
        graph, weights = compiler.lower()

        self.assertEqual(len(weights), 6)
        self.assertEqual(
            [node["opType"] for node in graph["nodes"]], ["QBatchMatMul"]
        )
        node = graph["nodes"][0]
        self.assertEqual(node["inputs"], {"a": "input0", "b": "input1"})
        self.assertEqual(_output_descriptor(node)["shape"], [2, 5, 3, 6])
        self.assertEqual(
            _resolved_quantization(graph, weights, "input0"),
            {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
        )
        self.assertEqual(
            _resolved_quantization(graph, weights, "input1"),
            {"scheme": "per_tensor", "scale": 0.5, "zero_point": 127},
        )
        self.assertEqual(
            len(compiler.publication_report["features"]["w8a8_qbatch_matmul"]), 1
        )
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

        symbolic_path = _save_model(
            self.root,
            "qdq_qbatch_symbolic_output.onnx",
            nodes=[
                helper.make_node(
                    "DequantizeLinear", ["raw_a", "a_scale", "a_zero"], ["a"]
                ),
                helper.make_node(
                    "DequantizeLinear", ["raw_b", "b_scale", "b_zero"], ["b"]
                ),
                helper.make_node("MatMul", ["a", "b"], ["product"]),
                helper.make_node(
                    "QuantizeLinear",
                    ["product", "output_scale", "output_zero"],
                    ["result"],
                ),
            ],
            inputs=[
                _value("raw_a", TensorProto.UINT8, ["B", 8, "T", 40]),
                _value("raw_b", TensorProto.UINT8, ["B", 8, 40, "M"]),
            ],
            outputs=[
                _value("result", TensorProto.UINT8, ["B", 8, "T", "M"])
            ],
            initializers=[
                _initializer("a_scale", np.asarray(0.25, dtype=np.float32)),
                _initializer("a_zero", np.asarray(128, dtype=np.uint8)),
                _initializer("b_scale", np.asarray(0.5, dtype=np.float32)),
                _initializer("b_zero", np.asarray(127, dtype=np.uint8)),
                _initializer("output_scale", np.asarray(0.125, dtype=np.float32)),
                _initializer("output_zero", np.asarray(120, dtype=np.uint8)),
            ],
        )
        symbolic = OnnxCompiler(
            str(symbolic_path),
            dimension_bounds={
                "B": {"min": 1, "max": 1},
                "T": {"min": 1, "max": 192},
                "M": {"min": 211, "max": 402},
            },
        )
        symbolic_graph, symbolic_weights = symbolic.lower(
            enable_static_qdq_layout_optimization=False,
        )
        self.assertEqual(
            [node["opType"] for node in symbolic_graph["nodes"]],
            ["QBatchMatMul"],
        )
        self.assertEqual(
            _output_descriptor(symbolic_graph["nodes"][0])["shape"],
            ["B", 8, "T", "M"],
        )
        self.assertTrue(
            validate_graph(
                symbolic_graph, ["portable"], weights=symbolic_weights,
            ).supported
        )

        biased = OnnxCompiler(str(save("qdq_qbatch_bias.onnx", add_bias=True)))
        self.assertFalse(biased.qbatch_matmul_replacements)

        overflowing = OnnxCompiler(str(save(
            "qdq_qbatch_overflow.onnx",
            left_shape=(1, 1, 140_000),
            right_shape=(1, 140_000, 1),
        )))
        self.assertFalse(overflowing.qbatch_matmul_replacements)

        invalid_scale = OnnxCompiler(str(save(
            "qdq_qbatch_invalid_scale.onnx", left_scale=-0.25
        )))
        self.assertFalse(invalid_scale.qbatch_matmul_replacements)

        underflowing_multiplier = OnnxCompiler(str(save(
            "qdq_qbatch_multiplier_underflow.onnx",
            left_scale=1e-30,
            right_scale=1e-30,
            output_scale=1.0,
        )))
        self.assertFalse(underflowing_multiplier.qbatch_matmul_replacements)

        overflowing_multiplier = OnnxCompiler(str(save(
            "qdq_qbatch_multiplier_overflow.onnx",
            left_scale=1e30,
            right_scale=1e30,
            output_scale=1.0,
        )))
        self.assertFalse(overflowing_multiplier.qbatch_matmul_replacements)

        rank_nine = OnnxCompiler(str(save(
            "qdq_qbatch_rank_nine.onnx",
            left_shape=(1, 1, 1, 1, 1, 1, 1, 3, 4),
            right_shape=(1, 1, 1, 1, 1, 1, 1, 4, 6),
        )))
        self.assertFalse(rank_nine.qbatch_matmul_replacements)

    def test_qdq_conv_lowers_exact_u8s8_nchw_bytes_to_canonical_qconv(self):
        raw_weight = (
            np.arange(3 * 2 * 3 * 3, dtype=np.int16).reshape(3, 2, 3, 3)
            % 11 - 5
        ).astype(np.int8)
        weight_scales = np.asarray([0.25, 0.375, 0.5], dtype=np.float32)
        weight_zeros = np.asarray([0, 1, -2], dtype=np.int8)
        path = _qdq_conv_model(
            self.root,
            "qdq_qconv.onnx",
            raw_weight=raw_weight,
            weight_scales=weight_scales,
            weight_zero_points=weight_zeros,
        )

        compiler = OnnxCompiler(str(path))
        graph, weights = compiler.lower()

        self.assertEqual(
            [node["opType"] for node in graph["nodes"]],
            ["Transpose", "QConv2D", "Transpose"],
        )
        before, qconv, after = graph["nodes"]
        self.assertEqual(before["params"], {"perm": [0, 2, 3, 1]})
        self.assertEqual(_output_descriptor(before)["shape"], [1, 3, 4, 2])
        self.assertEqual(after["params"], {"perm": [0, 3, 1, 2]})
        self.assertEqual(_output_descriptor(after)["shape"], [1, 3, 3, 4])
        self.assertEqual(
            _resolved_quantization(graph, weights, _output_tensor(before)),
            {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
        )
        self.assertEqual(
            _resolved_quantization(graph, weights, _output_tensor(qconv)),
            _resolved_quantization(graph, weights, _output_tensor(after)),
        )
        self.assertEqual(
            qconv["params"],
            {
                "stride": [1, 1],
                "dilation": [1, 1],
                "groups": 1,
                "pads": [1, 1, 1, 1],
                "padding": [1, 1],
                "data_layout": "NHWC",
                "weight_layout": "OHWI",
            },
        )
        np.testing.assert_array_equal(
            weights[qconv["inputs"]["weight"]],
            np.transpose(raw_weight, (0, 2, 3, 1)),
        )
        np.testing.assert_array_equal(
            weights[qconv["inputs"]["bias"]],
            np.zeros(3, dtype=np.int32),
        )
        self.assertEqual(
            _resolved_quantization(graph, weights, qconv["inputs"]["weight"]),
            {
                "scheme": "per_axis",
                "axis": 0,
                "scales": [0.25, 0.375, 0.5],
                "zero_points": [0, 1, -2],
            },
        )
        self.assertEqual(len(compiler.publication_report["features"]["w8a8_qconv2d"]), 1)
        self.assertEqual(compiler.publication_report["package_class"], "w8a8-v1")
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

        raw_input = (
            np.arange(1 * 2 * 3 * 4, dtype=np.uint16).reshape(1, 2, 3, 4)
            + 119
        ).astype(np.uint8)
        accumulators = np.zeros((1, 3, 3, 4), dtype=np.int32)
        centered_input = raw_input.astype(np.int32) - 128
        for output_channel in range(3):
            centered_weight = (
                raw_weight[output_channel].astype(np.int32)
                - int(weight_zeros[output_channel])
            )
            for output_y in range(3):
                for output_x in range(4):
                    accumulator = 0
                    for input_channel in range(2):
                        for kernel_y in range(3):
                            input_y = output_y + kernel_y - 1
                            if input_y < 0 or input_y >= 3:
                                continue
                            for kernel_x in range(3):
                                input_x = output_x + kernel_x - 1
                                if input_x < 0 or input_x >= 4:
                                    continue
                                accumulator += (
                                    int(centered_input[0, input_channel, input_y, input_x])
                                    * int(centered_weight[input_channel, kernel_y, kernel_x])
                                )
                    accumulators[0, output_channel, output_y, output_x] = accumulator
        multipliers = (
            np.float32(0.25) * weight_scales / np.float32(0.125)
        )
        expected = np.clip(
            np.rint(
                accumulators.astype(np.float32)
                * multipliers.reshape(1, 3, 1, 1)
                + np.float32(120)
            ),
            0,
            255,
        ).astype(np.uint8)
        from onnx.reference import ReferenceEvaluator

        source_result = ReferenceEvaluator(str(path)).run(
            None, {"raw_input": raw_input}
        )[0]
        np.testing.assert_array_equal(expected, source_result)

    def test_qdq_conv_preserves_a_bounded_symbolic_batch(self):
        path = _qdq_conv_model(
            self.root,
            "qdq_qconv_symbolic_batch.onnx",
            input_shape=("B", 2, 3, 4),
        )
        compiler = OnnxCompiler(
            str(path), dimension_bounds={"B": {"min": 1, "max": 4}}
        )

        graph, weights = compiler.lower()

        self.assertEqual(
            [node["opType"] for node in graph["nodes"]],
            ["Transpose", "QConv2D", "Transpose"],
        )
        self.assertEqual(
            _output_descriptor(graph["nodes"][0])["shape"],
            ["B", 3, 4, 2],
        )
        self.assertEqual(
            _output_descriptor(graph["nodes"][1])["shape"],
            ["B", 3, 4, 3],
        )
        self.assertEqual(
            _output_descriptor(graph["nodes"][2])["shape"],
            ["B", 3, 3, 4],
        )
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

    def test_qdq_conv_rejects_noncanonical_descriptors_and_unsafe_regions(self):
        invalid_paths = [
            _qdq_conv_model(
                self.root, "qconv_activation_i8.onnx",
                activation_dtype=np.int8,
            ),
            _qdq_conv_model(
                self.root, "qconv_weight_u8.onnx",
                weight_dtype=np.uint8,
            ),
            _qdq_conv_model(
                self.root, "qconv_weight_axis.onnx", weight_axis=1
            ),
            _qdq_conv_model(
                self.root, "qconv_scale_dtype.onnx",
                weight_scale_dtype=np.float64,
            ),
            _qdq_conv_model(
                self.root, "qconv_negative_scale.onnx",
                weight_scales=np.asarray([-0.25, 0.375, 0.5], dtype=np.float32),
            ),
            _qdq_conv_model(
                self.root, "qconv_source_bias.onnx", include_bias=True
            ),
            _qdq_conv_model(
                self.root, "qconv_asymmetric_stride.onnx", strides=(1, 2)
            ),
            _qdq_conv_model(
                self.root, "qconv_dilated.onnx", dilations=(2, 2)
            ),
            _qdq_conv_model(
                self.root, "qconv_unpadded.onnx", pads=(0, 0, 0, 0)
            ),
            _qdq_conv_model(
                self.root,
                "qconv_grouped.onnx",
                raw_weight=np.ones((4, 1, 3, 3), dtype=np.int8),
                group=2,
            ),
            _qdq_conv_model(
                self.root,
                "qconv_multiplier_underflow.onnx",
                input_scale=1e-30,
                weight_scales=np.full(3, 1e-30, dtype=np.float32),
                output_scale=1.0,
            ),
            _qdq_conv_model(
                self.root,
                "qconv_multiplier_overflow.onnx",
                input_scale=1e30,
                weight_scales=np.full(3, 1e30, dtype=np.float32),
                output_scale=1.0,
            ),
            _qdq_conv_model(
                self.root,
                "qconv_accumulator_overflow.onnx",
                input_shape=(1, 20_000, 1, 1),
                raw_weight=np.full(
                    (1, 20_000, 3, 3), 127, dtype=np.int8
                ),
                weight_scales=np.ones(1, dtype=np.float32),
                weight_zero_points=np.zeros(1, dtype=np.int8),
            ),
        ]
        for path in invalid_paths:
            with self.subTest(model=path.name):
                self.assertFalse(OnnxCompiler(str(path)).qconv_replacements)

        fanout_path = _qdq_conv_model(
            self.root, "qconv_activation_fanout.onnx", float_fanout=True
        )
        fanout_graph, fanout_weights = OnnxCompiler(str(fanout_path)).lower()
        fanout_ops = [node["opType"] for node in fanout_graph["nodes"]]
        self.assertEqual(fanout_ops.count("QConv2D"), 1)
        self.assertEqual(fanout_ops.count("DequantizeLinear"), 1)
        self.assertEqual(fanout_ops.count("Identity"), 1)
        self.assertTrue(
            validate_graph(
                fanout_graph, ["portable"], weights=fanout_weights
            ).supported
        )

        output_fanout = onnx.load(
            _qdq_conv_model(self.root, "qconv_output_fanout_base.onnx")
        )
        quantize_index = next(
            index
            for index, node in enumerate(output_fanout.graph.node)
            if node.op_type == "QuantizeLinear"
        )
        output_fanout.graph.node.insert(
            quantize_index,
            helper.make_node(
                "Identity",
                ["convolution"],
                ["float_convolution"],
                name="float_convolution_consumer",
            ),
        )
        output_fanout.graph.output.extend([
            _value("float_convolution", TensorProto.FLOAT, [1, 3, 3, 4])
        ])
        output_fanout_path = self.root / "qconv_output_fanout.onnx"
        onnx.checker.check_model(output_fanout)
        onnx.save(output_fanout, output_fanout_path)
        self.assertFalse(
            OnnxCompiler(str(output_fanout_path)).qconv_replacements
        )

    def test_lora_recognition_normalizes_gemm_transposed_weights(self):
        path = _save_model(
            self.root,
            "gemm_transb_lora.onnx",
            nodes=[
                helper.make_node(
                    "Gemm", ["tokens", "base_weight"], ["base"],
                    name="base", transB=1,
                ),
                helper.make_node(
                    "Gemm", ["tokens", "a_weight"], ["low_rank"],
                    name="lora_a", transB=1,
                ),
                helper.make_node(
                    "Gemm", ["low_rank", "b_weight"], ["delta_raw"],
                    name="lora_b", transB=1,
                ),
                helper.make_node("Mul", ["delta_raw", "scale"], ["delta"], name="scale"),
                helper.make_node("Add", ["base", "delta"], ["result"], name="residual"),
            ],
            inputs=[_value("tokens", TensorProto.FLOAT, [2, 3])],
            outputs=[_value("result", TensorProto.FLOAT, [2, 4])],
            initializers=[
                _initializer("base_weight", np.ones((4, 3), dtype=np.float32)),
                _initializer("a_weight", np.ones((2, 3), dtype=np.float32)),
                _initializer("b_weight", np.ones((4, 2), dtype=np.float32)),
                _initializer("scale", np.asarray(0.25, dtype=np.float32)),
            ],
        )

        compiler = OnnxCompiler(str(path), weight_dtype="float32")
        graph, _ = compiler.lower()

        self.assertEqual(compiler.publication_report["features"]["lora"][0]["rank"], 2)
        self.assertEqual(
            [node["opType"] for node in graph["nodes"]],
            ["Linear", "Linear", "Linear", "Mul", "Add"],
        )

    def test_transpose_lowers_only_f32_and_i32_execution_storage(self):
        dtype_cases = {
            "float32": TensorProto.FLOAT,
            "int32": TensorProto.INT32,
            "int8": TensorProto.INT8,
            "uint8": TensorProto.UINT8,
        }
        model_paths = {}
        for dtype, onnx_dtype in dtype_cases.items():
            model_paths[dtype] = _save_model(
                self.root,
                f"transpose_{dtype}.onnx",
                nodes=[
                    helper.make_node(
                        "Transpose",
                        ["values"],
                        ["result"],
                        name=f"transpose_{dtype}",
                        perm=[1, 0],
                    )
                ],
                inputs=[_value("values", onnx_dtype, [2, 3])],
                outputs=[_value("result", onnx_dtype, [3, 2])],
            )

        for dtype in ("float32", "int32"):
            with self.subTest(dtype=dtype):
                graph, weights = OnnxCompiler(
                    str(model_paths[dtype]), weight_dtype="float32"
                ).lower()
                self.assertEqual(
                    [node["opType"] for node in graph["nodes"]],
                    ["Transpose"],
                )
                self.assertEqual(
                    _output_descriptor(graph["nodes"][0])["dtype"],
                    dtype,
                )
                self.assertTrue(
                    validate_graph(graph, ["portable"], weights=weights).supported
                )

        for dtype in ("int8", "uint8"):
            with self.subTest(dtype=dtype):
                error = self.assert_diagnostic(
                    "VXOPERAND_DTYPE",
                    lambda: OnnxCompiler(
                        str(model_paths[dtype]), weight_dtype="float32"
                    ).lower(),
                )
                self.assertEqual(error.diagnostic.source_op, "Transpose")
                self.assertIn(
                    f"has {dtype} execution dtype; expected float32, int32",
                    error.diagnostic.message,
                )

    def test_unsupported_pt_cli_does_not_replace_existing_package(self):
        source = self.root / "checkpoint.pt"
        source.write_bytes(b"not an ONNX graph")
        output_directory = self.root / "published"
        output_directory.mkdir()
        weights = output_directory / "model.safetensors"
        graph = output_directory / "graph.json"
        weights.write_bytes(b"existing weights")
        graph.write_text("existing graph", encoding="utf-8")

        completed = self.run_cli("--model", source, "--out", weights)

        self.assertEqual(completed.returncode, 1)
        self.assertIn("VXSOURCE001", completed.stderr)
        self.assertIn("accepts only .onnx and .tflite", completed.stderr)
        self.assertEqual(weights.read_bytes(), b"existing weights")
        self.assertEqual(graph.read_text(encoding="utf-8"), "existing graph")
        self.assert_no_stage_directories(output_directory)


if __name__ == "__main__":
    unittest.main()
