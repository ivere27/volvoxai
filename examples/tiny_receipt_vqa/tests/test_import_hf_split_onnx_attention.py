from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
importer = importlib.import_module("import_hf_split_onnx")


def _value(name: str, shape, dtype: int = TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def _attribute(node, name: str):
    return next(value for value in node.attribute if value.name == name)


def _node(model, name: str):
    return next(value for value in model.graph.node if value.name == name)


def _initializer(model, name: str):
    return next(value for value in model.graph.initializer if value.name == name)


def _set_initializer(model, name: str, array: np.ndarray) -> None:
    value = _initializer(model, name)
    value.CopyFrom(numpy_helper.from_array(array, name))


def _attention_model(*, quantized_projection: bool = False):
    """Author the exact producer attention layout in a compact synthetic graph."""

    nodes = []
    inputs = [
        _value("qkv_source", ["M", "B", 320]),
        _value("rank4_reference", ["B", 8, "M", 40]),
        _value("key_restore_reference", ["B", 8, 40, "M"]),
        _value("mask_source", ["B", 8, 1, "M"]),
        _value("projection_restore_reference", ["M", "B", 320]),
    ]
    value_infos = []
    initializers = [
        numpy_helper.from_array(
            np.asarray([0, -1, 40], dtype=np.int64), "qkv_flat_target"
        ),
        numpy_helper.from_array(
            np.asarray([-1, 320], dtype=np.int64), "projection_flat_target"
        ),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), "axis_0"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis_1"),
        numpy_helper.from_array(np.asarray([2], dtype=np.int64), "axis_2"),
        numpy_helper.from_array(np.asarray([3], dtype=np.int64), "axis_3"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "one_extent"),
    ]

    nodes.extend([
        helper.make_node(
            "Shape", ["rank4_reference"], ["rank4_target"],
            name="rank4_target_shape",
        ),
        helper.make_node(
            "Shape", ["key_restore_reference"], ["key_restore_target"],
            name="key_restore_target_shape",
        ),
        helper.make_node(
            "Shape", ["mask_source"], ["mask_rank4_target"],
            name="mask_rank4_target_shape",
        ),
        helper.make_node(
            "Shape", ["projection_restore_reference"],
            ["projection_restore_target"], name="projection_restore_target_shape",
        ),
        helper.make_node(
            "Shape", ["rank4_reference"], ["rank4_shape"],
            name="rank4_shape_value",
        ),
        helper.make_node(
            "Gather", ["rank4_shape", "axis_0"], ["batch_extent"],
            name="batch_extent_value", axis=0,
        ),
        helper.make_node(
            "Gather", ["rank4_shape", "axis_1"], ["head_extent"],
            name="head_extent_value", axis=0,
        ),
        helper.make_node(
            "Gather", ["rank4_shape", "axis_2"], ["memory_extent"],
            name="memory_extent_value", axis=0,
        ),
        helper.make_node(
            "Gather", ["rank4_shape", "axis_3"], ["width_extent"],
            name="width_extent_value", axis=0,
        ),
        helper.make_node(
            "Mul", ["batch_extent", "head_extent"], ["batch_head_extent"],
            name="batch_head_extent_value",
        ),
        helper.make_node(
            "Concat",
            ["batch_head_extent", "memory_extent", "width_extent"],
            ["key_flat_target"], name="key_flat_target_value", axis=0,
        ),
        helper.make_node(
            "Concat",
            ["batch_head_extent", "one_extent", "memory_extent"],
            ["mask_flat_target"], name="mask_flat_target_value", axis=0,
        ),
    ])

    qkv_final_names = []
    for index in range(18):
        flat = f"qkv_flat_{index}"
        transposed = f"qkv_transposed_{index}"
        final = f"qkv_final_{index}"
        nodes.extend([
            helper.make_node(
                "Reshape", ["qkv_source", "qkv_flat_target"], [flat],
                name=f"qkv_flatten_{index}", allowzero=0,
            ),
            helper.make_node(
                "Transpose", [flat], [transposed],
                name=f"qkv_transpose_{index}", perm=[1, 0, 2],
            ),
            helper.make_node(
                "Reshape", [transposed, "rank4_target"], [final],
                name=f"qkv_restore_{index}", allowzero=0,
            ),
        ])
        value_infos.extend([
            _value(flat, ["M", "8*B", 40]),
            _value(transposed, ["8*B", "M", 40]),
            _value(final, ["B", 8, "M", 40]),
        ])
        qkv_final_names.append(final)

    nodes.extend([
        helper.make_node(
            "Reshape", ["mask_source", "mask_flat_target"], ["mask_flat"],
            name="mask_flatten", allowzero=0,
        ),
        helper.make_node(
            "Reshape", ["mask_flat", "mask_rank4_target"], ["mask_restored"],
            name="mask_restore", allowzero=0,
        ),
    ])
    value_infos.extend([
        _value("mask_flat", ["8*B", 1, "M"]),
        _value("mask_restored", ["B", 8, 1, "M"]),
    ])

    for index in range(6):
        flat = f"key_flat_{index}"
        transposed = f"key_transposed_{index}"
        final = f"key_final_{index}"
        nodes.extend([
            helper.make_node(
                "Reshape", [qkv_final_names[index], "key_flat_target"], [flat],
                name=f"key_flatten_{index}", allowzero=0,
            ),
            helper.make_node(
                "Transpose", [flat], [transposed],
                name=f"key_transpose_{index}", perm=[0, 2, 1],
            ),
            helper.make_node(
                "Reshape", [transposed, "key_restore_target"], [final],
                name=f"key_restore_{index}", allowzero=0,
            ),
        ])
        value_infos.extend([
            _value(flat, ["8*B", "M", 40]),
            _value(transposed, ["8*B", 40, "M"]),
            _value(final, ["B", 8, 40, "M"]),
        ])

    graph_outputs = []
    base_weight = np.arange(320 * 320, dtype=np.float32).reshape(320, 320)
    for index in range(6):
        transposed = f"projection_transposed_{index}"
        flattened = f"projection_flat_{index}"
        activation = flattened
        nodes.extend([
            helper.make_node(
                "Transpose", [qkv_final_names[index + 6]], [transposed],
                name=f"projection_transpose_{index}", perm=[2, 0, 1, 3],
            ),
            helper.make_node(
                "Reshape", [transposed, "projection_flat_target"], [flattened],
                name=f"projection_flatten_{index}", allowzero=0,
            ),
        ])
        value_infos.extend([
            _value(transposed, ["M", "B", 8, 40]),
            _value(flattened, ["B*M", 320]),
        ])

        weight_name = f"layer.{index}.self_attn.out_proj.weight"
        if quantized_projection:
            activation_q = f"projection_activation_q_{index}"
            activation_dq = f"projection_activation_dq_{index}"
            activation_scale = f"projection_activation_scale_{index}"
            activation_zero = f"projection_activation_zero_{index}"
            initializers.extend([
                numpy_helper.from_array(
                    np.asarray(0.125, dtype=np.float32), activation_scale
                ),
                numpy_helper.from_array(
                    np.asarray(127, dtype=np.uint8), activation_zero
                ),
            ])
            nodes.extend([
                helper.make_node(
                    "QuantizeLinear",
                    [flattened, activation_scale, activation_zero],
                    [activation_q], name=f"projection_activation_q_node_{index}",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [activation_q, activation_scale, activation_zero],
                    [activation_dq], name=f"projection_activation_dq_node_{index}",
                ),
            ])
            value_infos.extend([
                _value(activation_q, ["B*M", 320], TensorProto.UINT8),
                _value(activation_dq, ["B*M", 320]),
            ])
            activation = activation_dq
            raw_weight = f"{weight_name}_quantized"
            weight_scale = f"{weight_name}_scale"
            weight_zero = f"{weight_name}_zero_point"
            initializers.extend([
                numpy_helper.from_array(
                    (base_weight.astype(np.int64) % 251 - 125).astype(np.int8),
                    raw_weight,
                ),
                numpy_helper.from_array(
                    np.full((320,), 0.01, dtype=np.float32), weight_scale
                ),
                numpy_helper.from_array(
                    np.zeros((320,), dtype=np.int8), weight_zero
                ),
            ])
            nodes.append(helper.make_node(
                "DequantizeLinear", [raw_weight, weight_scale, weight_zero],
                [weight_name], name=f"projection_weight_dq_{index}", axis=0,
            ))
            value_infos.append(_value(weight_name, [320, 320]))
        else:
            initializers.append(numpy_helper.from_array(
                base_weight + np.float32(index), weight_name
            ))

        bias_name = f"layer.{index}.self_attn.out_proj.bias"
        initializers.append(numpy_helper.from_array(
            np.arange(320, dtype=np.float32), bias_name
        ))
        gemm_output = f"projection_gemm_output_{index}"
        restored = f"projection_restored_{index}"
        final = f"projection_final_{index}"
        nodes.extend([
            helper.make_node(
                "Gemm", [activation, weight_name, bias_name], [gemm_output],
                name=f"projection_gemm_{index}",
                alpha=1.0, beta=1.0, transA=0, transB=1,
            ),
            helper.make_node(
                "Reshape", [gemm_output, "projection_restore_target"], [restored],
                name=f"projection_restore_{index}", allowzero=0,
            ),
            helper.make_node(
                "Transpose", [restored], [final],
                name=f"projection_final_transpose_{index}", perm=[1, 0, 2],
            ),
        ])
        value_infos.extend([
            _value(gemm_output, ["B*M", 320]),
            _value(restored, ["M", "B", 320]),
        ])
        graph_outputs.append(_value(final, ["B", "M", 320]))

    return helper.make_model(
        helper.make_graph(
            nodes, "qualified-encoder-attention", inputs, graph_outputs,
            initializer=initializers, value_info=value_infos,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )


class EncoderAttentionBatchRewriteTests(unittest.TestCase):
    def rewrite(self, model):
        return importer._rewrite_encoder_attention_batch_layout(model, onnx=onnx)

    def test_fp32_rewrites_every_qualified_family_and_is_deterministic(self):
        first = _attention_model()
        second = _attention_model()
        before = numpy_helper.to_array(
            _initializer(first, "layer.0.self_attn.out_proj.weight")
        ).copy()
        expected = {
            "attention_qkv_layouts_rewritten": 18,
            "attention_mask_layouts_rewritten": 1,
            "attention_key_transposes_rewritten": 6,
            "attention_output_projections_rewritten": 6,
        }
        self.assertEqual(self.rewrite(first), expected)
        self.assertEqual(self.rewrite(second), expected)
        self.assertEqual(first.SerializeToString(), second.SerializeToString())
        self.assertEqual(_node(first, "qkv_transpose_0").attribute[0].ints[:], [1, 2, 0, 3])
        self.assertEqual(_node(first, "mask_flatten").op_type, "Identity")
        self.assertEqual(_node(first, "key_transpose_0").attribute[0].ints[:], [0, 1, 3, 2])
        self.assertEqual(_node(first, "projection_gemm_0__rank3").op_type, "MatMul")
        np.testing.assert_array_equal(
            numpy_helper.to_array(
                _initializer(first, "layer.0.self_attn.out_proj.weight")
            ),
            before.T,
        )
        self.assertEqual(
            importer._derived_batch_dimensions(first, role="encoder", onnx=onnx),
            [],
        )

    def test_qdq_projection_transposes_private_axis_zero_weight(self):
        model = _attention_model(quantized_projection=True)
        raw_name = "layer.0.self_attn.out_proj.weight_quantized"
        before = numpy_helper.to_array(_initializer(model, raw_name)).copy()
        self.assertEqual(
            self.rewrite(model)["attention_output_projections_rewritten"], 6
        )
        np.testing.assert_array_equal(
            numpy_helper.to_array(_initializer(model, raw_name)), before.T
        )
        self.assertEqual(_attribute(_node(model, "projection_weight_dq_0"), "axis").i, 1)

    def test_exact_pattern_count_rejects_partial_match(self):
        model = _attention_model()
        output = next(
            value for value in model.graph.value_info if value.name == "qkv_final_17"
        )
        output.type.tensor_type.shape.dim[1].dim_value = 7
        with self.assertRaisesRegex(importer.ImportFailure, "Q/K/V layouts=17"):
            self.rewrite(model)

    def test_qkv_rejects_target_permutation_and_shared_edges(self):
        mutations = {
            "target": lambda model: _set_initializer(
                model, "qkv_flat_target", np.asarray([0, -1, 20], np.int64)
            ),
            "permutation": lambda model: _attribute(
                _node(model, "qkv_transpose_0"), "perm"
            ).ints.__setitem__(slice(None), [0, 1, 2]),
            "shared": lambda model: model.graph.node.append(helper.make_node(
                "Identity", ["qkv_flat_0"], ["qkv_flat_extra"], name="qkv_extra"
            )),
        }
        messages = {
            "target": "Q/K/V flatten target",
            "permutation": "Q/K/V transpose",
            "shared": "Q/K/V head layout is shared",
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                model = _attention_model()
                mutate(model)
                with self.assertRaisesRegex(importer.ImportFailure, messages[name]):
                    self.rewrite(model)

    def test_mask_rejects_target_and_shared_edges(self):
        for name in ("target", "shared"):
            with self.subTest(name=name):
                model = _attention_model()
                if name == "target":
                    model.graph.initializer.append(numpy_helper.from_array(
                        np.asarray([0, 0, 0, 0], np.int64), "bad_mask_target"
                    ))
                    _node(model, "mask_flatten").input[1] = "bad_mask_target"
                    message = "mask flatten target"
                else:
                    model.graph.node.append(helper.make_node(
                        "Identity", ["mask_flat"], ["mask_extra"], name="mask_extra"
                    ))
                    message = "must feed input 0 of one Reshape"
                with self.assertRaisesRegex(importer.ImportFailure, message):
                    self.rewrite(model)

    def test_key_rejects_permutation_shape_and_shared_edges(self):
        for name in ("permutation", "shape", "shared"):
            with self.subTest(name=name):
                model = _attention_model()
                if name == "permutation":
                    _attribute(_node(model, "key_transpose_0"), "perm").ints[:] = [0, 1, 2]
                    message = "key transpose permutation"
                elif name == "shape":
                    info = next(
                        value for value in model.graph.value_info
                        if value.name == "key_transposed_0"
                    )
                    info.type.tensor_type.shape.dim[1].dim_value = 39
                    message = "key transpose metadata"
                else:
                    model.graph.node.append(helper.make_node(
                        "Identity", ["key_flat_0"], ["key_extra"], name="key_extra"
                    ))
                    message = "must feed input 0 of one Transpose"
                with self.assertRaisesRegex(importer.ImportFailure, message):
                    self.rewrite(model)

    def test_projection_rejects_attrs_layout_and_shared_edges(self):
        for name in ("attribute", "layout", "shared"):
            with self.subTest(name=name):
                model = _attention_model()
                if name == "attribute":
                    _attribute(_node(model, "projection_gemm_0"), "beta").f = 0.5
                    message = "Gemm attributes"
                elif name == "layout":
                    _attribute(
                        _node(model, "projection_transpose_0"), "perm"
                    ).ints[:] = [0, 2, 1, 3]
                    message = "projection transpose"
                else:
                    model.graph.node.append(helper.make_node(
                        "Identity", ["projection_transposed_0"],
                        ["projection_extra"], name="projection_extra"
                    ))
                    message = "projection source layout is shared"
                with self.assertRaisesRegex(importer.ImportFailure, message):
                    self.rewrite(model)

    def test_qdq_projection_rejects_axis_scale_and_shared_raw_weight(self):
        for name in ("axis", "scale", "shared"):
            with self.subTest(name=name):
                model = _attention_model(quantized_projection=True)
                if name == "axis":
                    _attribute(_node(model, "projection_weight_dq_0"), "axis").i = 1
                    message = "axis-0 private"
                elif name == "scale":
                    _set_initializer(
                        model, "layer.0.self_attn.out_proj.weight_scale",
                        np.ones((319,), np.float32),
                    )
                    message = "DQ descriptor"
                else:
                    model.graph.node.append(helper.make_node(
                        "Identity",
                        ["layer.0.self_attn.out_proj.weight_quantized"],
                        ["raw_weight_extra"], name="raw_weight_extra_node",
                    ))
                    message = "axis-0 private"
                with self.assertRaisesRegex(importer.ImportFailure, message):
                    self.rewrite(model)

    def test_generated_tensor_namespace_collision_is_rejected(self):
        model = _attention_model()
        model.graph.initializer.append(numpy_helper.from_array(
            np.asarray(0, np.float32),
            "__volvox_encoder_attention_projection_0",
        ))
        with self.assertRaisesRegex(importer.ImportFailure, "tensor collision"):
            self.rewrite(model)

    def test_malformed_slice_shape_program_is_rejected_before_mutation(self):
        model = _attention_model()
        model.graph.initializer.extend([
            numpy_helper.from_array(np.asarray([0, -1, 40], np.int64), "slice_data"),
            numpy_helper.from_array(np.asarray([0, 1], np.int64), "slice_starts"),
            numpy_helper.from_array(np.asarray([2], np.int64), "slice_ends"),
            numpy_helper.from_array(np.asarray([0, 0], np.int64), "slice_axes"),
            numpy_helper.from_array(np.asarray([1, 1], np.int64), "slice_steps"),
        ])
        model.graph.node.append(helper.make_node(
            "Slice",
            ["slice_data", "slice_starts", "slice_ends", "slice_axes", "slice_steps"],
            ["malformed_slice_target"], name="malformed_slice_target_node",
        ))
        _node(model, "qkv_flatten_0").input[1] = "malformed_slice_target"
        with self.assertRaisesRegex(importer.ImportFailure, "parameter lengths differ"):
            self.rewrite(model)

    def test_unqualified_derived_batch_extent_is_never_erased(self):
        model = helper.make_model(helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"], name="identity")],
            "unsafe-derived-batch",
            [_value("input", ["2*B", 4])],
            [_value("output", ["2*B", 4])],
        ))
        with self.assertRaisesRegex(importer.ImportFailure, "qualified attention pattern"):
            self.rewrite(model)


class BatchContractTests(unittest.TestCase):
    def test_max_batch_validation_is_closed_and_producer_bounded(self):
        self.assertEqual(importer._validated_max_batch_size(1, 8), 1)
        self.assertEqual(importer._validated_max_batch_size(2, 8), 2)
        self.assertEqual(importer._validated_max_batch_size(16, 16), 16)
        for invalid in (0, 9, True, 2.0, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(importer.ImportFailure, "max batch size"):
                    importer._validated_max_batch_size(invalid, 8)
        with self.assertRaisesRegex(importer.ImportFailure, "max batch size"):
            importer._validated_max_batch_size(3, 2)

    def test_distinct_lane_parity_case_exercises_requested_batch(self):
        calls = []

        class SessionOptions:
            intra_op_num_threads = 0
            inter_op_num_threads = 0

        class InferenceSession:
            def __init__(self, *_args, **_kwargs):
                pass

            def run(self, _outputs, inputs):
                calls.append({name: value.copy() for name, value in inputs.items()})
                family_ids = inputs["family_ids"]
                return [family_ids.astype(np.float32).reshape(-1, 1)]

        fake_ort = types.SimpleNamespace(
            SessionOptions=SessionOptions,
            InferenceSession=InferenceSession,
        )
        with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            report = importer._verify_dynamic_authoring_parity(
                Path("original.onnx"), Path("normalized.onnx"),
                role="encoder", max_batch_size=2,
            )
        self.assertEqual(report["cases"][-1]["case"], "batch_max_distinct_lanes")
        self.assertEqual(report["cases"][-1]["B"], 2)
        original_b2, normalized_b2 = calls[-2:]
        self.assertEqual(original_b2["family_ids"].tolist(), [0, 1])
        self.assertFalse(np.array_equal(original_b2["image"][0], original_b2["image"][1]))
        self.assertFalse(np.array_equal(
            original_b2["question_ids"][0], original_b2["question_ids"][1]
        ))
        self.assertEqual(
            normalized_b2["question_position_ids"].shape,
            (2, report["cases"][-1]["Q"]),
        )


if __name__ == "__main__":
    unittest.main()
