from __future__ import annotations

import json
import unittest

import numpy as np

from tools.exporter.errors import ExporterError
from tools.exporter.capabilities import refresh_package_class
from tools.exporter.optimizer.typed_passes import (
    OutputArgMaxSpecialization,
    RedundantQDQPass,
    RuntimeDeadCodePass,
)
from tools.exporter.optimizer.typed_pipeline import (
    optimize_runtime_graph,
    optimize_runtime_package,
)
from tools.exporter.pipeline import PassGroup, VerifiedPipeline
from tools.exporter.runtime_ir import import_runtime_package


def qdq_graph(*, output_scale="s", output_zero="z"):
    tensors = {
        "s": np.asarray([0.25], dtype=np.float32),
        "z": np.asarray([0], dtype=np.int8),
        "s2": np.asarray([0.5], dtype=np.float32),
        "z2": np.asarray([1], dtype=np.int8),
        "dead_weight": np.asarray([1.0], dtype=np.float32),
    }
    quant = {
        "scheme": "per_tensor", "scale_tensor": "s",
        "zero_point_tensor": "z",
    }
    document = {
        "format": "volvox-graph/v1",
        "inputs": {"x": {"shape": [2], "dtype": "int8"}},
        "outputs": ["y"],
        "nodes": [
            {"id": "dq", "opType": "DequantizeLinear",
             "inputs": {"input": "x", "scale": "s", "zero_point": "z"},
             "outputs": {"out": "f"}, "outputs_shape": {"out": [2]},
             "outputs_dtype": {"out": "float32"}},
            {"id": "q", "opType": "QuantizeLinear",
             "inputs": {"input": "f", "scale": output_scale,
                        "zero_point": output_zero},
             "outputs": {"out": "y"}, "outputs_shape": {"out": [2]},
             "outputs_dtype": {"out": "int8"}},
            {"id": "dead", "opType": "Add",
             "inputs": {"a": "f", "b": "dead_weight"},
             "outputs": {"out": "unused"}, "outputs_shape": {"out": [2]},
             "outputs_dtype": {"out": "float32"}},
        ],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "x": dict(quant),
                "y": {
                    "scheme": "per_tensor", "scale_tensor": output_scale,
                    "zero_point_tensor": output_zero,
                },
            },
        },
    }
    return import_runtime_package(document, tensors)


def dequantized_logits_package(*, scale=np.float32(0.25)):
    tensors = {
        "logits_scale": np.asarray([scale], dtype=np.float32),
        "logits_zero": np.asarray([-3], dtype=np.int8),
    }
    document = {
        "format": "volvox-graph/v1",
        "source": {
            "package_class": "fp32",
            "abi_changes": [{
                "kind": "input-dtype", "name": "ids",
                "source": "int64", "exported": "int32",
            }],
        },
        "inputs": {
            "logits_byte": {"shape": [1, 2, 4], "dtype": "int8"},
        },
        "outputs": ["logits"],
        "nodes": [{
            "id": "logits.dequantize",
            "opType": "DequantizeLinear",
            "inputs": {
                "input": "logits_byte",
                "scale": "logits_scale",
                "zero_point": "logits_zero",
            },
            "outputs": {"out": "logits"},
            "outputs_shape": {"out": [1, 2, 4]},
            "outputs_dtype": {"out": "float32"},
        }],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "logits_byte": {
                    "scheme": "per_tensor",
                    "scale_tensor": "logits_scale",
                    "zero_point_tensor": "logits_zero",
                },
            },
        },
    }
    return document, tensors


def packed_qlinear_package(*, post_bias=True, duplicate_last_group=False):
    tensors = {
        "input_scale": np.asarray([0.125], dtype=np.float32),
        "input_zero": np.asarray([-3], dtype=np.int8),
        "packed_weight": np.asarray([
            [1, 2], [3, 4], [5, 6],
            [7, 8], [9, 10], [11, 12],
        ], dtype=np.int8),
        "weight_scale": np.asarray(
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float32,
        ),
        "weight_zero": np.asarray([0, 1, -1, 2, -2, 3], dtype=np.int8),
        "packed_bias_i32": np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int32),
        "output_scale": np.asarray([0.25], dtype=np.float32),
        "output_zero": np.asarray([121], dtype=np.uint8),
    }
    if post_bias:
        tensors["post_bias"] = np.asarray(
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32,
        )

    nodes = [
        {
            "id": "packed-qlinear", "opType": "QLinear",
            "inputs": {
                "input": "x", "weight": "packed_weight",
                "bias": "packed_bias_i32",
            },
            "outputs": {"out": "packed_byte"},
            "outputs_shape": {"out": [2, 6]},
            "outputs_dtype": {"out": "uint8"},
        },
        {
            "id": "packed-dq", "opType": "DequantizeLinear",
            "inputs": {
                "input": "packed_byte", "scale": "output_scale",
                "zero_point": "output_zero",
            },
            "outputs": {"out": "packed_float"},
            "outputs_shape": {"out": [2, 6]},
            "outputs_dtype": {"out": "float32"},
        },
    ]
    movement_input = "packed_float"
    if post_bias:
        nodes.append({
            "id": "post-dq-bias", "opType": "Add",
            # Reverse the usual operand order to exercise exact port rewiring.
            "inputs": {"a": "post_bias", "b": "packed_float"},
            "outputs": {"out": "biased"},
            "outputs_shape": {"out": [2, 6]},
            "outputs_dtype": {"out": "float32"},
        })
        movement_input = "biased"
    nodes.extend([
        {
            "id": "group-reshape", "opType": "Reshape",
            "inputs": {"input": movement_input},
            "outputs": {"out": "grouped"},
            "outputs_shape": {"out": [1, 2, 3, 2]},
            "outputs_dtype": {"out": "float32"},
        },
        {
            "id": "group-transpose", "opType": "Transpose",
            "inputs": {"input": "grouped"},
            "outputs": {"out": "groups_first"},
            "outputs_shape": {"out": [3, 2, 1, 2]},
            "outputs_dtype": {"out": "float32"},
            "params": {"perm": [2, 1, 0, 3]},
        },
    ])
    # Deliberately scramble leaf order; assignment must follow proven channel
    # indices, not Slice node order.
    for group in (1, 0, 2):
        selected = 1 if duplicate_last_group and group == 2 else group
        nodes.append({
            "id": f"slice-{group}", "opType": "Slice",
            "inputs": {"input": "groups_first"},
            "outputs": {"out": f"part{group}"},
            "outputs_shape": {"out": [1, 2, 1, 2]},
            "outputs_dtype": {"out": "float32"},
            "params": {"starts": [selected], "axes": [0], "steps": [1]},
        })

    document = {
        "format": "volvox-graph/v1",
        "inputs": {"x": {"shape": [2, 2], "dtype": "int8"}},
        "outputs": ["part0", "part1", "part2"],
        "nodes": nodes,
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "x": {
                    "scheme": "per_tensor",
                    "scale_tensor": "input_scale",
                    "zero_point_tensor": "input_zero",
                },
                "packed_weight": {
                    "scheme": "per_axis", "axis": 0,
                    "scale_tensor": "weight_scale",
                    "zero_point_tensor": "weight_zero",
                },
                "packed_byte": {
                    "scheme": "per_tensor",
                    "scale_tensor": "output_scale",
                    "zero_point_tensor": "output_zero",
                },
            },
        },
    }
    return document, tensors


class TypedOptimizerTests(unittest.TestCase):
    def test_typed_graph_entry_is_transactional_and_composable(self):
        document, tensors = dequantized_logits_package()
        graph = import_runtime_package(document, tensors)
        working_tensors = dict(tensors)
        before_graph = graph.fingerprint()
        before_tensors = {
            name: np.array(value, copy=True)
            for name, value in working_tensors.items()
        }

        with self.assertRaisesRegex(Exception, "not a public graph output"):
            optimize_runtime_graph(
                graph,
                working_tensors,
                output_argmax=(
                    OutputArgMaxSpecialization("missing", "token_ids"),
                ),
            )

        self.assertEqual(graph.fingerprint(), before_graph)
        self.assertEqual(set(working_tensors), set(before_tensors))
        for name, value in before_tensors.items():
            np.testing.assert_array_equal(working_tensors[name], value)

        report = optimize_runtime_graph(
            graph,
            working_tensors,
            output_argmax=(
                OutputArgMaxSpecialization("logits", "token_ids"),
            ),
        )
        self.assertEqual(report.total_changes, 2)
        self.assertEqual(graph.outputs, ["token_ids"])
        self.assertEqual([node.op_type for node in graph.nodes], ["QArgMax"])

    def test_package_entry_rejects_recursive_affine_params_and_empty_outputs(self):
        document = {
            "format": "volvox-graph/v1",
            "inputs": {"x": {"shape": [1], "dtype": "float32"}},
            "nodes": [{
                "id": "identity",
                "opType": "Identity",
                "inputs": {"input": "x"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [1]},
                "outputs_dtype": {"out": "float32"},
                "params": {
                    "scale": 0.5,
                    "private": [{"affine": {"input_scale": 0.25}}],
                },
            }],
            "outputs": ["y"],
        }
        with self.assertRaises(ExporterError) as retired:
            optimize_runtime_package(document, {})
        self.assertEqual(retired.exception.diagnostic.code, "VXRTIR023")

        document["nodes"][0]["params"] = {"scale": 0.5}
        document["outputs"] = []
        with self.assertRaises(ExporterError) as empty:
            optimize_runtime_package(document, {})
        self.assertEqual(empty.exception.diagnostic.code, "VXRTIR015")

    def test_post_ptq_publication_replaces_stale_package_class(self):
        document = {
            "format": "volvox-graph/v1",
            "source": {
                "package_class": "fp32",
                "quantized_graph_contract": "w8a8-v1",
            },
            "inputs": {
                "memory": {"shape": [1, 2, 3], "dtype": "float32"},
                "tokens": {"shape": [1, 2, 4], "dtype": "int8"},
            },
            "outputs": ["token_ids"],
            "nodes": [{
                "id": "select",
                "opType": "QArgMax",
                "inputs": {"input": "tokens"},
                "outputs": {"out": "token_ids"},
                "outputs_shape": {"out": [1, 2]},
                "outputs_dtype": {"out": "int32"},
                "params": {"axis": -1},
            }],
        }
        self.assertEqual(refresh_package_class(document, {}), "hybrid")
        self.assertEqual(document["source"]["package_class"], "hybrid")
        self.assertNotIn("quantized_graph_contract", document["source"])

    def test_explicit_output_argmax_removes_terminal_dq_and_records_abi(self):
        document, tensors = dequantized_logits_package()
        optimized, optimized_tensors, report = optimize_runtime_package(
            document,
            tensors,
            output_argmax=(OutputArgMaxSpecialization("logits", "token_ids"),),
        )

        self.assertEqual(optimized["outputs"], ["token_ids"])
        self.assertEqual([node["opType"] for node in optimized["nodes"]],
                         ["QArgMax"])
        qargmax = optimized["nodes"][0]
        self.assertEqual(qargmax["inputs"], {"input": "logits_byte"})
        self.assertEqual(qargmax["outputs"], {"out": "token_ids"})
        self.assertEqual(qargmax["outputs_shape"], {"out": [1, 2]})
        self.assertEqual(qargmax["outputs_dtype"], {"out": "int32"})
        self.assertEqual(qargmax["params"], {"axis": -1})
        self.assertEqual(set(optimized_tensors), {"logits_scale", "logits_zero"})

        abi_changes = optimized["source"]["abi_changes"]
        self.assertEqual(abi_changes[0]["kind"], "input-dtype")
        self.assertEqual(abi_changes[-1]["kind"], "output-specialization")
        self.assertEqual(abi_changes[-1]["source"]["name"], "logits")
        self.assertEqual(abi_changes[-1]["exported"]["name"], "token_ids")
        self.assertEqual(abi_changes[-1]["exported"]["tie_policy"], "first-index")
        self.assertEqual(optimized["source"]["package_class"], "w8a8-v1")
        self.assertEqual(
            optimized["source"]["quantized_graph_contract"], "w8a8-v1",
        )
        specialization_run = next(
            run for run in report.runs if run.name == "runtime-output-qargmax"
        )
        self.assertEqual(specialization_run.changes, 1)
        self.assertIn("logits", specialization_run.notes[0])

    def test_output_argmax_is_never_inferred_and_invalid_requests_fail_closed(self):
        document, tensors = dequantized_logits_package()
        unchanged, _, _ = optimize_runtime_package(document, tensors)
        self.assertEqual(unchanged["outputs"], ["logits"])
        self.assertEqual([node["opType"] for node in unchanged["nodes"]],
                         ["DequantizeLinear"])
        self.assertEqual(unchanged["source"]["package_class"], "hybrid")

        with self.assertRaisesRegex(Exception, "not a public graph output"):
            optimize_runtime_package(
                document,
                tensors,
                output_argmax=(OutputArgMaxSpecialization("missing", "token_ids"),),
            )

        invalid = dict(document)
        invalid["nodes"] = [dict(document["nodes"][0])]
        invalid["nodes"][0]["opType"] = "Identity"
        invalid["nodes"][0]["inputs"] = {"input": "logits_byte"}
        with self.assertRaises(ExporterError) as invalid_identity:
            optimize_runtime_package(
                invalid,
                tensors,
                output_argmax=(OutputArgMaxSpecialization("logits", "token_ids"),),
            )
        self.assertEqual(invalid_identity.exception.diagnostic.code, "VXRTIR025")

        non_monotonic_document, non_monotonic_tensors = dequantized_logits_package(
            scale=np.finfo(np.float32).max,
        )
        with self.assertRaisesRegex(Exception, "not proven strictly increasing"):
            optimize_runtime_package(
                non_monotonic_document,
                non_monotonic_tensors,
                output_argmax=(OutputArgMaxSpecialization("logits", "token_ids"),),
            )

    def test_positive_per_tensor_dequantization_preserves_argmax_and_first_ties(self):
        byte = np.asarray([[[-3, 7, 7, -8], [9, -1, 8, 9]]], dtype=np.int8)
        dequantized = (byte.astype(np.float32) - np.float32(-3)) * np.float32(0.25)
        np.testing.assert_array_equal(
            np.argmax(byte, axis=-1),
            np.argmax(dequantized, axis=-1),
        )

    def test_packed_qlinear_split_slices_every_quantized_operand_exactly(self):
        document, tensors = packed_qlinear_package()
        original_names = set(tensors)
        original_weight = tensors["packed_weight"].copy()
        original_weight_scale = tensors["weight_scale"].copy()
        original_weight_zero = tensors["weight_zero"].copy()
        original_accumulator_bias = tensors["packed_bias_i32"].copy()
        original_float_bias = tensors["post_bias"].copy()

        optimized, optimized_tensors, report = optimize_runtime_package(
            document, tensors,
        )

        # The entry point owns a private mutable inventory; authoring split
        # tensors must not mutate the caller's package data on success or error.
        self.assertEqual(set(tensors), original_names)
        split_run = next(
            run for run in report.runs
            if run.name == "runtime-packed-qlinear-split"
        )
        self.assertEqual(split_run.changes, 1)
        self.assertIn("3 exact OUT_IN projections", split_run.notes[0])

        op_types = [node["opType"] for node in optimized["nodes"]]
        self.assertEqual(op_types.count("QLinear"), 3)
        self.assertEqual(op_types.count("DequantizeLinear"), 3)
        self.assertEqual(op_types.count("Add"), 3)
        self.assertEqual(op_types.count("Reshape"), 3)
        self.assertNotIn("Slice", op_types)
        self.assertNotIn("Transpose", op_types)

        quantization = optimized["quantization"]["tensors"]
        qlinear_nodes = [
            node for node in optimized["nodes"] if node["opType"] == "QLinear"
        ]
        dequantize_nodes = {
            node["inputs"]["input"]: node for node in optimized["nodes"]
            if node["opType"] == "DequantizeLinear"
        }
        add_nodes = {
            node["outputs"]["out"]: node for node in optimized["nodes"]
            if node["opType"] == "Add"
        }
        reshape_nodes = {
            node["outputs"]["out"]: node for node in optimized["nodes"]
            if node["opType"] == "Reshape"
        }
        for group, qlinear in enumerate(qlinear_nodes):
            channel_slice = slice(group * 2, (group + 1) * 2)
            weight_name = qlinear["inputs"]["weight"]
            accumulator_bias_name = qlinear["inputs"]["bias"]
            np.testing.assert_array_equal(
                optimized_tensors[weight_name], original_weight[channel_slice, :],
            )
            np.testing.assert_array_equal(
                optimized_tensors[accumulator_bias_name],
                original_accumulator_bias[channel_slice],
            )
            weight_quantization = quantization[weight_name]
            self.assertEqual(weight_quantization["scheme"], "per_axis")
            self.assertEqual(weight_quantization["axis"], 0)
            np.testing.assert_array_equal(
                optimized_tensors[weight_quantization["scale_tensor"]],
                original_weight_scale[channel_slice],
            )
            np.testing.assert_array_equal(
                optimized_tensors[weight_quantization["zero_point_tensor"]],
                original_weight_zero[channel_slice],
            )

            byte_name = qlinear["outputs"]["out"]
            self.assertEqual(qlinear["outputs_shape"], {"out": [2, 2]})
            self.assertEqual(qlinear["outputs_dtype"], {"out": "uint8"})
            self.assertEqual(quantization[byte_name], {
                "scheme": "per_tensor",
                "scale_tensor": "output_scale",
                "zero_point_tensor": "output_zero",
            })
            dequantize = dequantize_nodes[byte_name]
            self.assertEqual(dequantize["inputs"]["scale"], "output_scale")
            self.assertEqual(dequantize["inputs"]["zero_point"], "output_zero")

            reshape = reshape_nodes[f"part{group}"]
            add = add_nodes[reshape["inputs"]["input"]]
            float_bias_name = next(
                name for name in add["inputs"].values()
                if name in optimized_tensors
                and optimized_tensors[name].dtype == np.dtype(np.float32)
                and optimized_tensors[name].shape == (2,)
            )
            np.testing.assert_array_equal(
                optimized_tensors[float_bias_name],
                original_float_bias[channel_slice],
            )
            # The source used bias as Add.a.  Keeping that port order proves
            # this is a post-DQ F32 Add, not an accumulator-domain fold.
            self.assertEqual(add["inputs"]["a"], float_bias_name)

        for old_name in (
            "packed_weight", "weight_scale", "weight_zero",
            "packed_bias_i32", "post_bias",
        ):
            self.assertNotIn(old_name, optimized_tensors)

    def test_packed_qlinear_split_handles_no_float_bias_and_fails_closed(self):
        document, tensors = packed_qlinear_package(post_bias=False)
        optimized, _, report = optimize_runtime_package(document, tensors)
        split_run = next(
            run for run in report.runs
            if run.name == "runtime-packed-qlinear-split"
        )
        self.assertEqual(split_run.changes, 1)
        self.assertNotIn("Add", [node["opType"] for node in optimized["nodes"]])

        duplicate, duplicate_tensors = packed_qlinear_package(
            duplicate_last_group=True,
        )
        unchanged, _, duplicate_report = optimize_runtime_package(
            duplicate, duplicate_tensors,
        )
        failed_closed = next(
            run for run in duplicate_report.runs
            if run.name == "runtime-packed-qlinear-split"
        )
        self.assertEqual(failed_closed.changes, 0)
        self.assertEqual(
            [node["opType"] for node in unchanged["nodes"]].count("QLinear"),
            1,
        )
        self.assertEqual(
            [node["opType"] for node in unchanged["nodes"]].count("Slice"),
            3,
        )

        source_slice, source_slice_tensors = packed_qlinear_package()
        first_slice = next(
            node for node in source_slice["nodes"] if node["opType"] == "Slice"
        )
        # `ends` is an ONNX-source attribute, not part of the canonical runtime
        # Slice contract.  Even a numerically plausible value must fail closed
        # instead of being ignored by the proof.
        first_slice["params"]["ends"] = [2]
        source_style, _, source_style_report = optimize_runtime_package(
            source_slice, source_slice_tensors,
        )
        source_style_split = next(
            run for run in source_style_report.runs
            if run.name == "runtime-packed-qlinear-split"
        )
        self.assertEqual(source_style_split.changes, 0)
        self.assertEqual(
            [node["opType"] for node in source_style["nodes"]].count("QLinear"),
            1,
        )

    def test_packed_qlinear_split_is_byte_stable_after_reoptimization(self):
        document, tensors = packed_qlinear_package()
        optimized, optimized_tensors, _ = optimize_runtime_package(
            document, tensors,
        )
        repeated, repeated_tensors, report = optimize_runtime_package(
            optimized, optimized_tensors,
        )

        split_run = next(
            run for run in report.runs
            if run.name == "runtime-packed-qlinear-split"
        )
        self.assertEqual(split_run.changes, 0)
        serialize = lambda value: (
            json.dumps(value, indent=1, allow_nan=False) + "\n"
        ).encode("utf-8")
        self.assertEqual(serialize(repeated), serialize(optimized))
        self.assertEqual(set(repeated_tensors), set(optimized_tensors))
        for name in optimized_tensors:
            np.testing.assert_array_equal(
                repeated_tensors[name], optimized_tensors[name],
            )

    def test_identical_qdq_becomes_identity_and_dce_prunes_dead_data(self):
        graph = qdq_graph()
        report = VerifiedPipeline([
            PassGroup((RedundantQDQPass(), RuntimeDeadCodePass()),
                      fixed_point=True, max_iterations=4),
        ]).run(graph)
        self.assertGreaterEqual(report.total_changes, 2)
        self.assertEqual([node.op_type for node in graph.nodes], ["Identity"])
        self.assertEqual(graph.nodes[0].input_map(), {"input": "x"})
        self.assertNotIn("f", graph.tensors)
        self.assertNotIn("dead_weight", graph.tensors)
        graph.verify()

    def test_mismatched_refs_do_not_cancel(self):
        graph = qdq_graph(output_scale="s2", output_zero="z2")
        report = VerifiedPipeline([RedundantQDQPass()]).run(graph)
        self.assertEqual(report.total_changes, 0)
        self.assertEqual(
            [node.op_type for node in graph.nodes],
            ["DequantizeLinear", "QuantizeLinear", "Add"],
        )

    def test_runtime_verifier_rejects_qdq_without_central_descriptor(self):
        graph = qdq_graph()
        graph.tensors["y"].quantization = None
        with self.assertRaises(Exception):
            # The runtime verifier itself prevents a malformed Q output before
            # the optimizer can make an unsafe decision.
            VerifiedPipeline([RedundantQDQPass()]).run(graph)

    def test_package_entry_runs_only_after_typed_import_and_prunes_weights(self):
        graph = qdq_graph()
        from tools.exporter.runtime_ir import export_runtime_package

        source_tensors = {
            name: value for name, value in {
                "s": np.asarray([0.25], dtype=np.float32),
                "z": np.asarray([0], dtype=np.int8),
                "s2": np.asarray([0.5], dtype=np.float32),
                "z2": np.asarray([1], dtype=np.int8),
                "dead_weight": np.asarray([1.0], dtype=np.float32),
            }.items()
        }
        document, tensors = export_runtime_package(graph, source_tensors)
        optimized, optimized_tensors, report = optimize_runtime_package(
            document, tensors)
        self.assertGreater(report.total_changes, 0)
        self.assertEqual([node["opType"] for node in optimized["nodes"]],
                         ["Identity"])
        self.assertEqual(set(optimized_tensors), {"s", "z"})

    def test_default_pipeline_composes_and_removes_shape_only_chain(self):
        document = {
            "format": "volvox-graph/v1",
            "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}},
            "nodes": [
                {"id": "first", "opType": "Reshape",
                 "inputs": {"input": "x"}, "outputs": {"out": "middle"},
                 "outputs_shape": {"out": [3, 2]},
                 "outputs_dtype": {"out": "float32"}, "params": {}},
                {"id": "second", "opType": "Reshape",
                 "inputs": {"input": "middle"}, "outputs": {"out": "restored"},
                 "outputs_shape": {"out": [2, 3]},
                 "outputs_dtype": {"out": "float32"}, "params": {}},
                {"id": "relu", "opType": "ReLU",
                 "inputs": {"input": "restored"}, "outputs": {"out": "y"},
                 "outputs_shape": {"out": [2, 3]},
                 "outputs_dtype": {"out": "float32"}, "params": {}},
            ],
            "outputs": ["y"],
        }
        optimized, tensors, report = optimize_runtime_package(document, {})
        self.assertGreaterEqual(report.total_changes, 2)
        self.assertEqual([node["opType"] for node in optimized["nodes"]],
                         ["ReLU"])
        self.assertEqual(optimized["nodes"][0]["inputs"], {"input": "x"})
        self.assertEqual(tensors, {})

    def test_shape_pass_batches_more_than_eight_independent_chain_rewrites(self):
        nodes = []
        outputs = []
        for branch in range(2):
            source = f"x{branch}"
            for index in range(12):
                destination = f"b{branch}_shape_{index}"
                shape = [2, 2] if index % 2 == 0 else [1, 4]
                nodes.append({
                    "id": f"b{branch}_reshape_{index}",
                    "opType": "Reshape",
                    "inputs": {"input": source},
                    "outputs": {"out": destination},
                    "outputs_shape": {"out": shape},
                    "outputs_dtype": {"out": "float32"},
                    "params": {},
                })
                source = destination
            output = f"y{branch}"
            nodes.append({
                "id": f"b{branch}_relu",
                "opType": "ReLU",
                "inputs": {"input": source},
                "outputs": {"out": output},
                "outputs_shape": {"out": [1, 4]},
                "outputs_dtype": {"out": "float32"},
                "params": {},
            })
            outputs.append(output)
        document = {
            "format": "volvox-graph/v1",
            "inputs": {
                "x0": {"shape": [1, 4], "dtype": "float32"},
                "x1": {"shape": [1, 4], "dtype": "float32"},
            },
            "outputs": outputs,
            "nodes": nodes,
        }

        optimized, _, report = optimize_runtime_package(document, {})

        shape_runs = [
            run for run in report.runs if run.name == "runtime-shape-chain"
        ]
        self.assertGreater(shape_runs[0].changes, 8)
        self.assertEqual(shape_runs[0].changes, 22)
        self.assertEqual(shape_runs[-1].changes, 0)
        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["ReLU", "ReLU"],
        )
        self.assertEqual(
            [node["inputs"] for node in optimized["nodes"]],
            [{"input": "x0"}, {"input": "x1"}],
        )

    def test_default_pipeline_rejects_unregistered_runtime_operator(self):
        document = {
            "format": "volvox-graph/v1",
            "inputs": {"x": {"shape": [1], "dtype": "float32"}},
            "nodes": [{
                "id": "mystery", "opType": "ProducerPrivateOp",
                "inputs": {"input": "x"}, "outputs": {"out": "y"},
                "outputs_shape": {"out": [1]},
                "outputs_dtype": {"out": "float32"}, "params": {},
            }],
            "outputs": ["y"],
        }
        with self.assertRaises(Exception):
            optimize_runtime_package(document, {})


if __name__ == "__main__":
    unittest.main()
