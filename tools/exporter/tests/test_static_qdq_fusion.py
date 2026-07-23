from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.exporter.optimizer.static_qdq_fusion import (
    RuntimeStaticQDQComputeFusionPass,
)
from tools.exporter.optimizer.typed_pipeline import optimize_runtime_package
from tools.exporter.reference_executor import ReferenceExecutor
from tools.exporter.runtime_ir import import_runtime_package


def _node(identifier, op_type, inputs, output, shape, dtype, params=None):
    return {
        "id": identifier,
        "opType": op_type,
        "inputs": dict(inputs),
        "outputs": {"out": output},
        "outputs_shape": {"out": list(shape)},
        "outputs_dtype": {"out": dtype},
        "params": dict(params or {}),
    }


def _affine(scale, zero):
    return {
        "scheme": "per_tensor",
        "scale_tensor": scale,
        "zero_point_tensor": zero,
    }


def _unary_package(op_type, *, shape=(2, 4), params=None):
    tensors = {
        "input_scale": np.asarray([0.125], dtype=np.float32),
        "input_zero": np.asarray([-3], dtype=np.int8),
        "output_scale": np.asarray([0.0625], dtype=np.float32),
        "output_zero": np.asarray([7], dtype=np.uint8),
    }
    inputs = {"input": "input_float"}
    nodes = [
        _node(
            "dq", "DequantizeLinear",
            {"input": "input_byte", "scale": "input_scale",
             "zero_point": "input_zero"},
            "input_float", shape, "float32",
        ),
    ]
    if op_type in {"LayerNorm", "GroupNorm"}:
        channels = shape[-1]
        tensors["weight"] = np.linspace(
            0.75, 1.25, channels, dtype=np.float32,
        )
        tensors["bias"] = np.linspace(
            -0.25, 0.25, channels, dtype=np.float32,
        )
        inputs.update({"weight": "weight", "bias": "bias"})
    nodes.extend((
        _node("compute", op_type, inputs, "float_output", shape, "float32", params),
        _node(
            "q", "QuantizeLinear",
            {"input": "float_output", "scale": "output_scale",
             "zero_point": "output_zero"},
            "output_byte", shape, "uint8",
        ),
    ))
    document = {
        "format": "volvox-graph/v1",
        "inputs": {"input_byte": {"shape": list(shape), "dtype": "int8"}},
        "nodes": nodes,
        "outputs": ["output_byte"],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "input_byte": _affine("input_scale", "input_zero"),
                "output_byte": _affine("output_scale", "output_zero"),
            },
        },
    }
    return document, tensors


def _add_package(*, right_shape=(2, 4), params=None):
    shape = (2, 4)
    tensors = {
        "left_scale": np.asarray([0.125], dtype=np.float32),
        "left_zero": np.asarray([-3], dtype=np.int8),
        "right_scale": np.asarray([0.25], dtype=np.float32),
        "right_zero": np.asarray([129], dtype=np.uint8),
        "output_scale": np.asarray([0.0625], dtype=np.float32),
        "output_zero": np.asarray([5], dtype=np.uint8),
    }
    document = {
        "format": "volvox-graph/v1",
        "inputs": {
            "left_byte": {"shape": list(shape), "dtype": "int8"},
            "right_byte": {"shape": list(right_shape), "dtype": "uint8"},
        },
        "nodes": [
            _node(
                "left_dq", "DequantizeLinear",
                {"input": "left_byte", "scale": "left_scale",
                 "zero_point": "left_zero"},
                "left_float", shape, "float32",
            ),
            _node(
                "right_dq", "DequantizeLinear",
                {"input": "right_byte", "scale": "right_scale",
                 "zero_point": "right_zero"},
                "right_float", right_shape, "float32",
            ),
            _node(
                "add", "Add", {"a": "left_float", "b": "right_float"},
                "sum_float", shape, "float32", params,
            ),
            _node(
                "q", "QuantizeLinear",
                {"input": "sum_float", "scale": "output_scale",
                 "zero_point": "output_zero"},
                "sum_byte", shape, "uint8",
            ),
        ],
        "outputs": ["sum_byte"],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "left_byte": _affine("left_scale", "left_zero"),
                "right_byte": _affine("right_scale", "right_zero"),
                "sum_byte": _affine("output_scale", "output_zero"),
            },
        },
    }
    return document, tensors


class StaticQDQComputeFusionTests(unittest.TestCase):
    def test_requires_explicit_numerical_migration_opt_in(self):
        with self.assertRaisesRegex(ValueError, "explicit numerical-migration"):
            RuntimeStaticQDQComputeFusionPass(allow_numerical_migration=False)

        document, tensors = _unary_package("GELU", params={"approximate": "none"})
        optimized, _, report = optimize_runtime_package(document, tensors)
        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["DequantizeLinear", "GELU", "QuantizeLinear"],
        )
        self.assertNotIn(
            "runtime-static-qdq-compute-fusion",
            [run.name for run in report.runs],
        )

    def test_fuses_every_supported_canonical_compute_island(self):
        cases = (
            ("GELU", (2, 4), {"approximate": "none"}, "QGELU"),
            ("SiLU", (2, 4), {}, "QSiLU"),
            ("LayerNorm", (2, 4), {"eps": 1e-5, "d_model": 4}, "QLayerNorm"),
            ("GroupNorm", (1, 2, 3, 4),
             {"num_groups": 2, "eps": 1e-5, "data_layout": "NHWC"},
             "QGroupNorm"),
        )
        for op_type, shape, params, expected in cases:
            with self.subTest(op_type=op_type):
                document, tensors = _unary_package(
                    op_type, shape=shape, params=params,
                )
                original_payloads = {
                    name: value.copy() for name, value in tensors.items()
                }
                optimized, optimized_tensors, report = optimize_runtime_package(
                    document,
                    tensors,
                    allow_static_qdq_compute_numerical_migration=True,
                )

                self.assertEqual(
                    [node["opType"] for node in optimized["nodes"]], [expected],
                )
                fused = optimized["nodes"][0]
                self.assertEqual(fused["inputs"]["input"], "input_byte")
                self.assertEqual(fused["outputs"], {"out": "output_byte"})
                self.assertEqual(
                    optimized["quantization"]["tensors"]["input_byte"],
                    document["quantization"]["tensors"]["input_byte"],
                )
                self.assertEqual(
                    optimized["quantization"]["tensors"]["output_byte"],
                    document["quantization"]["tensors"]["output_byte"],
                )
                for name, value in original_payloads.items():
                    self.assertIn(name, optimized_tensors)
                    np.testing.assert_array_equal(optimized_tensors[name], value)
                    np.testing.assert_array_equal(tensors[name], value)
                import_runtime_package(optimized, optimized_tensors)
                fusion_run = next(
                    run for run in report.runs
                    if run.name == "runtime-static-qdq-compute-fusion"
                )
                self.assertEqual(fusion_run.changes, 1)
                self.assertEqual(
                    dict(fusion_run.metrics),
                    {
                        "static_qdq_candidates_considered": 1,
                        "static_qdq_candidates_fused": 1,
                        "static_qdq_candidates_refused": 0,
                    },
                )
                self.assertIn("numerical migration", fusion_run.notes[0])
                self.assertIn("changed no initializer payload", fusion_run.notes[0])

    def test_fuses_exact_shape_two_dequantize_add(self):
        document, tensors = _add_package(params={"relu": 2})
        original = {name: value.copy() for name, value in tensors.items()}
        optimized, optimized_tensors, report = optimize_runtime_package(
            document,
            tensors,
            allow_static_qdq_compute_numerical_migration=True,
        )
        self.assertEqual([node["opType"] for node in optimized["nodes"]], ["QAdd"])
        self.assertEqual(optimized["nodes"][0]["inputs"], {
            "a": "left_byte", "b": "right_byte",
        })
        self.assertEqual(optimized["nodes"][0]["params"], {"relu": 2})
        for name, value in original.items():
            np.testing.assert_array_equal(optimized_tensors[name], value)
        import_runtime_package(optimized, optimized_tensors)
        fusion_run = next(
            run for run in report.runs
            if run.name == "runtime-static-qdq-compute-fusion"
        )
        self.assertEqual(fusion_run.changes, 1)

    def test_lowers_static_broadcast_to_descriptor_preserving_expand_and_qadd(self):
        broadcast, broadcast_tensors = _add_package(right_shape=(1, 4))
        inputs = {
            "left_byte": np.asarray(
                [[-3, -2, 1, 5], [7, -8, 12, 0]], dtype=np.int8,
            ),
            "right_byte": np.asarray([[129, 130, 127, 140]], dtype=np.uint8),
        }
        expected = ReferenceExecutor(
            import_runtime_package(copy.deepcopy(broadcast), broadcast_tensors),
            broadcast_tensors,
        ).run(inputs).outputs["sum_byte"]
        original_payloads = {
            name: value.copy() for name, value in broadcast_tensors.items()
        }
        optimized, optimized_tensors, report = optimize_runtime_package(
            broadcast,
            broadcast_tensors,
            allow_static_qdq_compute_numerical_migration=True,
        )
        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["Expand", "QAdd"],
        )
        expand, qadd = optimized["nodes"]
        self.assertEqual(expand["inputs"], {"input": "right_byte"})
        expanded_name = expand["outputs"]["out"]
        self.assertEqual(expand["outputs_shape"]["out"], [2, 4])
        self.assertEqual(qadd["inputs"], {
            "a": "left_byte", "b": expanded_name,
        })
        self.assertEqual(
            optimized["quantization"]["tensors"][expanded_name],
            optimized["quantization"]["tensors"]["right_byte"],
        )
        for name, value in original_payloads.items():
            np.testing.assert_array_equal(optimized_tensors[name], value)
        actual = ReferenceExecutor(
            import_runtime_package(optimized, optimized_tensors),
            optimized_tensors,
        ).run(inputs).outputs["sum_byte"]
        np.testing.assert_array_equal(actual, expected)
        run = next(
            item for item in report.runs
            if item.name == "runtime-static-qdq-compute-fusion"
        )
        self.assertEqual(run.changes, 1)

    def test_refuses_shared_and_noncanonical_attributes(self):

        shared, shared_tensors = _unary_package("GELU")
        shared["nodes"].insert(2, _node(
            "shared", "Identity", {"input": "input_float"},
            "shared_float", (2, 4), "float32",
        ))
        shared["outputs"].append("shared_float")
        unchanged, _, shared_report = optimize_runtime_package(
            shared,
            shared_tensors,
            allow_static_qdq_compute_numerical_migration=True,
        )
        self.assertIn("GELU", [node["opType"] for node in unchanged["nodes"]])
        self.assertNotIn("QGELU", [node["opType"] for node in unchanged["nodes"]])
        shared_run = next(
            run for run in shared_report.runs
            if run.name == "runtime-static-qdq-compute-fusion"
        )
        self.assertEqual(
            dict(shared_run.metrics),
            {
                "static_qdq_candidates_considered": 1,
                "static_qdq_candidates_fused": 0,
                "static_qdq_candidates_refused": 1,
            },
        )
        self.assertEqual(shared_run.diagnostics[0].code, "VXQF003")
        self.assertIn("shared", shared_run.diagnostics[0].message)

        approximate, approximate_tensors = _unary_package(
            "GELU", params={"approximate": "tanh"},
        )
        unchanged, _, approximate_report = optimize_runtime_package(
            approximate,
            approximate_tensors,
            allow_static_qdq_compute_numerical_migration=True,
        )
        self.assertIn("GELU", [node["opType"] for node in unchanged["nodes"]])
        self.assertNotIn("QGELU", [node["opType"] for node in unchanged["nodes"]])
        approximate_run = next(
            run for run in approximate_report.runs
            if run.name == "runtime-static-qdq-compute-fusion"
        )
        self.assertEqual(approximate_run.diagnostics[0].code, "VXQF005")

    def test_reports_compound_f32_input_without_inventing_an_affine(self):
        document, tensors = _unary_package(
            "GELU", params={"approximate": "none"},
        )
        document["inputs"]["residual"] = {
            "shape": [1, 4], "dtype": "float32",
        }
        document["nodes"].insert(1, _node(
            "compound", "Add",
            {"a": "input_float", "b": "residual"},
            "compound_float", (2, 4), "float32",
        ))
        document["nodes"][2]["inputs"]["input"] = "compound_float"

        unchanged, _, report = optimize_runtime_package(
            document,
            tensors,
            allow_static_qdq_compute_numerical_migration=True,
        )
        self.assertEqual(
            [node["opType"] for node in unchanged["nodes"]],
            ["DequantizeLinear", "Add", "GELU", "QuantizeLinear"],
        )
        fusion_run = next(
            run for run in report.runs
            if run.name == "runtime-static-qdq-compute-fusion"
        )
        diagnostic = next(
            item for item in fusion_run.diagnostics
            if item.source_node == "compute"
        )
        self.assertEqual(diagnostic.code, "VXQF002")
        self.assertIn("compound F32 value", diagnostic.message)
        self.assertIn(
            "qualified composite kernel", diagnostic.constraint,
        )

    def test_is_idempotent_and_does_not_mutate_caller_documents(self):
        document, tensors = _add_package()
        source_document = copy.deepcopy(document)
        source_tensors = {name: value.copy() for name, value in tensors.items()}
        optimized, optimized_tensors, _ = optimize_runtime_package(
            document,
            tensors,
            allow_static_qdq_compute_numerical_migration=True,
        )
        repeated, repeated_tensors, report = optimize_runtime_package(
            optimized,
            optimized_tensors,
            allow_static_qdq_compute_numerical_migration=True,
        )
        self.assertEqual(document, source_document)
        for name, value in source_tensors.items():
            np.testing.assert_array_equal(tensors[name], value)
        self.assertEqual(repeated, optimized)
        self.assertEqual(set(repeated_tensors), set(optimized_tensors))
        for name in optimized_tensors:
            np.testing.assert_array_equal(
                repeated_tensors[name], optimized_tensors[name],
            )
        fusion_run = next(
            run for run in report.runs
            if run.name == "runtime-static-qdq-compute-fusion"
        )
        self.assertEqual(fusion_run.changes, 0)


if __name__ == "__main__":
    unittest.main()
