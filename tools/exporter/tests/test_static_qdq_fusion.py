from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.exporter.capabilities import validate_graph
from tools.exporter.optimizer.static_qdq_fusion import (
    RuntimeStaticQDQComputeFusionPass,
)
from tools.exporter.optimizer.typed_groupnorm_silu_island import (
    RuntimeStaticQDQGroupNormSiLUFusionPass,
)
from tools.exporter.optimizer.typed_pipeline import optimize_runtime_package
from tools.exporter.reference_executor import execute_reference
from tools.exporter.runtime_ir import (
    import_runtime_package,
    prove_dynamic_quantized_runtime_domain,
)


def _node(identifier, op_type, inputs, output, shape, dtype, params=None):
    return {
        "id": identifier,
        "opType": op_type,
        "inputs": dict(inputs),
        "outputs": {"out": {
            "tensor": output, "shape": list(shape), "dtype": dtype,
        }},
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
        "dimensions": {},
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
    nodes = [
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
    ]
    right_float = "right_float"
    if tuple(right_shape) != shape:
        right_float = "right_expanded"
        nodes.append(_node(
            "right_expand", "Expand", {"input": "right_float"},
            right_float, shape, "float32", {"shape": list(shape)},
        ))
    nodes.extend((
        _node(
            "add", "Add", {"a": "left_float", "b": right_float},
            "sum_float", shape, "float32", params,
        ),
        _node(
            "q", "QuantizeLinear",
            {"input": "sum_float", "scale": "output_scale",
             "zero_point": "output_zero"},
            "sum_byte", shape, "uint8",
        ),
    ))
    document = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {
            "left_byte": {"shape": list(shape), "dtype": "int8"},
            "right_byte": {"shape": list(right_shape), "dtype": "uint8"},
        },
        "nodes": nodes,
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


def _batch_matmul_package(
    *,
    left_shape=(1, 2, 3, 4),
    right_shape=(1, 2, 4, 5),
    output_shape=(1, 2, 3, 5),
    dimensions=None,
    left_scale=0.125,
    right_scale=0.25,
    output_scale=0.0625,
    public_right=False,
):
    tensors = {
        "left_scale": np.asarray([left_scale], dtype=np.float32),
        "left_zero": np.asarray([127], dtype=np.uint8),
        "right_scale": np.asarray([right_scale], dtype=np.float32),
        "right_zero": np.asarray([129], dtype=np.uint8),
        "output_scale": np.asarray([output_scale], dtype=np.float32),
        "output_zero": np.asarray([131], dtype=np.uint8),
    }
    nodes = [
        _node(
            "left_dq", "DequantizeLinear",
            {"input": "left_byte", "scale": "left_scale",
             "zero_point": "left_zero"},
            "left_float", left_shape, "float32",
        ),
        _node(
            "right_dq", "DequantizeLinear",
            {"input": "right_byte", "scale": "right_scale",
             "zero_point": "right_zero"},
            "right_float", right_shape, "float32",
        ),
        _node(
            "product", "BatchMatMul",
            {"a": "left_float", "b": "right_float"},
            "product_float", output_shape, "float32",
        ),
        _node(
            "product_q", "QuantizeLinear",
            {"input": "product_float", "scale": "output_scale",
             "zero_point": "output_zero"},
            "product_byte", output_shape, "uint8",
        ),
    ]
    document = {
        "format": "volvox-graph/v1",
        "dimensions": dict(dimensions or {}),
        "inputs": {
            "left_byte": {"shape": list(left_shape), "dtype": "uint8"},
            "right_byte": {"shape": list(right_shape), "dtype": "uint8"},
        },
        "nodes": nodes,
        "outputs": ["product_byte"] + (["right_float"] if public_right else []),
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "left_byte": _affine("left_scale", "left_zero"),
                "right_byte": _affine("right_scale", "right_zero"),
                "product_byte": _affine("output_scale", "output_zero"),
            },
        },
    }
    return document, tensors


def _groupnorm_silu_package(
    *,
    transpose=True,
    residual=False,
    public_normalized=False,
    symbolic_batch=False,
):
    input_shape = (("B", 2, 3, 4) if symbolic_batch else (1, 2, 3, 4))
    output_shape = (
        (("B", 4, 2, 3) if symbolic_batch else (1, 4, 2, 3))
        if transpose else input_shape
    )
    tensors = {
        "input_scale": np.asarray([0.125], dtype=np.float32),
        "input_zero": np.asarray([-3], dtype=np.int8),
        "output_scale": np.asarray([0.0625], dtype=np.float32),
        "output_zero": np.asarray([7], dtype=np.uint8),
        "weight": np.asarray([0.75, 1.0, 1.25, 0.875], dtype=np.float32),
        "bias": np.asarray([-0.25, 0.125, 0.25, -0.125], dtype=np.float32),
    }
    nodes = [
        _node(
            "dq", "DequantizeLinear",
            {"input": "input_byte", "scale": "input_scale",
             "zero_point": "input_zero"},
            "input_float", input_shape, "float32",
        ),
        _node(
            "norm", "GroupNorm",
            {"input": "input_float", "weight": "weight", "bias": "bias"},
            "normalized", input_shape, "float32",
            {"num_groups": 2, "eps": 1e-5},
        ),
    ]
    current = "normalized"
    if transpose:
        nodes.append(_node(
            "layout", "Transpose", {"input": current}, "laid_out",
            output_shape, "float32", {"perm": [0, 3, 1, 2]},
        ))
        current = "laid_out"
    if residual:
        nodes.append(_node(
            "residual", "Add", {"a": current, "b": current},
            "residual_output", output_shape, "float32",
        ))
        current = "residual_output"
    nodes.extend((
        _node("activation", "SiLU", {"input": current}, "activated",
              output_shape, "float32"),
        _node(
            "q", "QuantizeLinear",
            {"input": "activated", "scale": "output_scale",
             "zero_point": "output_zero"},
            "output_byte", output_shape, "uint8",
        ),
    ))
    outputs = ["output_byte"]
    if public_normalized:
        outputs.append("normalized")
    return {
        "format": "volvox-graph/v1",
        "dimensions": ({"B": {"min": 1, "max": 4}}
                       if symbolic_batch else {}),
        "inputs": {
            "input_byte": {"shape": list(input_shape), "dtype": "int8"},
        },
        "nodes": nodes,
        "outputs": outputs,
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "input_byte": _affine("input_scale", "input_zero"),
                "output_byte": _affine("output_scale", "output_zero"),
            },
        },
    }, tensors


class StaticQDQComputeFusionTests(unittest.TestCase):
    def test_requires_explicit_numerical_migration_opt_in(self):
        with self.assertRaisesRegex(ValueError, "explicit numerical-migration"):
            RuntimeStaticQDQComputeFusionPass(allow_numerical_migration=False)

        document, tensors = _unary_package("GELU", params={"approximate": "none"})
        optimized, _, report = optimize_runtime_package(
            document, tensors, shape_profile={},
        )
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
             {"num_groups": 2, "eps": 1e-5},
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
                    shape_profile={},
                )

                self.assertEqual(
                    [node["opType"] for node in optimized["nodes"]], [expected],
                )
                fused = optimized["nodes"][0]
                self.assertEqual(fused["inputs"]["input"], "input_byte")
                self.assertEqual(
                    fused["outputs"]["out"]["tensor"], "output_byte",
                )
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
            shape_profile={},
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

    def test_fuses_no_broadcast_batch_matmul_with_exact_affines(self):
        document, tensors = _batch_matmul_package()
        left = np.asarray(
            [
                [
                    [[127, 128, 126, 130], [131, 125, 129, 124],
                     [126, 127, 128, 129]],
                    [[130, 129, 128, 127], [126, 125, 124, 123],
                     [127, 129, 131, 133]],
                ],
            ],
            dtype=np.uint8,
        )
        right = np.asarray(
            [
                [
                    [[129, 130, 128, 131, 127], [128, 129, 130, 127, 131],
                     [131, 128, 129, 130, 126], [127, 131, 128, 129, 130]],
                    [[130, 129, 128, 127, 126], [129, 131, 127, 130, 128],
                     [128, 127, 131, 129, 130], [131, 128, 129, 126, 127]],
                ],
            ],
            dtype=np.uint8,
        )
        inputs = {"left_byte": left, "right_byte": right}
        expected = execute_reference(
            import_runtime_package(copy.deepcopy(document), tensors),
            tensors,
            inputs,
        ).outputs["product_byte"]

        broad_only, _, broad_report = optimize_runtime_package(
            document,
            tensors,
            allow_static_qdq_compute_numerical_migration=True,
            shape_profile={},
        )
        self.assertIn(
            "BatchMatMul", [node["opType"] for node in broad_only["nodes"]],
        )
        self.assertNotIn(
            "runtime-static-qdq-qbatch-matmul-fusion",
            [run.name for run in broad_report.runs],
        )

        optimized, optimized_tensors, report = optimize_runtime_package(
            document,
            tensors,
            allow_static_qdq_qbatch_matmul_numerical_migration=True,
            shape_profile={},
        )

        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["QBatchMatMul"],
        )
        self.assertEqual(optimized["nodes"][0]["inputs"], {
            "a": "left_byte", "b": "right_byte",
        })
        self.assertEqual(optimized["nodes"][0]["params"], {})
        self.assertEqual(
            set(optimized["quantization"]["tensors"]),
            {"left_byte", "right_byte", "product_byte"},
        )
        actual = execute_reference(
            import_runtime_package(optimized, optimized_tensors),
            optimized_tensors,
            inputs,
        ).outputs["product_byte"]
        np.testing.assert_array_equal(actual, expected)
        run = next(
            item for item in report.runs
            if item.name == "runtime-static-qdq-qbatch-matmul-fusion"
        )
        self.assertEqual(run.changes, 1)
        self.assertIn("BatchMatMul=1", run.notes[0])

    def test_fuses_batch_matmul_while_preserving_public_f32_dq_boundary(self):
        document, tensors = _batch_matmul_package(public_right=True)
        left = np.asarray(
            [[[[127, 128, 126, 130], [131, 125, 129, 124],
               [126, 127, 128, 129]],
              [[130, 129, 128, 127], [126, 125, 124, 123],
               [127, 129, 131, 133]]]],
            dtype=np.uint8,
        )
        right = np.asarray(
            [[[[129, 130, 128, 131, 127], [128, 129, 130, 127, 131],
               [131, 128, 129, 130, 126], [127, 131, 128, 129, 130]],
              [[130, 129, 128, 127, 126], [129, 131, 127, 130, 128],
               [128, 127, 131, 129, 130], [131, 128, 129, 126, 127]]]],
            dtype=np.uint8,
        )
        inputs = {"left_byte": left, "right_byte": right}
        expected = execute_reference(
            import_runtime_package(copy.deepcopy(document), tensors),
            tensors,
            inputs,
        ).outputs

        optimized, optimized_tensors, report = optimize_runtime_package(
            document,
            tensors,
            allow_static_qdq_qbatch_matmul_numerical_migration=True,
            shape_profile={},
        )

        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["DequantizeLinear", "QBatchMatMul"],
        )
        retained_dq, qbatch_matmul = optimized["nodes"]
        self.assertEqual(retained_dq["id"], "right_dq")
        self.assertEqual(retained_dq["outputs"]["out"]["tensor"], "right_float")
        self.assertEqual(qbatch_matmul["inputs"], {
            "a": "left_byte", "b": "right_byte",
        })
        self.assertEqual(optimized["outputs"], ["product_byte", "right_float"])
        actual = execute_reference(
            import_runtime_package(optimized, optimized_tensors),
            optimized_tensors,
            inputs,
        ).outputs
        np.testing.assert_array_equal(
            actual["product_byte"], expected["product_byte"],
        )
        np.testing.assert_array_equal(
            actual["right_float"], expected["right_float"],
        )
        run = next(
            item for item in report.runs
            if item.name == "runtime-static-qdq-qbatch-matmul-fusion"
        )
        self.assertEqual(run.changes, 1)

    def test_refuses_shared_or_noncanonical_public_f32_dq_boundary(self):
        cases = []

        shared, shared_tensors = _batch_matmul_package(public_right=True)
        shared["nodes"].insert(2, _node(
            "right_copy", "Identity", {"input": "right_float"},
            "right_copy_float", (1, 2, 4, 5), "float32",
        ))
        shared["outputs"].append("right_copy_float")
        cases.append(("shared", shared, shared_tensors))

        noncanonical, noncanonical_tensors = _batch_matmul_package(
            public_right=True,
        )
        noncanonical["nodes"][1]["params"] = {"axis": 0}
        cases.append(("noncanonical", noncanonical, noncanonical_tensors))

        for label, document, tensors in cases:
            with self.subTest(label=label):
                unchanged, _, report = optimize_runtime_package(
                    document,
                    tensors,
                    allow_static_qdq_qbatch_matmul_numerical_migration=True,
                    shape_profile={},
                )
                self.assertIn(
                    "BatchMatMul",
                    [node["opType"] for node in unchanged["nodes"]],
                )
                self.assertNotIn(
                    "QBatchMatMul",
                    [node["opType"] for node in unchanged["nodes"]],
                )
                run = next(
                    item for item in report.runs
                    if item.name == "runtime-static-qdq-qbatch-matmul-fusion"
                )
                self.assertEqual(dict(run.metrics), {
                    "static_qdq_candidates_considered": 1,
                    "static_qdq_candidates_fused": 0,
                    "static_qdq_candidates_refused": 1,
                })

    def test_requires_independent_explicit_opt_in_and_is_off_by_default(self):
        with self.assertRaisesRegex(ValueError, "explicit numerical-migration"):
            RuntimeStaticQDQGroupNormSiLUFusionPass(
                allow_numerical_migration=False,
                tensor_data={},
            )

        document, tensors = _groupnorm_silu_package()
        unchanged, _, report = optimize_runtime_package(
            document, tensors, shape_profile={},
        )
        self.assertEqual(
            [node["opType"] for node in unchanged["nodes"]],
            ["DequantizeLinear", "GroupNorm", "Transpose", "SiLU",
             "QuantizeLinear"],
        )
        self.assertNotIn(
            "runtime-static-qdq-groupnorm-silu-fusion",
            [run.name for run in report.runs],
        )

    def test_fuses_direct_and_transposed_closed_islands_without_payload_changes(self):
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                document, tensors = _groupnorm_silu_package(
                    transpose=transpose,
                )
                original_payloads = {
                    name: value.copy() for name, value in tensors.items()
                }
                optimized, optimized_tensors, report = optimize_runtime_package(
                    document,
                    tensors,
                    allow_static_qdq_groupnorm_silu_numerical_migration=True,
                    shape_profile={},
                )

                expected_ops = ["QGroupNorm", "QSiLU"]
                if transpose:
                    expected_ops.insert(1, "Transpose")
                self.assertEqual(
                    [node["opType"] for node in optimized["nodes"]],
                    expected_ops,
                )
                norm = optimized["nodes"][0]
                activation = optimized["nodes"][-1]
                self.assertEqual(norm["inputs"], {
                    "input": "input_byte", "weight": "weight", "bias": "bias",
                })
                self.assertEqual(norm["params"], {
                    "num_groups": 2, "eps": 1e-5, "data_layout": "NHWC",
                })
                self.assertEqual(set(activation["inputs"]), {"input"})
                self.assertEqual(
                    activation["outputs"]["out"]["tensor"], "output_byte",
                )
                output_affine = optimized["quantization"]["tensors"][
                    "output_byte"
                ]
                self.assertEqual(
                    optimized["quantization"]["tensors"]["normalized"],
                    output_affine,
                )
                if transpose:
                    self.assertEqual(
                        optimized["quantization"]["tensors"]["laid_out"],
                        output_affine,
                    )
                    self.assertEqual(
                        optimized["nodes"][1]["outputs"]["out"]["dtype"],
                        "uint8",
                    )
                for name, original in original_payloads.items():
                    np.testing.assert_array_equal(tensors[name], original)
                    np.testing.assert_array_equal(
                        optimized_tensors[name], original,
                    )
                import_runtime_package(optimized, optimized_tensors)
                run = next(
                    item for item in report.runs
                    if item.name
                    == "runtime-static-qdq-groupnorm-silu-fusion"
                )
                self.assertEqual(run.changes, 1)
                self.assertEqual(dict(run.metrics), {
                    "groupnorm_silu_candidates_considered": 1,
                    "groupnorm_silu_candidates_fused": 1,
                    "groupnorm_silu_candidates_refused": 0,
                    "groupnorm_silu_layout_nodes_retyped": int(transpose),
                })

    def test_symbolic_batch_preserves_dynamic_shapes_and_runs_reference(self):
        document, tensors = _groupnorm_silu_package(symbolic_batch=True)
        optimized, optimized_tensors, _ = optimize_runtime_package(
            document,
            tensors,
            allow_static_qdq_groupnorm_silu_numerical_migration=True,
        )
        self.assertEqual(
            optimized["nodes"][0]["outputs"]["out"]["shape"],
            ["B", 2, 3, 4],
        )
        graph = import_runtime_package(
            optimized,
            optimized_tensors,
            bounded_domain_proof=prove_dynamic_quantized_runtime_domain(
                optimized, optimized_tensors,
            ),
        )
        result = execute_reference(
            graph,
            optimized_tensors,
            {"input_byte": np.arange(48, dtype=np.int8).reshape(2, 2, 3, 4)},
        )
        self.assertEqual(result.outputs["output_byte"].shape, (2, 4, 2, 3))

    def test_rewritten_island_is_qualified_for_portable_and_native_gpu_profiles(self):
        document, tensors = _groupnorm_silu_package(symbolic_batch=True)
        optimized, optimized_tensors, _ = optimize_runtime_package(
            document,
            tensors,
            allow_static_qdq_groupnorm_silu_numerical_migration=True,
        )
        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["QGroupNorm", "Transpose", "QSiLU"],
        )
        validation = validate_graph(
            optimized,
            (
                "portable", "backend:vulkan", "backend:opengl",
                "backend:cuda",
            ),
            weights=optimized_tensors,
        )
        self.assertTrue(
            validation.supported,
            [diagnostic.to_dict() for diagnostic in validation.diagnostics],
        )
        import_runtime_package(
            optimized,
            optimized_tensors,
            bounded_domain_proof=prove_dynamic_quantized_runtime_domain(
                optimized, optimized_tensors,
            ),
        )

    def test_refuses_residual_shared_public_and_nonfinite_regions(self):
        cases = []
        residual, residual_tensors = _groupnorm_silu_package(residual=True)
        cases.append(("residual", residual, residual_tensors))
        public, public_tensors = _groupnorm_silu_package(
            public_normalized=True,
        )
        cases.append(("public", public, public_tensors))
        nonfinite, nonfinite_tensors = _groupnorm_silu_package()
        nonfinite_tensors["weight"] = nonfinite_tensors["weight"].copy()
        nonfinite_tensors["weight"][0] = np.nan
        cases.append(("nonfinite", nonfinite, nonfinite_tensors))

        for label, document, tensors in cases:
            with self.subTest(label=label):
                unchanged, _, report = optimize_runtime_package(
                    document,
                    tensors,
                    allow_static_qdq_groupnorm_silu_numerical_migration=True,
                    shape_profile={},
                )
                self.assertIn(
                    "GroupNorm", [node["opType"] for node in unchanged["nodes"]],
                )
                self.assertNotIn(
                    "QGroupNorm", [node["opType"] for node in unchanged["nodes"]],
                )
                run = next(
                    item for item in report.runs
                    if item.name
                    == "runtime-static-qdq-groupnorm-silu-fusion"
                )
                self.assertEqual(dict(run.metrics), {
                    "groupnorm_silu_candidates_considered": 1,
                    "groupnorm_silu_candidates_fused": 0,
                    "groupnorm_silu_candidates_refused": 1,
                    "groupnorm_silu_layout_nodes_retyped": 0,
                })

    def test_fuses_symbolic_batch_matmul_only_with_identical_batch_prefixes(self):
        document, tensors = _batch_matmul_package(
            left_shape=(1, 8, "M", "M"),
            right_shape=(1, 8, "M", 40),
            output_shape=(1, 8, "M", 40),
            dimensions={"M": {"min": 1, "max": 192}},
        )
        optimized, _, _ = optimize_runtime_package(
            document,
            tensors,
            allow_static_qdq_qbatch_matmul_numerical_migration=True,
        )
        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["QBatchMatMul"],
        )
        self.assertEqual(
            optimized["nodes"][0]["outputs"]["out"]["shape"],
            [1, 8, "M", 40],
        )

        broadcast, broadcast_tensors = _batch_matmul_package(
            left_shape=(1, 8, 3, 4),
            right_shape=(1, 1, 4, 5),
            output_shape=(1, 8, 3, 5),
        )
        unchanged, _, report = optimize_runtime_package(
            broadcast,
            broadcast_tensors,
            allow_static_qdq_qbatch_matmul_numerical_migration=True,
            shape_profile={},
        )
        self.assertEqual(
            [node["opType"] for node in unchanged["nodes"]],
            ["DequantizeLinear", "DequantizeLinear", "BatchMatMul",
             "QuantizeLinear"],
        )
        run = next(
            item for item in report.runs
            if item.name == "runtime-static-qdq-qbatch-matmul-fusion"
        )
        self.assertEqual(dict(run.metrics), {
            "static_qdq_candidates_considered": 1,
            "static_qdq_candidates_fused": 0,
            "static_qdq_candidates_refused": 1,
        })

    def test_refuses_unsafe_batch_matmul_integer_domains(self):
        cases = (
            {
                "left_scale": 1e-30,
                "right_scale": 1e-30,
                "output_scale": 1.0,
            },
            {
                "left_shape": (1, 1, 140_000),
                "right_shape": (1, 140_000, 1),
                "output_shape": (1, 1, 1),
            },
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                document, tensors = _batch_matmul_package(**overrides)
                unchanged, _, _ = optimize_runtime_package(
                    document,
                    tensors,
                    allow_static_qdq_qbatch_matmul_numerical_migration=True,
                    shape_profile={},
                )
                self.assertIn(
                    "BatchMatMul",
                    [node["opType"] for node in unchanged["nodes"]],
                )
                self.assertNotIn(
                    "QBatchMatMul",
                    [node["opType"] for node in unchanged["nodes"]],
                )

    def test_lowers_static_broadcast_to_descriptor_preserving_expand_and_qadd(self):
        broadcast, broadcast_tensors = _add_package(right_shape=(1, 4))
        inputs = {
            "left_byte": np.asarray(
                [[-3, -2, 1, 5], [7, -8, 12, 0]], dtype=np.int8,
            ),
            "right_byte": np.asarray([[129, 130, 127, 140]], dtype=np.uint8),
        }
        expected = execute_reference(
            import_runtime_package(copy.deepcopy(broadcast), broadcast_tensors),
            broadcast_tensors,
            inputs,
        ).outputs["sum_byte"]
        original_payloads = {
            name: value.copy() for name, value in broadcast_tensors.items()
        }
        optimized, optimized_tensors, report = optimize_runtime_package(
            broadcast,
            broadcast_tensors,
            allow_static_qdq_compute_numerical_migration=True,
            shape_profile={},
        )
        self.assertEqual(
            [node["opType"] for node in optimized["nodes"]],
            ["Expand", "QAdd"],
        )
        expand, qadd = optimized["nodes"]
        self.assertEqual(expand["inputs"], {"input": "right_byte"})
        expanded_name = expand["outputs"]["out"]["tensor"]
        self.assertEqual(expand["outputs"]["out"]["shape"], [2, 4])
        self.assertEqual(qadd["inputs"], {
            "a": "left_byte", "b": expanded_name,
        })
        self.assertEqual(
            optimized["quantization"]["tensors"][expanded_name],
            optimized["quantization"]["tensors"]["right_byte"],
        )
        for name, value in original_payloads.items():
            np.testing.assert_array_equal(optimized_tensors[name], value)
        actual = execute_reference(
            import_runtime_package(optimized, optimized_tensors),
            optimized_tensors,
            inputs,
        ).outputs["sum_byte"]
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
            shape_profile={},
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
            shape_profile={},
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
            "shape": [2, 4], "dtype": "float32",
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
            shape_profile={},
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
            shape_profile={},
        )
        repeated, repeated_tensors, report = optimize_runtime_package(
            optimized,
            optimized_tensors,
            allow_static_qdq_compute_numerical_migration=True,
            shape_profile={},
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
