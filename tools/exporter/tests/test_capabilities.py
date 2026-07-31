from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np

from tools.exporter.capabilities import (
    classify_package,
    expand_targets,
    normalize_targets,
    validate_graph,
    validate_runtime_descriptors,
)
from tools.exporter.quantization_storage import externalize_quantization


class ArrayFixture(SimpleNamespace):
    def __init__(self, values, *, shape, dtype):
        super().__init__(shape=shape, dtype=dtype)
        self.values = values

    def tolist(self):
        return self.values

    def reshape(self, *_shape):
        flat = []

        def visit(value):
            if isinstance(value, list):
                for item in value:
                    visit(item)
            else:
                flat.append(value)

        visit(self.values)
        return ArrayFixture(flat, shape=(len(flat),), dtype=self.dtype)

    def __array__(self, dtype=None, copy=None):
        value = np.asarray(self.values, dtype=dtype or self.dtype).reshape(self.shape)
        return value.copy() if copy else value


def canonical_quantized_graph(graph, descriptors, weights=None):
    authored = copy.deepcopy(graph)
    tensors = dict(weights or {})
    externalize_quantization(authored, tensors, descriptors)
    return authored, tensors


def qsdpa_graph(*, rank3=True, mask_shape=None):
    q_shape = [2, 3, 16] if rank3 else [3, 16]
    kv_shape = [2, 5, 16] if rank3 else [5, 16]
    inputs = {
        "q": {"shape": q_shape, "dtype": "int8"},
        "k": {"shape": kv_shape, "dtype": "uint8"},
        "v": {"shape": kv_shape, "dtype": "int8"},
    }
    node_inputs = {"q": "q", "k": "k", "v": "v"}
    if mask_shape is not None:
        inputs["mask"] = {"shape": list(mask_shape), "dtype": "int32"}
        node_inputs["mask"] = "mask"
    graph = {
        "format": "volvox-graph/v1",
        "inputs": inputs,
        "outputs": ["y"],
        "nodes": [{
            "id": "qsdpa",
            "opType": "QSDPA",
            "inputs": node_inputs,
            "outputs": {"out": "y"},
            "outputs_shape": {"out": q_shape},
            "outputs_dtype": {"out": "uint8"},
            "params": {"heads": 4, "causal": False, "scale": None},
        }],
    }
    return canonical_quantized_graph(graph, {
        "q": {"scheme": "per_tensor", "scale": 0.125, "zero_point": -3},
        "k": {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
        "v": {"scheme": "per_tensor", "scale": 0.5, "zero_point": 1},
        "y": {"scheme": "per_tensor", "scale": 0.0625, "zero_point": 127},
    })


def qnorm_graph(op_type):
    if op_type == "QGroupNorm":
        shape = [2, 3, 4, 8]
        params = {"num_groups": 4, "eps": None, "data_layout": "NHWC"}
    elif op_type == "QLayerNorm":
        shape = [2, 3, 8]
        params = {"eps": None, "d_model": 8}
    else:
        raise ValueError(op_type)
    graph = {
        "format": "volvox-graph/v1",
        "inputs": {"x": {"shape": shape, "dtype": "uint8"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "qnorm",
            "opType": op_type,
            "inputs": {"input": "x", "weight": "gamma", "bias": "beta"},
            "outputs": {"out": "y"},
            "outputs_shape": {"out": shape},
            "outputs_dtype": {"out": "int8"},
            "params": params,
        }],
    }
    weights = {
        "gamma": np.linspace(0.5, 1.5, 8, dtype=np.float32),
        "beta": np.linspace(-0.25, 0.25, 8, dtype=np.float32),
    }
    return canonical_quantized_graph(graph, {
        "x": {"scheme": "per_tensor", "scale": 0.125, "zero_point": 128},
        "y": {"scheme": "per_tensor", "scale": 0.0625, "zero_point": -3},
    }, weights)


def graph_with_node(
    op_type: str,
    *,
    inputs: dict[str, str] | None = None,
    params: dict[str, object] | None = None,
    input_dtype: str = "float32",
    output_dtype: str = "float32",
) -> dict[str, object]:
    node: dict[str, object] = {
        "id": "node0",
        "opType": op_type,
        "inputs": inputs or {"input": "x"},
        "outputs": {"out": "y"},
        "outputs_shape": {"out": [1, 4]},
        "outputs_dtype": {"out": output_dtype},
        "params": params or {},
    }
    return {
        "format": "volvox-graph/v1",
        "inputs": {"x": {"shape": [1, 4], "dtype": input_dtype}},
        "outputs": ["y"],
        "nodes": [node],
    }


class TargetExpansionTests(unittest.TestCase):
    def test_default_and_browser_profiles_expand_deterministically(self):
        self.assertEqual(normalize_targets(None), ("portable",))
        self.assertEqual(
            expand_targets(None),
            ("cpu-js", "wasm", "webgpu", "native-cpu"),
        )
        self.assertEqual(
            expand_targets(["browser", "wasm", "browser"]),
            ("cpu-js", "wasm", "webgpu"),
        )

    def test_unknown_target_is_a_usage_failure_candidate(self):
        with self.assertRaisesRegex(ValueError, "Unknown target"):
            normalize_targets(["portable", "backend:unknown"])


class PackageClassificationTests(unittest.TestCase):
    def test_classifier_uses_live_descriptors_and_ignores_stale_declaration(self):
        graph = graph_with_node(
            "QGELU", input_dtype="int8", output_dtype="int8",
        )
        graph["source"] = {"package_class": "fp32"}
        self.assertEqual(classify_package(graph), "w8a8-v1")

        graph["nodes"].append({
            "id": "dequantize",
            "opType": "DequantizeLinear",
            "inputs": {"input": "y", "scale": "s", "zero_point": "z"},
            "outputs": {"out": "float_y"},
            "outputs_shape": {"out": [1, 4]},
            "outputs_dtype": {"out": "float32"},
        })
        self.assertEqual(classify_package(graph), "hybrid")

    def test_classifier_distinguishes_weight_only_and_float_packages(self):
        graph = graph_with_node(
            "Linear",
            inputs={"input": "x", "weight": "weight"},
            params={"weight_layout": "OUT_IN"},
        )
        self.assertEqual(classify_package(graph), "fp32")
        self.assertEqual(
            classify_package(
                graph,
                {"weight": np.asarray([[1, 2, 3, 4]], dtype=np.int8)},
            ),
            "w8a32",
        )


class CapabilityValidationTests(unittest.TestCase):
    def diagnostic_codes(self, graph, targets=None, *, weights=None):
        return [
            item.code
            for item in validate_graph(graph, targets, weights=weights).diagnostics
        ]

    def test_common_descriptor_is_supported_by_portable_profile(self):
        result = validate_graph(graph_with_node("Identity"), ["portable"])
        self.assertTrue(result.supported)
        self.assertEqual(result.targets, ("cpu-js", "wasm", "webgpu", "native-cpu"))

    def test_runtime_descriptor_gate_is_target_neutral_and_exact(self):
        valid = validate_runtime_descriptors(graph_with_node("Identity"))
        self.assertTrue(valid.supported)
        self.assertEqual(valid.targets, ())

        malformed = graph_with_node("Conv2D")
        result = validate_runtime_descriptors(malformed)
        self.assertFalse(result.supported)
        self.assertIn("VXDESC_PORTS", {item.code for item in result.diagnostics})

    def test_quantized_dense_aliases_share_one_descriptor_contract(self):
        for op_type in ("QLinear", "QMatMul", "QGemm"):
            with self.subTest(op_type=op_type):
                graph = {
                    "format": "volvox-graph/v1",
                    "inputs": {"x": {"shape": [2, 3, 4], "dtype": "int8"}},
                    "outputs": ["y"],
                    "nodes": [{
                        "opType": op_type,
                        "inputs": {"input": "x", "weight": "w", "bias": "b"},
                        "outputs": {"out": "y"},
                        "outputs_shape": {"out": [2, 3, 5]},
                        "outputs_dtype": {"out": "int8"},
                        "params": {},
                    }],
                }
                graph, weights = canonical_quantized_graph(
                    graph,
                    {
                        "x": {"scheme": "per_tensor", "scale": 0.125,
                              "zero_point": 0},
                        "w": {"scheme": "per_axis", "axis": 0,
                              "scales": [0.25] * 5, "zero_points": [0] * 5},
                        "y": {"scheme": "per_tensor", "scale": 0.5,
                              "zero_point": 0},
                    },
                    {
                        "w": np.ones((5, 4), dtype=np.int8),
                        "b": np.zeros((5,), dtype=np.int32),
                    },
                )
                self.assertTrue(
                    validate_runtime_descriptors(graph, weights=weights).supported
                )

    def test_typed_resize_is_byte_only_nearest_asymmetric_floor(self):
        graph = {
            "format": "volvox-graph/v1",
            "inputs": {"x": {"shape": [1, 2, 3, 4], "dtype": "uint8"}},
            "outputs": ["y"],
            "nodes": [{
                "opType": "Resize",
                "inputs": {"input": "x"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [1, 4, 6, 4]},
                "outputs_dtype": {"out": "uint8"},
                "params": {
                    "mode": "nearest",
                    "coordinate_transformation_mode": "asymmetric",
                    "nearest_mode": "floor",
                },
            }],
        }
        graph, weights = canonical_quantized_graph(graph, {
            "x": {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
            "y": {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
        })
        self.assertTrue(
            validate_runtime_descriptors(graph, weights=weights).supported
        )

        graph["nodes"][0]["params"]["mode"] = "linear"
        result = validate_runtime_descriptors(graph, weights=weights)
        self.assertIn("VXDESC_RESIZE", {item.code for item in result.diagnostics})

    def test_typed_expand_requires_exact_broadcast_and_preserved_affine(self):
        def authored(output_shape, output_scale=0.25):
            graph = {
                "format": "volvox-graph/v1",
                "inputs": {
                    "x": {"shape": [1, 1, 64], "dtype": "uint8"},
                },
                "nodes": [{
                    "id": "expand",
                    "opType": "Expand",
                    "inputs": {"input": "x"},
                    "outputs": {"out": "y"},
                    "outputs_shape": {"out": list(output_shape)},
                    "outputs_dtype": {"out": "uint8"},
                    "params": {},
                }],
                "outputs": ["y"],
            }
            return canonical_quantized_graph(graph, {
                "x": {"scheme": "per_tensor", "scale": 0.25,
                      "zero_point": 128},
                "y": {"scheme": "per_tensor", "scale": output_scale,
                      "zero_point": 128},
            })

        graph, weights = authored((1, 402, 64))
        self.assertTrue(
            validate_runtime_descriptors(graph, weights=weights).supported
        )

        mismatch, mismatch_weights = authored((1, 402, 64), output_scale=0.5)
        self.assertIn(
            "VXDESC_DTYPE",
            {item.code for item in validate_runtime_descriptors(
                mismatch, weights=mismatch_weights,
            ).diagnostics},
        )

        incompatible, incompatible_weights = authored((1, 402, 63))
        self.assertIn(
            "VXDESC_EXPAND",
            {item.code for item in validate_runtime_descriptors(
                incompatible, weights=incompatible_weights,
            ).diagnostics},
        )

        rank_nine, rank_nine_weights = authored((1,) * 8 + (64,))
        self.assertIn(
            "VXDESC_EXPAND",
            {item.code for item in validate_runtime_descriptors(
                rank_nine, weights=rank_nine_weights,
            ).diagnostics},
        )

    def test_portable_profile_admits_f32_div(self):
        graph = graph_with_node("Div", inputs={"a": "x", "b": "x"})
        browser = validate_graph(graph, ["browser"])
        portable = validate_graph(graph, ["portable"])

        self.assertTrue(browser.supported)
        self.assertTrue(portable.supported)

    def test_typed_router_and_batch_matmul_contracts_are_portable(self):
        batch_matmul = {
            "format": "volvox-graph/v1",
            "inputs": {
                "a": {"shape": [2, 3, 4], "dtype": "float32"},
                "b": {"shape": [1, 4, 5], "dtype": "float32"},
            },
            "nodes": [{
                "id": "bmm",
                "opType": "BatchMatMul",
                "inputs": {"a": "a", "b": "b"},
                "outputs": {"out": "product"},
                "outputs_shape": {"out": [2, 3, 5]},
                "outputs_dtype": {"out": "float32"},
                "params": {},
            }],
            "outputs": ["product"],
        }
        self.assertTrue(validate_graph(batch_matmul, ["portable"]).supported)
        batch_matmul["nodes"][0]["outputs_shape"]["out"] = [2, 3, 4]
        self.assertIn(
            "VXDESC_BATCH_MATMUL",
            self.diagnostic_codes(batch_matmul, ["portable"]),
        )

        where = {
            "format": "volvox-graph/v1",
            "inputs": {
                "condition": {"shape": [2, 3], "dtype": "int32"},
                "x": {"shape": [2, 3], "dtype": "int32"},
                "y": {"shape": [2, 3], "dtype": "int32"},
            },
            "nodes": [{
                "id": "where",
                "opType": "Where",
                "inputs": {"condition": "condition", "x": "x", "y": "y"},
                "outputs": {"out": "selected"},
                "outputs_shape": {"out": [2, 3]},
                "outputs_dtype": {"out": "int32"},
                "params": {},
            }],
            "outputs": ["selected"],
        }
        self.assertTrue(validate_graph(where, ["portable"]).supported)
        where["inputs"]["condition"]["dtype"] = "float32"
        self.assertIn("VXDESC_DTYPE", self.diagnostic_codes(where, ["portable"]))

    def test_split_and_i32_shape_ops_have_exact_contracts(self):
        split = {
            "format": "volvox-graph/v1",
            "inputs": {"x": {"shape": [1, 4], "dtype": "int32"}},
            "nodes": [{
                "id": "split",
                "opType": "Split",
                "inputs": {"input": "x"},
                "outputs": {"left": "left", "right": "right"},
                "outputs_shape": {"left": [1, 2], "right": [1, 2]},
                "outputs_dtype": {"left": "int32", "right": "int32"},
                "params": {"axis": 1},
            }],
            "outputs": ["left", "right"],
        }
        self.assertTrue(validate_graph(split, ["portable"]).supported)
        split["nodes"][0]["outputs_shape"]["right"] = [1, 1]
        self.assertIn("VXDESC_SPLIT", self.diagnostic_codes(split, ["portable"]))

        sliced = graph_with_node(
            "Slice",
            params={"starts": [1], "axes": [1], "steps": [2]},
            input_dtype="int32",
            output_dtype="int32",
        )
        sliced["inputs"]["x"]["shape"] = [1, 8]
        self.assertTrue(validate_graph(sliced, ["portable"]).supported)
        sliced["nodes"][0]["params"]["steps"] = [0]
        self.assertIn("VXDESC_SLICE", self.diagnostic_codes(sliced, ["portable"]))

    def test_transpose_requires_quantization_for_portable_byte_storage(self):
        def transpose_graph(dtype: str):
            return {
                "format": "volvox-graph/v1",
                "inputs": {"x": {"shape": [2, 3], "dtype": dtype}},
                "nodes": [{
                    "id": "transpose",
                    "opType": "Transpose",
                    "inputs": {"input": "x"},
                    "outputs": {"out": "y"},
                    "outputs_shape": {"out": [3, 2]},
                    "outputs_dtype": {"out": dtype},
                    "params": {"perm": [1, 0]},
                }],
                "outputs": ["y"],
            }

        for dtype in ("float32", "int32"):
            with self.subTest(dtype=dtype):
                self.assertTrue(
                    validate_graph(transpose_graph(dtype), ["portable"]).supported
                )

        for dtype in ("int8", "uint8"):
            with self.subTest(dtype=dtype):
                result = validate_graph(transpose_graph(dtype), ["portable"])
                self.assertFalse(result.supported)
                quantization_failures = [
                    diagnostic
                    for diagnostic in result.diagnostics
                    if diagnostic.code == "VXDESC_QUANT"
                ]
                self.assertEqual(len(quantization_failures), 1)
                self.assertIn(
                    "exact per-tensor byte-domain descriptor",
                    quantization_failures[0].message,
                )

    def test_webnn_rejects_cross_attention_without_fallback(self):
        result = validate_graph(
            graph_with_node("CrossSDPA", inputs={"q": "x", "k": "x", "v": "x"}),
            ["backend:webnn"],
        )
        self.assertFalse(result.supported)
        self.assertEqual(result.diagnostics[0].code, "VXCAP001")
        self.assertEqual(result.diagnostics[0].target, "backend:webnn")

    def test_unqualified_backend_routes_and_native_average_pool_fail_closed(self):
        average_pool = graph_with_node("AveragePool2D")
        portable = validate_graph(average_pool, ["portable"])
        self.assertIn(
            ("VXCAP001", "native-cpu"),
            [(item.code, item.target) for item in portable.diagnostics],
        )

        transpose = graph_with_node("Transpose")
        webnn = validate_graph(transpose, ["backend:webnn"])
        self.assertIn("VXCAP001", [item.code for item in webnn.diagnostics])

        for target in (
            "backend:vulkan", "backend:opengl", "backend:metal", "backend:cuda",
        ):
            with self.subTest(target=target):
                result = validate_graph(graph_with_node("Identity"), [target])
                self.assertFalse(result.supported)
                self.assertEqual(result.diagnostics[0].target, target)

    def test_webnn_w8a32_rejection_does_not_depend_on_target_order(self):
        graph = graph_with_node(
            "Linear",
            inputs={"input": "x", "weight": "w", "weight_scale": "s"},
            params={"weight_layout": "OUT_IN"},
        )
        weights = {
            "w": SimpleNamespace(shape=(4, 4), dtype="int8"),
            "s": SimpleNamespace(shape=(4,), dtype="float32"),
        }
        for targets in (
            ["backend:webnn", "cpu-js"],
            ["cpu-js", "backend:webnn"],
        ):
            with self.subTest(targets=targets):
                diagnostics = validate_graph(graph, targets, weights=weights).diagnostics
                failures = [item for item in diagnostics if item.code == "VXW8A32_TARGET"]
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0].target, "backend:webnn")

    def test_control_flow_has_a_stable_diagnostic(self):
        result = validate_graph(graph_with_node("If"), ["cpu-js"])
        codes = [item.code for item in result.diagnostics]
        self.assertIn("VXCTL001", codes)
        diagnostic = next(item for item in result.diagnostics if item.code == "VXCTL001")
        self.assertEqual(diagnostic.required_pass, "control_flow")

    def test_linear_layout_softmax_axis_and_argmax_ties_fail_closed(self):
        linear = graph_with_node("MatMul", inputs={"input": "x", "weight": "w"})
        linear_result = validate_graph(
            linear,
            ["cpu-js"],
            weights={"w": SimpleNamespace(shape=(4,), dtype="float32")},
        )
        linear_codes = [item.code for item in linear_result.diagnostics]
        self.assertIn("VXLINEAR001", linear_codes)
        self.assertIn("VXLINEAR002", linear_codes)

        softmax = graph_with_node("Softmax", params={"axis": 0})
        self.assertIn("VXAXIS001", self.diagnostic_codes(softmax, ["cpu-js"]))

        argmax = graph_with_node("ArgMax", params={"axis": -1, "select_last_index": 1}, output_dtype="int32")
        self.assertIn("VXARGMAX001", self.diagnostic_codes(argmax, ["cpu-js"]))

    def test_embedding_ids_and_quantized_index_outputs_are_typed(self):
        embedding = graph_with_node("Embedding")
        self.assertIn("VXEMBED001", self.diagnostic_codes(embedding, ["cpu-js"]))

        embedding["inputs"]["x"]["dtype"] = "int32"
        self.assertNotIn("VXEMBED001", self.diagnostic_codes(embedding, ["cpu-js"]))

        qargmax = graph_with_node("QArgMax", params={"axis": -1})
        self.assertIn("VXQUANT001", self.diagnostic_codes(qargmax, ["cpu-js"]))
        qargmax["nodes"][0]["outputs_dtype"] = {"out": "int32"}
        self.assertNotIn("VXQUANT001", self.diagnostic_codes(qargmax, ["cpu-js"]))

    def test_malformed_root_contract_is_reported_without_throwing(self):
        result = validate_graph({"format": "wrong", "inputs": [], "nodes": {}, "outputs": []})
        self.assertFalse(result.supported)
        self.assertEqual(
            {item.code for item in result.diagnostics},
            {"VXPKG001", "VXPKG002", "VXPKG003", "VXPKG004"},
        )

    def test_topology_and_public_outputs_fail_before_publication(self):
        aliased_op = graph_with_node("Identity")
        aliased_op["nodes"][0]["op"] = "Identity"
        self.assertIn(
            "VXPKG019",
            self.diagnostic_codes(aliased_op, ["cpu-js"]),
        )

        malformed_params = graph_with_node("Identity")
        malformed_params["nodes"][0]["params"] = []
        self.assertIn(
            "VXPKG020",
            self.diagnostic_codes(malformed_params, ["cpu-js"]),
        )
        malformed_params["nodes"][0]["params"] = None
        self.assertIn(
            "VXPKG020",
            self.diagnostic_codes(malformed_params, ["cpu-js"]),
        )

        missing_input = graph_with_node("Identity", inputs={"input": "missing"})
        self.assertIn(
            "VXPKG012",
            self.diagnostic_codes(missing_input, ["cpu-js"]),
        )

        collision = graph_with_node("Identity")
        collision["nodes"][0]["outputs"] = {"out": "x"}
        collision["outputs"] = ["x"]
        self.assertIn(
            "VXPKG014",
            self.diagnostic_codes(collision, ["cpu-js"]),
        )

        missing_output = graph_with_node("Identity")
        missing_output["outputs"] = ["not_produced"]
        self.assertIn(
            "VXPKG016",
            self.diagnostic_codes(missing_output, ["cpu-js"]),
        )

    def test_recursive_affine_params_fail_without_reserving_semantic_scale(self):
        extension = graph_with_node(
            "Identity",
            params={"scale": 0.5, "private": [{"alpha": 1.0}]},
        )
        self.assertTrue(validate_graph(extension, ["cpu-js"]).supported)

        forbidden = (
            "quantization", "zero_point",
            "input_scale", "input_zero_point",
            "output_scale", "output_zero_point",
            "weight_scale", "weight_zero_point",
            "scales", "zero_points",
            "scale_tensor", "zero_point_tensor",
        )
        for field in forbidden:
            with self.subTest(field=field):
                graph = copy.deepcopy(extension)
                graph["nodes"][0]["params"]["private"][0]["affine"] = {
                    field: 0.25,
                }
                diagnostics = validate_graph(graph, ["cpu-js"]).diagnostics
                retired = [
                    item for item in diagnostics if item.code == "VXPKG021"
                ]
                self.assertEqual(len(retired), 1)
                self.assertIn(
                    f"params.private[0].affine.{field}", retired[0].message,
                )

    def test_quantized_linear_requires_the_exact_typed_descriptor(self):
        graph = {
            "format": "volvox-graph/v1",
            "inputs": {
                "x": {
                    "shape": [1, 3], "dtype": "uint8",
                }
            },
            "nodes": [{
                "id": "linear", "opType": "QLinear",
                "inputs": {"input": "x", "weight": "w", "bias": "bias"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [1, 4]},
                "outputs_dtype": {"out": "uint8"},
                "params": {},
            }],
            "outputs": ["y"],
        }
        weights = {
            "w": ArrayFixture([[0, 0, 0]] * 4, shape=(4, 3), dtype="int8"),
            "bias": ArrayFixture([0] * 4, shape=(4,), dtype="int32"),
        }
        descriptors = {
            "x": {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
            "w": {
                "scheme": "per_axis", "axis": 0,
                "scales": [0.5] * 4, "zero_points": [0] * 4,
            },
            "y": {"scheme": "per_tensor", "scale": 0.5, "zero_point": 120},
        }
        graph, weights = canonical_quantized_graph(graph, descriptors, weights)
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

        for label, input_scale, weight_scale in (
            ("underflow", 1e-30, 1e-30),
            ("overflow", 1e30, 1e30),
        ):
            with self.subTest(multiplier=label):
                invalid_multiplier = copy.deepcopy(graph)
                invalid_weights = dict(weights)
                table = invalid_multiplier["quantization"]["tensors"]
                invalid_weights[table["x"]["scale_tensor"]] = np.asarray(
                    [input_scale], dtype=np.float32
                )
                invalid_weights[table["w"]["scale_tensor"]] = np.asarray(
                    [weight_scale] * 4, dtype=np.float32
                )
                self.assertIn(
                    "VXDESC_QLINEAR_MULTIPLIER",
                    {
                        item.code
                        for item in validate_graph(
                            invalid_multiplier, ["portable"], weights=invalid_weights
                        ).diagnostics
                    },
                )

        malformed = copy.deepcopy(graph)
        malformed["quantization"]["tensors"].pop("y")
        self.assertIn(
            "VXDESC_QLINEAR",
            self.diagnostic_codes(malformed, ["portable"], weights=weights),
        )

    def test_qgemm_uses_the_same_physical_weight_and_i32_bias_contract(self):
        graph = {
            "format": "volvox-graph/v1",
            "inputs": {
                "x": {
                    "shape": [2, 3],
                    "dtype": "uint8",
                }
            },
            "nodes": [{
                "id": "gemm",
                "opType": "QGemm",
                "inputs": {"input": "x", "weight": "w", "bias": "bias"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [2, 4]},
                "outputs_dtype": {"out": "uint8"},
                "params": {},
            }],
            "outputs": ["y"],
        }
        weights = {
            "w": ArrayFixture([[1, -2, 3]] * 4, shape=(4, 3), dtype="int8"),
            "bias": ArrayFixture([0] * 4, shape=(4,), dtype="int32"),
        }
        graph, weights = canonical_quantized_graph(graph, {
            "x": {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
            "w": {
                "scheme": "per_axis", "axis": 0,
                "scales": [0.5] * 4, "zero_points": [0] * 4,
            },
            "y": {"scheme": "per_tensor", "scale": 0.5, "zero_point": 120},
        }, weights)

        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

        malformed = copy.deepcopy(graph)
        malformed["quantization"]["tensors"]["w"]["axis"] = 1
        self.assertIn(
            "VXDESC_QGEMM",
            {
                item.code
                for item in validate_graph(
                    malformed, ["portable"], weights=weights
                ).diagnostics
            },
        )

    def test_qbatch_matmul_validates_broadcast_descriptors_and_i32_bound(self):
        graph = {
            "format": "volvox-graph/v1",
            "inputs": {
                "a": {
                    "shape": [2, 1, 3, 4],
                    "dtype": "uint8",
                },
                "b": {
                    "shape": [1, 5, 4, 6],
                    "dtype": "uint8",
                },
            },
            "nodes": [{
                "id": "qbatch",
                "opType": "QBatchMatMul",
                "inputs": {"a": "a", "b": "b"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [2, 5, 3, 6]},
                "outputs_dtype": {"out": "uint8"},
                "params": {},
            }],
            "outputs": ["y"],
        }
        graph, weights = canonical_quantized_graph(graph, {
            "a": {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
            "b": {"scheme": "per_tensor", "scale": 0.5, "zero_point": 127},
            "y": {"scheme": "per_tensor", "scale": 0.125, "zero_point": 120},
        })
        self.assertTrue(validate_graph(graph, ["portable"], weights=weights).supported)

        malformed = dict(graph)
        malformed["nodes"] = [dict(graph["nodes"][0])]
        malformed["nodes"][0]["outputs_shape"] = {"out": [2, 1, 3, 6]}
        self.assertIn(
            "VXDESC_QBATCH_MATMUL",
            self.diagnostic_codes(malformed, ["portable"], weights=weights),
        )

        missing_descriptor = copy.deepcopy(graph)
        missing_descriptor["quantization"]["tensors"].pop("b")
        self.assertIn(
            "VXDESC_QBATCH_MATMUL",
            self.diagnostic_codes(
                missing_descriptor, ["portable"], weights=weights
            ),
        )

        rank_nine = {
            **graph,
            "inputs": {
                "a": {
                    **graph["inputs"]["a"],
                    "shape": [1, 1, 1, 1, 1, 1, 1, 3, 4],
                },
                "b": {
                    **graph["inputs"]["b"],
                    "shape": [1, 1, 1, 1, 1, 1, 1, 4, 6],
                },
            },
            "nodes": [{
                **graph["nodes"][0],
                "outputs_shape": {
                    "out": [1, 1, 1, 1, 1, 1, 1, 3, 6],
                },
            }],
        }
        self.assertIn(
            "VXDESC_QBATCH_MATMUL",
            self.diagnostic_codes(rank_nine, ["portable"], weights=weights),
        )

        overflowing = {
            **graph,
            "inputs": {
                "a": {
                    **graph["inputs"]["a"],
                    "shape": [1, 1, 140_000],
                },
                "b": {
                    **graph["inputs"]["b"],
                    "shape": [1, 140_000, 1],
                },
            },
            "nodes": [{
                **graph["nodes"][0],
                "outputs_shape": {"out": [1, 1, 1]},
            }],
        }
        self.assertIn(
            "VXDESC_QBATCH_MATMUL_BOUND",
            self.diagnostic_codes(overflowing, ["portable"], weights=weights),
        )

        for label, left_scale, right_scale in (
            ("underflow", 1e-30, 1e-30),
            ("overflow", 1e30, 1e30),
        ):
            with self.subTest(multiplier=label):
                invalid_multiplier = copy.deepcopy(graph)
                invalid_weights = dict(weights)
                table = invalid_multiplier["quantization"]["tensors"]
                invalid_weights[table["a"]["scale_tensor"]] = np.asarray(
                    [left_scale], dtype=np.float32
                )
                invalid_weights[table["b"]["scale_tensor"]] = np.asarray(
                    [right_scale], dtype=np.float32
                )
                self.assertIn(
                    "VXDESC_QBATCH_MATMUL_MULTIPLIER",
                    self.diagnostic_codes(
                        invalid_multiplier, ["portable"], weights=invalid_weights
                    ),
                )

    def test_qconv_validates_exact_byte_layout_bounds_and_multipliers(self):
        graph = {
            "format": "volvox-graph/v1",
            "source": {"package_class": "w8a8-v1"},
            "inputs": {
                "x": {
                    "shape": [1, 3, 4, 2],
                    "dtype": "uint8",
                }
            },
            "nodes": [{
                "id": "qconv",
                "opType": "QConv2D",
                "inputs": {"input": "x", "weight": "w", "bias": "bias"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [1, 3, 4, 3]},
                "outputs_dtype": {"out": "uint8"},
                "params": {
                    "stride": [1, 1],
                    "dilation": [1, 1],
                    "groups": 1,
                    "pads": [1, 1, 1, 1],
                    "padding": [1, 1],
                    "data_layout": "NHWC",
                    "weight_layout": "OHWI",
                },
            }],
            "outputs": ["y"],
        }
        weight_values = [
            [
                [[channel - 2, channel + 1] for _x in range(3)]
                for _y in range(3)
            ]
            for channel in range(3)
        ]
        weights = {
            "w": ArrayFixture(
                weight_values, shape=(3, 3, 3, 2), dtype="int8"
            ),
            "bias": ArrayFixture([0, 0, 0], shape=(3,), dtype="int32"),
        }
        graph, weights = canonical_quantized_graph(graph, {
            "x": {"scheme": "per_tensor", "scale": 0.25, "zero_point": 128},
            "w": {
                "scheme": "per_axis", "axis": 0,
                "scales": [0.25, 0.375, 0.5],
                "zero_points": [0, 1, -2],
            },
            "y": {"scheme": "per_tensor", "scale": 0.125, "zero_point": 120},
        }, weights)
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

        malformed_axis = copy.deepcopy(graph)
        malformed_axis["quantization"]["tensors"]["w"]["axis"] = 1
        self.assertIn(
            "VXDESC_QCONV",
            {
                item.code
                for item in validate_graph(
                    malformed_axis, ["portable"], weights=weights
                ).diagnostics
            },
        )

        malformed_geometry = {
            **graph,
            "nodes": [{
                **graph["nodes"][0],
                "params": {
                    **graph["nodes"][0]["params"],
                    "stride": [1, 2],
                },
            }],
        }
        self.assertIn(
            "VXDESC_QCONV",
            {
                item.code
                for item in validate_graph(
                    malformed_geometry, ["portable"], weights=weights
                ).diagnostics
            },
        )

        invalid_multiplier = copy.deepcopy(graph)
        invalid_weights = dict(weights)
        table = invalid_multiplier["quantization"]["tensors"]
        invalid_weights[table["x"]["scale_tensor"]] = np.asarray(
            [1e-30], dtype=np.float32
        )
        invalid_weights[table["w"]["scale_tensor"]] = np.asarray(
            [1e-30] * 3, dtype=np.float32
        )
        self.assertIn(
            "VXDESC_QCONV_MULTIPLIER",
            {
                item.code
                for item in validate_graph(
                    invalid_multiplier, ["portable"], weights=invalid_weights
                ).diagnostics
            },
        )

        overflowing_weights = {
            **weights,
            "w": ArrayFixture(
                weight_values, shape=(3, 3, 3, 2), dtype="int8"
            ),
        }
        overflowing_weights["w"].centered_abs_sums = [
            20_000_000, 20_000_000, 20_000_000,
        ]
        self.assertIn(
            "VXDESC_QCONV_BOUND",
            {
                item.code
                for item in validate_graph(
                    graph, ["portable"], weights=overflowing_weights
                ).diagnostics
            },
        )

    def test_byte_transpose_preserves_exact_quantization_descriptor(self):
        descriptor = {
            "scheme": "per_tensor", "scale": 0.25, "zero_point": 128,
        }
        graph = {
            "format": "volvox-graph/v1",
            "inputs": {
                "x": {
                    "shape": [1, 2, 3, 4],
                    "dtype": "uint8",
                }
            },
            "nodes": [{
                "id": "transpose",
                "opType": "Transpose",
                "inputs": {"input": "x"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [1, 3, 4, 2]},
                "outputs_dtype": {"out": "uint8"},
                "params": {"perm": [0, 2, 3, 1]},
            }],
            "outputs": ["y"],
        }
        graph, weights = canonical_quantized_graph(
            graph, {"x": descriptor, "y": dict(descriptor)}
        )
        self.assertTrue(validate_graph(
            graph, ["portable"], weights=weights
        ).supported)

        table = graph["quantization"]["tensors"]
        weights[table["y"]["zero_point_tensor"]] = np.asarray(
            [127], dtype=np.uint8
        )
        self.assertIn(
            "VXDESC_QUANT",
            self.diagnostic_codes(graph, ["portable"], weights=weights),
        )

    def test_attention_and_conv_membership_do_not_bypass_descriptor_checks(self):
        attention = graph_with_node(
            "CrossSDPA", inputs={"q": "x", "k": "x", "v": "x"},
            params={"heads": 3, "causal": False, "scale": 1.0},
        )
        self.assertIn(
            "VXDESC_CROSS_SDPA",
            self.diagnostic_codes(attention, ["browser"]),
        )

        convolution = graph_with_node(
            "Conv2D", inputs={"input": "x", "weight": "missing"},
            params={"data_layout": "NHWC", "weight_layout": "OHWI"},
        )
        codes = self.diagnostic_codes(convolution, ["browser"])
        self.assertIn("VXPKG012", codes)
        self.assertIn("VXDESC_CONV", codes)

    def test_qsdpa_accepts_only_the_canonical_byte_attention_descriptor(self):
        for rank3, mask_shapes in (
            (False, (None, [5], [1, 5], [3, 5], [1, 3, 5])),
            (True, (None, [5], [2, 5], [3, 5], [2, 3, 5])),
        ):
            for mask_shape in mask_shapes:
                with self.subTest(rank=3 if rank3 else 2, mask=mask_shape):
                    graph, weights = qsdpa_graph(
                        rank3=rank3, mask_shape=mask_shape,
                    )
                    result = validate_graph(
                        graph, ["portable"], weights=weights,
                    )
                    self.assertTrue(result.supported, result.diagnostics)
                    self.assertNotIn(
                        "VXDESC_UNVALIDATED",
                        {diagnostic.code for diagnostic in result.diagnostics},
                    )

        graph, weights = qsdpa_graph()
        graph["nodes"][0]["params"].pop("scale")
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )
        graph["nodes"][0]["params"]["scale"] = 0.5
        self.assertTrue(
            validate_graph(graph, ["portable"], weights=weights).supported
        )

    def test_qsdpa_fails_closed_on_every_physical_descriptor_boundary(self):
        graph, weights = qsdpa_graph(mask_shape=[2, 3, 5])

        extra_port = copy.deepcopy(graph)
        extra_port["nodes"][0]["inputs"]["bias"] = "q"
        self.assertIn(
            "VXDESC_PORTS",
            self.diagnostic_codes(extra_port, ["wasm"], weights=weights),
        )

        missing_affine = copy.deepcopy(graph)
        missing_affine["quantization"]["tensors"].pop("v")
        self.assertIn(
            "VXDESC_QSDPA_QUANT",
            self.diagnostic_codes(missing_affine, ["wasm"], weights=weights),
        )

        malformed_geometry = copy.deepcopy(graph)
        malformed_geometry["inputs"]["k"]["shape"][-1] = 12
        self.assertIn(
            "VXDESC_QSDPA_GEOMETRY",
            self.diagnostic_codes(
                malformed_geometry, ["wasm"], weights=weights,
            ),
        )

        malformed_mask = copy.deepcopy(graph)
        malformed_mask["inputs"]["mask"]["shape"] = [2, 4, 5]
        self.assertIn(
            "VXDESC_QSDPA_MASK",
            self.diagnostic_codes(malformed_mask, ["wasm"], weights=weights),
        )

        for params in (
            {"heads": 4},
            {"heads": 3, "causal": False},
            {"heads": 4, "causal": 0},
            {"heads": 4, "causal": False, "scale": 1e100},
            {"heads": 4, "causal": False, "scale": 10**1000},
            {"heads": 4, "causal": False, "extra": 1},
        ):
            with self.subTest(params=params):
                malformed_params = copy.deepcopy(graph)
                malformed_params["nodes"][0]["params"] = params
                self.assertIn(
                    "VXDESC_QSDPA_PARAMS",
                    self.diagnostic_codes(
                        malformed_params, ["wasm"], weights=weights,
                    ),
                )

        oversized = copy.deepcopy(graph)
        oversized["nodes"][0]["inputs"].pop("mask")
        oversized["inputs"]["q"]["shape"] = [65_536, 65_536, 4]
        oversized["inputs"]["k"]["shape"] = [65_536, 65_536, 4]
        oversized["inputs"]["v"]["shape"] = [65_536, 65_536, 4]
        oversized["nodes"][0]["outputs_shape"]["out"] = [65_536, 65_536, 4]
        oversized["nodes"][0]["params"]["heads"] = 1
        self.assertIn(
            "VXDESC_QSDPA_BOUND",
            self.diagnostic_codes(oversized, ["wasm"], weights=weights),
        )

        oversized_mask = copy.deepcopy(graph)
        oversized_mask["inputs"]["q"]["shape"] = [65_536, 256, 4]
        oversized_mask["inputs"]["k"]["shape"] = [65_536, 256, 4]
        oversized_mask["inputs"]["v"]["shape"] = [65_536, 256, 4]
        oversized_mask["inputs"]["mask"]["shape"] = [65_536, 256, 256]
        oversized_mask["nodes"][0]["outputs_shape"]["out"] = [65_536, 256, 4]
        oversized_mask["nodes"][0]["params"]["heads"] = 1
        self.assertIn(
            "VXDESC_QSDPA_BOUND",
            self.diagnostic_codes(
                oversized_mask, ["wasm"], weights=weights,
            ),
        )

        for label, q_scale, k_scale in (
            ("underflow", 1e-30, 1e-30),
            ("finite-multiplier-score-overflow", 1e18, 1e18),
        ):
            with self.subTest(score=label):
                invalid_score = copy.deepcopy(graph)
                invalid_weights = dict(weights)
                table = invalid_score["quantization"]["tensors"]
                invalid_weights[table["q"]["scale_tensor"]] = np.asarray(
                    [q_scale], dtype=np.float32,
                )
                invalid_weights[table["k"]["scale_tensor"]] = np.asarray(
                    [k_scale], dtype=np.float32,
                )
                self.assertIn(
                    "VXDESC_QSDPA_SCORE",
                    self.diagnostic_codes(
                        invalid_score, ["wasm"], weights=invalid_weights,
                    ),
                )

    def test_quantized_norms_use_the_exact_portable_physical_contract(self):
        for op_type in ("QGroupNorm", "QLayerNorm"):
            with self.subTest(op_type=op_type):
                graph, weights = qnorm_graph(op_type)
                result = validate_graph(
                    graph, ["portable"], weights=weights,
                )
                self.assertTrue(result.supported, result.diagnostics)
                self.assertNotIn(
                    "VXDESC_UNVALIDATED",
                    {diagnostic.code for diagnostic in result.diagnostics},
                )

    def test_quantized_norms_reject_bad_affines_parameters_and_bounds(self):
        group_graph, group_weights = qnorm_graph("QGroupNorm")
        nonfinite_group_weights = dict(group_weights)
        nonfinite_group_weights["gamma"] = np.asarray(
            [1.0] * 7 + [np.inf], dtype=np.float32,
        )
        self.assertIn(
            "VXDESC_QGROUPNORM_AFFINE",
            self.diagnostic_codes(
                group_graph, ["wasm"], weights=nonfinite_group_weights,
            ),
        )

        bad_groups = copy.deepcopy(group_graph)
        bad_groups["nodes"][0]["params"]["num_groups"] = 3
        self.assertIn(
            "VXDESC_QGROUPNORM_PARAMS",
            self.diagnostic_codes(
                bad_groups, ["wasm"], weights=group_weights,
            ),
        )
        malformed_layout = copy.deepcopy(group_graph)
        malformed_layout["nodes"][0]["params"]["data_layout"] = ["NHWC"]
        self.assertIn(
            "VXDESC_QGROUPNORM_PARAMS",
            self.diagnostic_codes(
                malformed_layout, ["wasm"], weights=group_weights,
            ),
        )

        layer_graph, layer_weights = qnorm_graph("QLayerNorm")
        bad_d_model = copy.deepcopy(layer_graph)
        bad_d_model["nodes"][0]["params"]["d_model"] = 4
        self.assertIn(
            "VXDESC_QLAYERNORM_PARAMS",
            self.diagnostic_codes(
                bad_d_model, ["wasm"], weights=layer_weights,
            ),
        )
        bad_d_model["nodes"][0]["params"]["d_model"] = 10**1000
        self.assertIn(
            "VXDESC_QLAYERNORM_PARAMS",
            self.diagnostic_codes(
                bad_d_model, ["wasm"], weights=layer_weights,
            ),
        )

        rank_nine = copy.deepcopy(layer_graph)
        rank_nine_shape = [1, 1, 1, 1, 1, 1, 1, 1, 8]
        rank_nine["inputs"]["x"]["shape"] = rank_nine_shape
        rank_nine["nodes"][0]["outputs_shape"]["out"] = rank_nine_shape
        self.assertIn(
            "VXDESC_QLAYERNORM_GEOMETRY",
            self.diagnostic_codes(
                rank_nine, ["wasm"], weights=layer_weights,
            ),
        )

        oversized = copy.deepcopy(layer_graph)
        oversized_shape = [65_536, 65_536, 8]
        oversized["inputs"]["x"]["shape"] = oversized_shape
        oversized["nodes"][0]["outputs_shape"]["out"] = oversized_shape
        self.assertIn(
            "VXDESC_QLAYERNORM_BOUND",
            self.diagnostic_codes(
                oversized, ["wasm"], weights=layer_weights,
            ),
        )

    def test_declared_package_class_must_match_live_descriptors(self):
        graph = graph_with_node("Identity")
        graph["source"] = {"package_class": "w8a8-v1"}
        self.assertIn(
            "VXPKG_CLASS",
            self.diagnostic_codes(graph, ["portable"]),
        )


if __name__ == "__main__":
    unittest.main()
