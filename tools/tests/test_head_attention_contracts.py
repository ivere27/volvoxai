"""Head relationships and target refusals without loading a runtime library."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.exporter.capabilities import validate_graph
from tools.exporter.operator_shape_contracts import (
    OperatorShapeContractError,
    infer_concrete_operator_shapes,
    prove_operator_shape_domain,
)
from tools.exporter.portable_domain import (
    prove_portable_graph_domain,
)
from tools.exporter.shape_system import ShapeEnvironment


def cross_request(heads=4, kv_heads=2, dimension=128):
    return {
        "inputs": {
            "q": {"shape": [3, 2, heads * dimension], "dtype": "float32"},
            "k": {"shape": [3, 5, kv_heads * dimension], "dtype": "float32"},
            "v": {"shape": [3, 5, kv_heads * dimension], "dtype": "float32"},
        },
        "params": {"heads": heads, "kv_heads": kv_heads, "causal": True},
    }


def graph_for(operator, request, *, dynamic=False):
    request = copy.deepcopy(request)
    shape = request["inputs"]["q" if operator == "CrossSDPA" else "input"]["shape"]
    graph = {
        "format": "volvox-graph/v1", "dimensions": {}, "inputs": request["inputs"],
        "nodes": [{"id": "operation", "opType": operator,
                   "inputs": {name: name for name in request["inputs"]},
                   "params": request["params"],
                   "outputs": {"out": {"tensor": "y", "shape": list(shape),
                                       "dtype": "float32"}}}],
        "outputs": ["y"],
    }
    if dynamic:
        graph["dimensions"] = {"S": {"min": 1, "max": 5}}
        for name, value in graph["inputs"].items():
            if name in {"input", "q", "k", "v"}:
                value["shape"][-2] = "S"
        graph["nodes"][0]["outputs"]["out"]["shape"][-2] = "S"
    return graph


class HeadAttentionContractTests(unittest.TestCase):
    def test_attention_vector_corpus_matches_python(self):
        path = Path(__file__).resolve().parents[2] / "tests/contracts/shape_vectors/attention_vectors.json"
        vectors = json.loads(path.read_text())
        for case in vectors["success_cases"]:
            with self.subTest(case=case["id"]):
                actual = infer_concrete_operator_shapes(case["operators"][0], case["request"])
                for port, expected in case["expected"].items():
                    self.assertEqual(list(actual[port].shape), expected["shape"])
                    self.assertEqual(actual[port].dtype, expected["dtype"])
        for case in vectors["failure_cases"]:
            with self.subTest(case=case["id"]):
                with self.assertRaises(OperatorShapeContractError) as failed:
                    infer_concrete_operator_shapes(case["operators"][0], case["request"])
                self.assertEqual(failed.exception.code, case["expected_error"]["code"])

    def test_mha_gqa_mqa_share_derived_dimension(self):
        for kv_heads in (4, 2, 1):
            for dimension in (8, 64, 128):
                with self.subTest(kv_heads=kv_heads, dimension=dimension):
                    request = cross_request(kv_heads=kv_heads, dimension=dimension)
                    result = infer_concrete_operator_shapes("CrossSDPA", request)
                    self.assertEqual(result["out"].shape, (3, 2, 4 * dimension))
                    graph = graph_for("CrossSDPA", request, dynamic=True)
                    for backend in ("native-cpu", "wasm"):
                        self.assertTrue(validate_graph(graph, [backend]).supported)
                        self.assertEqual(prove_portable_graph_domain(graph, {}, [backend]).backend_members,
                                         (backend,))
        legacy = cross_request(kv_heads=4)
        explicit = infer_concrete_operator_shapes("CrossSDPA", legacy)
        del legacy["params"]["kv_heads"]
        self.assertEqual(explicit, infer_concrete_operator_shapes("CrossSDPA", legacy))

    def test_explicit_new_attention_domain_qualified_gpu_route(self):
        graphs = (
            graph_for("CrossSDPA", cross_request(kv_heads=4)),
            graph_for("RoPE", {"inputs": {"input": {"shape": [3, 2, 512],
                                                     "dtype": "float32"}},
                                "params": {"heads": 4}}),
        )
        for graph in graphs:
            for backend in ("webgpu", "backend:cuda", "backend:vulkan",
                            "backend:opengl", "backend:metal"):
                with self.subTest(operator=graph["nodes"][0]["opType"], backend=backend):
                    result = validate_graph(graph, [backend])
                    self.assertTrue(result.supported)
                    proof = prove_portable_graph_domain(graph, {}, [backend])
                    self.assertEqual(proof.backend_members, (backend,))

    def test_bounded_rope_proves_per_head_width_for_every_binding(self):
        request = {"inputs": {"input": {"shape": [3, "S", "W"], "dtype": "float32"}},
                   "params": {"heads": 4}}
        def environment(multiple):
            return ShapeEnvironment([
                {"name": "S", "min": 1, "max": 5},
                {"name": "W", "min": 16, "max": 32, "multiple_of": multiple},
            ])
        request["environment"] = environment(8)
        proof = prove_operator_shape_domain("RoPE", request)
        self.assertTrue(proof.supported, getattr(proof, "reason", None))
        request["environment"] = environment(4)
        self.assertFalse(prove_operator_shape_domain("RoPE", request).supported)
        request["params"]["rotary_dim"] = 4
        self.assertTrue(prove_operator_shape_domain("RoPE", request).supported)
        request["params"]["rotary_dim"] = 6
        self.assertFalse(prove_operator_shape_domain("RoPE", request).supported)

    def test_head_parameters_reject_wrong_types_and_nonpositive_counts(self):
        for operator, parameter, request in (
            ("CrossSDPA", "kv_heads", cross_request()),
            ("RoPE", "heads", {"inputs": {"input": {"shape": [1, 3, 16],
                                                      "dtype": "float32"}},
                                "params": {}}),
        ):
            for value in (None, True, 0, -1, 1.5, "2"):
                with self.subTest(operator=operator, value=value):
                    request["params"][parameter] = value
                    with self.assertRaises(OperatorShapeContractError) as failed:
                        infer_concrete_operator_shapes(operator, request)
                    self.assertEqual(failed.exception.code, "INVALID_PARAMS")

    def test_flat_head_rope_preserves_odd_dimension_partial_tail(self):
        request = {"inputs": {"input": {"shape": [3, 2, 20], "dtype": "float32"}},
                   "params": {"heads": 4, "rotary_dim": 4}}
        self.assertEqual(infer_concrete_operator_shapes("RoPE", request)["out"].shape,
                         (3, 2, 20))
        for backend in ("native-cpu", "wasm"):
            self.assertTrue(validate_graph(graph_for("RoPE", request), [backend]).supported)
        del request["params"]["rotary_dim"]
        with self.assertRaises(OperatorShapeContractError) as failed:
            infer_concrete_operator_shapes("RoPE", request)
        self.assertEqual(failed.exception.code, "INVALID_PARAMS")

    def test_long_memory_permutations_masks_gqa_and_row_equivalence(self):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="volvox-attention-accuracy-") as directory:
            binary = Path(directory) / "accuracy"
            subprocess.run([compiler, "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
                "-I", str(root / "native/src/kernels"), str(root / "native/tests/test_attention_f32_accuracy.c"),
                "-lm", "-o", str(binary)], check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
