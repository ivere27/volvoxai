from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.exporter.errors import ExporterError
from tools.exporter.ir import OpAttribute
from tools.exporter.runtime_ir import (
    export_runtime_package,
    import_runtime_package,
    load_runtime_document,
    prove_dynamic_quantized_runtime_domain,
)
from tools.exporter.optimizer.typed_pipeline import optimize_runtime_package
from tools.exporter.portable_domain import (
    PortableDomainProofError,
    prove_portable_graph_domain,
)


def fixture():
    tensors = {
        "w": np.asarray([[1, 2], [3, 4]], dtype=np.int8),
        "bias": np.asarray([0, 0], dtype=np.int32),
        "w.scale": np.asarray([0.25, 0.5], dtype=np.float32),
        "w.zero": np.asarray([0, 0], dtype=np.int8),
        "x.scale": np.asarray([0.125], dtype=np.float32),
        "x.zero": np.asarray([0], dtype=np.int8),
        "y.scale": np.asarray([0.25], dtype=np.float32),
        "y.zero": np.asarray([0], dtype=np.int8),
    }
    document = {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 2], "dtype": "int8"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "linear",
            "opType": "QLinear",
            "inputs": {"input": "x", "weight": "w", "bias": "bias"},
            "outputs": {"out": {
                "tensor": "y", "shape": [1, 2], "dtype": "int8",
            }},
            "params": {},
        }],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "x": {"scheme": "per_tensor", "scale_tensor": "x.scale",
                      "zero_point_tensor": "x.zero"},
                "w": {"scheme": "per_axis", "axis": 0,
                      "scale_tensor": "w.scale", "zero_point_tensor": "w.zero"},
                "y": {"scheme": "per_tensor", "scale_tensor": "y.scale",
                      "zero_point_tensor": "y.zero"},
            },
        },
    }
    return document, tensors


def dynamic_quantized_fixture():
    document, tensors = fixture()
    document["dimensions"] = {"B": {"min": 1, "max": 4}}
    document["inputs"]["x"]["shape"] = ["B", 2]
    document["nodes"][0]["outputs"]["out"]["shape"] = ["B", 2]
    return document, tensors


def identity_fixture(params=None):
    return {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 4], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "identity",
            "opType": "Identity",
            "inputs": {"input": "x"},
            "outputs": {"out": {
                "tensor": "y", "shape": [1, 4], "dtype": "float32",
            }},
            "params": {} if params is None else params,
        }],
    }, {}


def resize_fixture():
    return {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {
            "x": {"shape": [2, 3, 5, 4], "dtype": "float32"},
        },
        "outputs": ["y"],
        "nodes": [{
            "id": "resize",
            "opType": "Resize",
            "inputs": {"input": "x"},
            "outputs": {"out": {
                "tensor": "y", "shape": [2, 6, 7, 4],
                "dtype": "float32",
            }},
            "params": {"mode": "linear"},
        }],
    }, {}


def linear_fixture(layout: str):
    return {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"shape": [1, 2], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "linear",
            "opType": "Linear",
            "inputs": {"input": "x", "weight": "weight"},
            "outputs": {"out": {
                "tensor": "y", "shape": [1, 3], "dtype": "float32",
            }},
            "params": {"weight_layout": layout},
        }],
    }, {"weight": np.zeros((2, 3), dtype=np.float32)}


class RuntimeIRTests(unittest.TestCase):
    def test_dynamic_quantized_import_requires_and_accepts_exact_domain_proof(self):
        document, tensors = dynamic_quantized_fixture()
        with self.assertRaises(ExporterError) as missing:
            import_runtime_package(document, tensors)
        self.assertEqual(missing.exception.diagnostic.code, "VXRTIR037")

        proof = prove_dynamic_quantized_runtime_domain(document, tensors)
        self.assertIsNotNone(proof)
        graph = import_runtime_package(
            document,
            tensors,
            bounded_domain_proof=proof,
        )
        self.assertEqual(graph.tensors["x"].shape, ("B", 2))
        self.assertEqual(
            proof.backend_members,
            ("cpu-js", "wasm", "webgpu", "native-cpu"),
        )

    def test_dynamic_quantized_import_rejects_browser_only_operator(self):
        document, tensors = dynamic_quantized_fixture()
        document["inputs"]["indices"] = {
            "shape": ["B", 2],
            "dtype": "int32",
        }
        document["nodes"][0] = {
            "id": "gather",
            "opType": "GatherElements",
            "inputs": {"input": "x", "indices": "indices"},
            "outputs": {"out": {
                "tensor": "y", "shape": ["B", 2], "dtype": "int8",
            }},
            "params": {"axis": 1},
        }
        tensors["y.scale"] = np.asarray([0.125], dtype=np.float32)

        browser_proof = prove_portable_graph_domain(
            document,
            tensors,
            ("cpu-js", "wasm", "webgpu"),
        )
        self.assertIsNotNone(browser_proof)
        with self.assertRaises(ExporterError) as caught:
            prove_dynamic_quantized_runtime_domain(document, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR037")
        self.assertIn("native-cpu", caught.exception.diagnostic.message)

    def test_portable_proof_rejects_retired_shape_system_field(self):
        document, tensors = dynamic_quantized_fixture()
        document["shape_system"] = "volvox-bounded-shape/v1"
        with self.assertRaises(PortableDomainProofError) as caught:
            prove_portable_graph_domain(document, tensors, ("cpu-js",))
        self.assertEqual(caught.exception.code, "VXDOMAIN_SCHEMA")
        self.assertIn("unsupported field 'shape_system'", str(caught.exception))

    def test_dynamic_quantized_proof_rejects_unsupported_shape_corner(self):
        document, tensors = dynamic_quantized_fixture()
        document["nodes"][0]["outputs"]["out"]["shape"] = ["B", 3]

        with self.assertRaises(ExporterError) as caught:
            prove_dynamic_quantized_runtime_domain(document, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR037")
        self.assertIn("bounded-domain", caught.exception.diagnostic.message)

    def test_dynamic_quantized_import_rejects_every_stale_proof_binding(self):
        document, tensors = dynamic_quantized_fixture()
        proof = prove_dynamic_quantized_runtime_domain(document, tensors)
        assert proof is not None

        stale_fingerprint = replace(
            proof,
            graph_fingerprint="sha256:" + "0" * 64,
        )
        with self.assertRaises(ExporterError) as fingerprint_error:
            import_runtime_package(
                document,
                tensors,
                bounded_domain_proof=stale_fingerprint,
            )
        self.assertEqual(fingerprint_error.exception.diagnostic.code, "VXRTIR037")

        changed_tensors = dict(tensors)
        changed_tensors["w"] = np.asarray(tensors["w"]).copy()
        changed_tensors["w"][0, 0] += np.int8(1)
        with self.assertRaises(ExporterError) as tensor_error:
            import_runtime_package(
                document,
                changed_tensors,
                bounded_domain_proof=proof,
            )
        self.assertEqual(tensor_error.exception.diagnostic.code, "VXRTIR037")

        changed_bounds = copy.deepcopy(document)
        changed_bounds["dimensions"]["B"]["max"] = 3
        with self.assertRaises(ExporterError) as bounds_error:
            import_runtime_package(
                changed_bounds,
                tensors,
                bounded_domain_proof=proof,
            )
        self.assertEqual(bounds_error.exception.diagnostic.code, "VXRTIR037")

        changed_quantization = copy.deepcopy(document)
        changed_quantization["quantization"]["tensors"]["x"][
            "scale_tensor"
        ] = "y.scale"
        with self.assertRaises(ExporterError) as quantization_error:
            import_runtime_package(
                changed_quantization,
                tensors,
                bounded_domain_proof=proof,
            )
        self.assertEqual(
            quantization_error.exception.diagnostic.code,
            "VXRTIR037",
        )

        wrong_backends = replace(proof, backend_members=("cpu-js",))
        with self.assertRaises(ExporterError) as backend_error:
            import_runtime_package(
                document,
                tensors,
                bounded_domain_proof=wrong_backends,
            )
        self.assertEqual(backend_error.exception.diagnostic.code, "VXRTIR037")

    def test_failed_dynamic_quantized_optimization_is_transactional(self):
        document, tensors = dynamic_quantized_fixture()
        document["nodes"][0]["outputs"]["out"]["shape"] = ["B", 3]
        document_before = copy.deepcopy(document)
        tensor_ids = {name: id(value) for name, value in tensors.items()}
        tensor_values = {
            name: np.asarray(value).copy() for name, value in tensors.items()
        }

        with self.assertRaises(ExporterError):
            optimize_runtime_package(document, tensors)

        self.assertEqual(document, document_before)
        self.assertEqual(
            {name: id(value) for name, value in tensors.items()},
            tensor_ids,
        )
        for name, value in tensor_values.items():
            np.testing.assert_array_equal(tensors[name], value)

    def test_rejects_retired_linear_layout_tokens(self):
        for layout in ("IN_OUT", "OUT_IN"):
            with self.subTest(layout=layout):
                document, tensors = linear_fixture(layout)
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(document, tensors)
                self.assertEqual(caught.exception.diagnostic.code, "VXRTIR035")
                self.assertIn("din_dout", caught.exception.diagnostic.message)

    def test_strict_document_loader_accepts_a_valid_graph(self):
        document, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(load_runtime_document(path), document)

    def test_strict_document_loader_rejects_ambiguous_json(self):
        cases = (
            ('{"format":"volvox-graph/v1","format":"other"}', "VXRTIR028"),
            ('{"value":NaN}', "VXRTIR024"),
            ('[]', "VXRTIR030"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            for source, code in cases:
                with self.subTest(code=code):
                    path.write_text(source, encoding="utf-8")
                    with self.assertRaises(ExporterError) as caught:
                        load_runtime_document(path)
                    self.assertEqual(caught.exception.diagnostic.code, code)

    def test_rejects_every_non_current_graph_format(self):
        document, tensors = fixture()
        document["format"] = "volvox-graph/v0"
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR004")

    def test_strict_package_round_trip_preserves_affine_refs(self):
        document, tensors = fixture()
        graph = import_runtime_package(document, tensors)
        self.assertEqual(graph.nodes[0].input_map()["weight"], "w")
        self.assertEqual(graph.tensors["w"].quantization.axis, 0)
        graph.abi_changes.append({"kind": "out-of-band-only"})
        exported, output_tensors = export_runtime_package(graph, tensors)
        self.assertEqual(exported["format"], "volvox-graph/v1")
        self.assertNotIn("source", exported)
        self.assertEqual(
            set(exported),
            {
                "format", "dimensions", "inputs", "nodes",
                "outputs", "quantization",
            },
        )
        self.assertEqual(
            exported["quantization"]["tensors"]["w"]["scale_tensor"],
            "w.scale",
        )
        self.assertEqual(set(output_tensors), set(tensors))

    def test_old_inline_graph_is_not_migrated(self):
        document, tensors = fixture()
        document["inputs"]["x"]["quantization"] = {
            "scale": 0.125, "zero_point": 0,
        }
        with self.assertRaises(ExporterError):
            import_runtime_package(document, tensors)

    def test_rejects_implicit_dtypes_and_op_alias(self):
        for mutate, code in (
            (lambda document: document["inputs"]["x"].pop("dtype"), "VXRTIR008"),
            (lambda document: document["inputs"]["x"].update({"dtype": "float16"}), "VXRTIR008"),
            (lambda document: document["nodes"][0]["outputs"]["out"].pop("dtype"), "VXRTIR020"),
            (lambda document: document["nodes"][0]["outputs"]["out"].update({"dtype": "float16"}), "VXRTIR014"),
            (lambda document: document["nodes"][0].update({"op": "QLinear"}), "VXRTIR019"),
            (lambda document: document["nodes"][0].update({"params": None}), "VXRTIR022"),
        ):
            with self.subTest(code=code):
                document, tensors = fixture()
                mutate(document)
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(document, tensors)
                self.assertEqual(caught.exception.diagnostic.code, code)

    def test_rejects_nonfinite_json_and_unsafe_integer_dimensions(self):
        for mutate in (
            lambda document: document["nodes"][0].update({
                "params": {"alpha": float("nan")},
            }),
            lambda document: document["inputs"]["x"].update({
                "shape": [1 << 53, 2],
            }),
            lambda document: document["nodes"][0]["outputs"]["out"].update({
                "shape": [1 << 53, 2],
            }),
        ):
            with self.subTest(mutate=mutate):
                document, tensors = fixture()
                mutate(document)
                with self.assertRaises(ExporterError):
                    import_runtime_package(document, tensors)

    def test_rejects_malformed_float_operator_contract(self):
        document, tensors = identity_fixture()
        document["nodes"][0]["opType"] = "Conv2D"
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR035")
        self.assertIn("weight", caught.exception.diagnostic.message)

    def test_static_resize_inference_receives_declared_output_assertion(self):
        graph = import_runtime_package(*resize_fixture())
        self.assertEqual(graph.tensors["y"].shape, (2, 6, 7, 4))

        malformed, tensors = resize_fixture()
        malformed["nodes"][0]["outputs"]["out"]["shape"] = [3, 6, 7, 4]
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(malformed, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR035")
        self.assertNotIn("INVALID_OUTPUT_PORTS", caught.exception.diagnostic.message)

    def test_rejects_incomplete_or_aliased_output_assertions(self):
        for field in ("shape", "dtype", "tensor"):
            with self.subTest(field=field):
                document, tensors = fixture()
                document["nodes"][0]["outputs"]["out"].pop(field)
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(document, tensors)
                self.assertEqual(caught.exception.diagnostic.code, "VXRTIR020")

    def test_rejects_empty_public_outputs_at_the_persisted_boundary(self):
        document, tensors = identity_fixture()
        document["outputs"] = []
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXRTIR015")

    def test_rejects_every_recursive_retired_affine_param_field(self):
        forbidden = (
            "quantization",
            "zero_point",
            "input_scale", "input_zero_point",
            "output_scale", "output_zero_point",
            "weight_scale", "weight_zero_point",
            "scales", "zero_points",
            "scale_tensor", "zero_point_tensor",
        )
        for field in forbidden:
            with self.subTest(field=field):
                document, tensors = identity_fixture({
                    "scale": 0.5,
                    "private": [{"affine": {field: 0.25}}],
                })
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(document, tensors)
                self.assertEqual(caught.exception.diagnostic.code, "VXRTIR023")
                self.assertIn(
                    f"params.private[0].affine.{field}",
                    caught.exception.diagnostic.message,
                )

    def test_semantic_scale_is_not_retired_but_operator_contract_still_applies(self):
        document, tensors = identity_fixture({"scale": 0.5})
        with self.assertRaises(ExporterError) as invalid_params:
            import_runtime_package(document, tensors)
        self.assertEqual(invalid_params.exception.diagnostic.code, "VXRTIR035")
        self.assertNotEqual(invalid_params.exception.diagnostic.code, "VXRTIR023")

        graph = import_runtime_package(*identity_fixture())
        graph.nodes[0].attributes = (OpAttribute(
            "params", "volvox.params",
            {"scale": 0.5, "hidden": {"output_zero_point": 3}},
        ),)
        with self.assertRaises(ExporterError) as caught:
            export_runtime_package(graph, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR042")

    def test_qlinear_requires_canonical_i32_bias_operand(self):
        document, tensors = fixture()
        del document["nodes"][0]["inputs"]["bias"]
        with self.assertRaises(ExporterError) as caught:
            import_runtime_package(document, tensors)
        self.assertEqual(caught.exception.diagnostic.code, "VXIR034")

    def test_dead_initializer_is_pruned_on_export(self):
        document, tensors = fixture()
        tensors["dead"] = np.asarray([1.0], dtype=np.float32)
        graph = import_runtime_package(document, tensors)
        del graph.tensors["dead"]
        exported, output_tensors = export_runtime_package(graph, tensors)
        self.assertNotIn("dead", output_tensors)
        self.assertEqual(exported["outputs"], ["y"])


if __name__ == "__main__":
    unittest.main()
