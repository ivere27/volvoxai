from __future__ import annotations

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
        "source": {"frontend": "test"},
        "inputs": {"x": {"shape": [1, 2], "dtype": "int8"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "linear",
            "opType": "QLinear",
            "inputs": {"input": "x", "weight": "w", "bias": "bias"},
            "outputs": {"out": "y"},
            "outputs_shape": {"out": [1, 2]},
            "outputs_dtype": {"out": "int8"},
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


def identity_fixture(params=None):
    return {
        "format": "volvox-graph/v1",
        "inputs": {"x": {"shape": [1, 4], "dtype": "float32"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "identity",
            "opType": "Identity",
            "inputs": {"input": "x"},
            "outputs": {"out": "y"},
            "outputs_shape": {"out": [1, 4]},
            "outputs_dtype": {"out": "float32"},
            "params": {} if params is None else params,
        }],
    }, {}


class RuntimeIRTests(unittest.TestCase):
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

    def test_strict_package_round_trip_preserves_refs_and_extensions(self):
        document, tensors = fixture()
        graph = import_runtime_package(document, tensors)
        self.assertEqual(graph.nodes[0].input_map()["weight"], "w")
        self.assertEqual(graph.tensors["w"].quantization.axis, 0)
        exported, output_tensors = export_runtime_package(graph, tensors)
        self.assertEqual(exported["format"], "volvox-graph/v1")
        self.assertEqual(exported["source"], {"frontend": "test"})
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
            (lambda document: document["nodes"][0].pop("outputs_dtype"), "VXRTIR011"),
            (lambda document: document["nodes"][0]["outputs_dtype"].update({"out": "float16"}), "VXRTIR014"),
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
            lambda document: document["nodes"][0]["outputs_shape"].update({
                "out": [1 << 53, 2],
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
        self.assertEqual(caught.exception.diagnostic.code, "VXDESC_PORTS")
        self.assertIn("weight", caught.exception.diagnostic.message)

    def test_rejects_incomplete_output_metadata_maps(self):
        for field, code in (
            ("outputs_shape", "VXRTIR020"),
            ("outputs_dtype", "VXRTIR021"),
        ):
            with self.subTest(field=field):
                document, tensors = fixture()
                document["nodes"][0][field]["extra"] = [1] if field == "outputs_shape" else "int8"
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(document, tensors)
                self.assertEqual(caught.exception.diagnostic.code, code)

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

    def test_semantic_scale_round_trips_but_runtime_ir_cannot_reemit_affine_params(self):
        document, tensors = identity_fixture({"scale": 0.5})
        graph = import_runtime_package(document, tensors)
        exported, _ = export_runtime_package(graph, tensors)
        self.assertEqual(exported["nodes"][0]["params"], {"scale": 0.5})

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
