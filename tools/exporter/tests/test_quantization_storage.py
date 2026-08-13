from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
import unittest

import numpy as np

from tools.exporter.capabilities import validate_graph
from tools.exporter.errors import ExporterError
from tools.exporter.quantization_storage import (
    GRAPH_FORMAT,
    QUANTIZATION_FORMAT,
    externalize_quantization,
    prune_external_quantization,
    reject_legacy_safetensors_metadata,
    validate_external_quantization,
)


def _new_v1(graph):
    result = copy.deepcopy(graph)
    result["dimensions"] = {}
    for index, node in enumerate(result.get("nodes", [])):
        outputs = node.get("outputs", {})
        shapes = node.pop("outputs_shape", {})
        dtypes = node.pop("outputs_dtype", {})
        node["outputs"] = {
            port: {
                "tensor": tensor,
                "shape": shapes[port],
                "dtype": dtypes[port],
            }
            for port, tensor in outputs.items()
        }
        node.setdefault("id", f"node-{index}")
        node.setdefault("params", {})
    return result


class QuantizationStorageTests(unittest.TestCase):
    def authoring_graph(self):
        graph = _new_v1({
            "format": GRAPH_FORMAT,
            "inputs": {
                "x": {
                    "shape": [1, 2],
                    "dtype": "int8",
                }
            },
            "nodes": [
                {
                    "opType": "QLinear",
                    "inputs": {
                        "input": "x",
                        "weight": "w",
                        "bias": "bias",
                    },
                    "outputs": {"out": "y"},
                    "outputs_shape": {"out": [1, 2]},
                    "outputs_dtype": {"out": "uint8"},
                    "params": {},
                }
            ],
            "outputs": ["y"],
        })
        descriptors = {
            "x": {
                "scheme": "per_tensor",
                "scale": 0.25,
                "zero_point": -3,
            },
            "w": {
                "scheme": "per_axis",
                "axis": 0,
                "scales": [0.5, 0.75],
                "zero_points": [0, 0],
            },
            "y": {
                "scheme": "per_tensor",
                "scale": 0.125,
                "zero_point": 123,
            },
        }
        return graph, descriptors

    def test_externalizes_every_numeric_descriptor(self):
        graph, descriptors = self.authoring_graph()
        tensors = {
            "w": np.asarray([[1, 2], [3, 4]], dtype=np.int8),
            "bias": np.asarray([0, 0], dtype=np.int32),
        }

        report = externalize_quantization(graph, tensors, descriptors)

        self.assertEqual(graph["format"], GRAPH_FORMAT)
        self.assertEqual(graph["quantization"]["format"], QUANTIZATION_FORMAT)
        self.assertEqual(report.tensors, 3)
        self.assertEqual(report.parameters_reused, 0)

        # The document contains tensor references and semantic attributes only;
        # no former numeric descriptor field survives serialization.
        encoded = json.dumps(graph)
        self.assertNotIn('"scale":', encoded)
        self.assertNotIn('"scales":', encoded)
        self.assertNotIn('"zero_point":', encoded)
        self.assertNotIn('"zero_points":', encoded)

        resolved = validate_external_quantization(graph, tensors)
        np.testing.assert_array_equal(resolved["x"].scales, np.asarray([0.25], np.float32))
        np.testing.assert_array_equal(resolved["x"].zero_points, np.asarray([-3], np.int8))
        np.testing.assert_array_equal(resolved["w"].scales, np.asarray([0.5, 0.75], np.float32))
        np.testing.assert_array_equal(resolved["y"].zero_points, np.asarray([123], np.uint8))
        validation = validate_graph(graph, ["cpu-js"], weights=tensors)
        self.assertEqual(validation.diagnostics, ())

    def test_explicit_quantize_parameters_are_reused_bit_exactly(self):
        graph = _new_v1({
            "format": "volvox-graph/v1",
            "inputs": {"x": {"shape": [2], "dtype": "float32"}},
            "nodes": [{
                "opType": "QuantizeLinear",
                "inputs": {"input": "x", "scale": "s", "zero_point": "z"},
                "outputs": {"out": "q"},
                "outputs_shape": {"out": [2]},
                "outputs_dtype": {"out": "uint8"},
            }],
            "outputs": ["q"],
        })
        tensors = {
            "s": np.asarray([0.5], dtype=np.float32),
            "z": np.asarray([127], dtype=np.uint8),
        }

        report = externalize_quantization(graph, tensors, {
            "q": {
                "scheme": "per_tensor", "scale": 0.5, "zero_point": 127,
            },
        })

        self.assertEqual(report.parameters_reused, 2)
        self.assertEqual(report.scales_created, 0)
        self.assertEqual(report.zero_points_created, 0)
        self.assertEqual(graph["quantization"]["tensors"]["q"], {
            "scheme": "per_tensor",
            "scale_tensor": "s",
            "zero_point_tensor": "z",
        })

    def test_parameter_names_are_short_deterministic_and_collision_safe(self):
        graph, descriptors = self.authoring_graph()
        digest = hashlib.sha256(b"x").hexdigest()
        occupied = f"__quant__.{digest[:16]}.scale"
        tensors = {
            "w": np.zeros((2, 2), dtype=np.int8),
            "bias": np.asarray([0, 0], dtype=np.int32),
            occupied: np.asarray([999.0], dtype=np.float32),
        }

        externalize_quantization(graph, tensors, descriptors)

        scale_name = graph["quantization"]["tensors"]["x"]["scale_tensor"]
        self.assertNotEqual(scale_name, occupied)
        self.assertLess(len(scale_name.encode("utf-8")), 128)
        np.testing.assert_array_equal(tensors[occupied], np.asarray([999.0], np.float32))

    def test_failed_conversion_is_transactional(self):
        graph, descriptors = self.authoring_graph()
        descriptors["y"]["scale"] = -1.0
        tensors = {
            "w": np.zeros((2, 2), dtype=np.int8),
            "bias": np.asarray([0, 0], dtype=np.int32),
        }
        graph_before = copy.deepcopy(graph)
        tensors_before = {name: value.copy() for name, value in tensors.items()}

        with self.assertRaises(ExporterError) as caught:
            externalize_quantization(graph, tensors, descriptors)

        self.assertEqual(caught.exception.diagnostic.code, "VXQSTORE016")
        self.assertEqual(graph, graph_before)
        self.assertEqual(set(tensors), set(tensors_before))
        for name in tensors:
            np.testing.assert_array_equal(tensors[name], tensors_before[name])

    def test_validator_rejects_inline_values_in_v1(self):
        graph = _new_v1({
            "format": GRAPH_FORMAT,
            "inputs": {"x": {
                "shape": [1], "dtype": "int8",
                "quantization": {"scale": 1.0, "zero_point": 0},
            }},
            "nodes": [],
            "outputs": ["x"],
        })
        with self.assertRaises(ExporterError) as caught:
            validate_external_quantization(graph, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXQSTORE022")

        graph = _new_v1({
            "format": GRAPH_FORMAT,
            "inputs": {},
            "nodes": [],
            "standaloneTensors": {
                "state": {
                    "shape": [1], "dtype": "int8", "hasBuffer": False,
                    "quantization": {"scale": 1.0, "zero_point": 0},
                },
            },
            "outputs": ["state"],
        })
        with self.assertRaises(ExporterError) as caught:
            validate_external_quantization(graph, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXQSTORE024")

    def test_node_params_reject_every_recursive_retired_affine_field(self):
        forbidden = (
            "quantization",
            "zero_point",
            "input_scale", "input_zero_point",
            "output_scale", "output_zero_point",
            "weight_scale", "weight_zero_point",
            "scales", "zero_points",
            "scale_tensor", "zero_point_tensor",
        )
        graph = _new_v1({
            "format": GRAPH_FORMAT,
            "inputs": {"x": {"shape": [1], "dtype": "float32"}},
            "nodes": [{
                "id": "identity",
                "opType": "Identity",
                "inputs": {"input": "x"},
                "outputs": {"out": "y"},
                "outputs_shape": {"out": [1]},
                "outputs_dtype": {"out": "float32"},
                "params": {"scale": 0.5},
            }],
            "outputs": ["y"],
        })
        self.assertEqual(validate_external_quantization(graph, {}), {})

        for field in forbidden:
            with self.subTest(field=field):
                rejected = copy.deepcopy(graph)
                rejected["nodes"][0]["params"]["private"] = [
                    {"affine": {field: 0.25}},
                ]
                with self.assertRaises(ExporterError) as caught:
                    validate_external_quantization(rejected, {})
                self.assertEqual(
                    caught.exception.diagnostic.code, "VXQSTORE050",
                )
                self.assertIn(
                    f"params.private[0].affine.{field}",
                    caught.exception.diagnostic.message,
                )

        authored = copy.deepcopy(graph)
        authored["nodes"][0]["params"]["hidden"] = {
            "weight_zero_point": 0,
        }
        with self.assertRaises(ExporterError) as caught:
            externalize_quantization(authored, {}, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXQSTORE050")

    def test_externalizer_does_not_migrate_legacy_graphs(self):
        graph, descriptors = self.authoring_graph()
        graph["weights_quantization"] = {"w": descriptors["w"]}
        with self.assertRaises(ExporterError) as caught:
            externalize_quantization(graph, {}, descriptors)
        self.assertEqual(caught.exception.diagnostic.code, "VXQSTORE021")

    def test_validator_rejects_non_v1_graph_format(self):
        graph = _new_v1({
            "format": "volvox-graph/v2",
            "inputs": {"x": {"shape": [1], "dtype": "float32"}},
            "nodes": [],
            "outputs": ["x"],
        })
        with self.assertRaises(ExporterError) as caught:
            validate_external_quantization(graph, {})
        self.assertEqual(caught.exception.diagnostic.code, "VXQSTORE020")

    def test_staged_package_rejects_companion_metadata(self):
        with self.assertRaises(ExporterError) as caught:
            reject_legacy_safetensors_metadata({
                "weights_quantization_storage": "obsolete",
            })
        self.assertEqual(caught.exception.diagnostic.code, "VXPKGWEIGHT002")

    def test_raw_integer_tensors_do_not_require_affine_metadata(self):
        graph = {
            "format": GRAPH_FORMAT,
            "inputs": {"bytes": {"shape": [4], "dtype": "uint8"}},
            "nodes": [],
            "outputs": ["bytes"],
        }
        self.assertEqual(validate_external_quantization(graph, {}), {})

    def test_optimizer_prunes_only_removed_activation_targets(self):
        graph = _new_v1({
            "format": GRAPH_FORMAT,
            "inputs": {"x": {"shape": [1], "dtype": "int8"}},
            "nodes": [{
                "opType": "Identity",
                "inputs": {"input": "x"},
                "outputs": {"out": "temporary"},
                "outputs_shape": {"out": [1]},
                "outputs_dtype": {"out": "int8"},
            }],
            "outputs": ["x"],
        })
        tensors = {}
        descriptor = {
            "scheme": "per_tensor", "scale": 0.25, "zero_point": 0,
        }
        externalize_quantization(
            graph, tensors, {"x": descriptor, "temporary": descriptor}
        )
        graph["nodes"] = []

        self.assertEqual(prune_external_quantization(graph, tensors), 1)
        self.assertEqual(set(graph["quantization"]["tensors"]), {"x"})

    def test_validator_normalizes_framework_dtype_metadata(self):
        graph = {
            "format": GRAPH_FORMAT,
            "inputs": {"x": {"shape": [1], "dtype": "float32"}},
            "nodes": [],
            "outputs": ["x"],
            "quantization": {
                "format": QUANTIZATION_FORMAT,
                "tensors": {
                    "metadata_only": {
                        "scheme": "per_tensor",
                        "scale_tensor": "metadata_only.scale",
                        "zero_point_tensor": "metadata_only.zero_point",
                    }
                },
            },
        }
        tensors = {
            "metadata_only": SimpleNamespace(shape=(3,), dtype="torch.int8"),
            "metadata_only.scale": np.asarray([0.25], dtype=np.float32),
            "metadata_only.zero_point": np.asarray([0], dtype=np.int8),
        }

        self.assertEqual(
            validate_external_quantization(graph, tensors)["metadata_only"].scheme,
            "per_tensor",
        )


if __name__ == "__main__":
    unittest.main()
