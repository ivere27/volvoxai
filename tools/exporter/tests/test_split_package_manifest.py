from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools.exporter.errors import ExporterError
from tools.exporter.optimizer.safetensors_io import write_safetensors
from tools.exporter.split_package_manifest import (
    graph_manifest_inputs,
    refresh_graph_manifest_inputs,
    refresh_split_package_identities,
    validate_split_package,
)


GRAPH_KINDS = ("first", "second")


def _write_package(root: Path) -> None:
    graphs = {}
    for kind in GRAPH_KINDS:
        directory = root / kind
        directory.mkdir(parents=True)
        if kind == "first":
            inputs = {"input0": {"shape": [1, 1], "dtype": "float32"}}
            outputs = ["feature"]
            nodes = [{
                "id": "feature_identity",
                "opType": "Identity",
                "inputs": {"input": "input0"},
                "outputs": {"out": {
                    "tensor": "feature", "shape": [1, 1], "dtype": "float32",
                }},
                "params": {},
            }]
            semantic_inputs = {"sample": "input0"}
            semantic_outputs = {"feature": "feature"}
        else:
            inputs = {"input0": {"shape": [1, 2], "dtype": "float32"}}
            outputs = ["score"]
            nodes = [
                {
                    "id": "keep",
                    "opType": "Identity",
                    "inputs": {"input": "input0"},
                    "outputs": {"out": {
                        "tensor": "kept", "shape": [1, 2], "dtype": "float32",
                    }},
                    "params": {},
                },
                {
                    "id": "score_identity",
                    "opType": "Identity",
                    "inputs": {"input": "kept"},
                    "outputs": {"out": {
                        "tensor": "score", "shape": [1, 2], "dtype": "float32",
                    }},
                    "params": {},
                },
            ]
            semantic_inputs = {"tokens": "input0"}
            semantic_outputs = {"score": "score"}
        document = {
            "format": "volvox-graph/v1",
            "dimensions": {},
            "inputs": inputs,
            "outputs": outputs,
            "nodes": nodes,
        }
        (directory / "graph.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        write_safetensors(directory / "model.safetensors", {})
        graphs[kind] = {
            "graph": {"path": f"{kind}/graph.json"},
            "weights": {"path": f"{kind}/model.safetensors"},
            "inputs": graph_manifest_inputs(document, semantic_inputs),
            "outputs": semantic_outputs,
        }
    manifest = {
        "format": "volvoxai-test-split-package",
        "graphs": graphs,
        "metadata": {},
    }
    refresh_split_package_identities(manifest, root)
    (root / "package_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _dynamic_qlinear_graph() -> tuple[dict, dict[str, np.ndarray]]:
    tensors = {
        "weight": np.asarray([[1, 2], [3, 4]], dtype=np.int8),
        "bias": np.asarray([0, 0], dtype=np.int32),
        "weight.scale": np.asarray([0.25, 0.5], dtype=np.float32),
        "weight.zero": np.asarray([0, 0], dtype=np.int8),
        "x.scale": np.asarray([0.125], dtype=np.float32),
        "x.zero": np.asarray([0], dtype=np.int8),
        "y.scale": np.asarray([0.25], dtype=np.float32),
        "y.zero": np.asarray([0], dtype=np.int8),
    }
    document = {
        "format": "volvox-graph/v1",
        "dimensions": {"B": {"min": 1, "max": 4}},
        "inputs": {"x": {"shape": ["B", 2], "dtype": "int8"}},
        "outputs": ["y"],
        "nodes": [{
            "id": "linear",
            "opType": "QLinear",
            "inputs": {"input": "x", "weight": "weight", "bias": "bias"},
            "outputs": {"out": {
                "tensor": "y", "shape": ["B", 2], "dtype": "int8",
            }},
            "params": {},
        }],
        "quantization": {
            "format": "volvox-affine-safetensors/v1",
            "tensors": {
                "x": {
                    "scheme": "per_tensor",
                    "scale_tensor": "x.scale",
                    "zero_point_tensor": "x.zero",
                },
                "weight": {
                    "scheme": "per_axis",
                    "axis": 0,
                    "scale_tensor": "weight.scale",
                    "zero_point_tensor": "weight.zero",
                },
                "y": {
                    "scheme": "per_tensor",
                    "scale_tensor": "y.scale",
                    "zero_point_tensor": "y.zero",
                },
            },
        },
    }
    return document, tensors


def _write_dynamic_quantized_package(root: Path) -> None:
    _write_package(root)
    manifest_path = root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for kind in GRAPH_KINDS:
        document, tensors = _dynamic_qlinear_graph()
        (root / kind / "graph.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        write_safetensors(root / kind / "model.safetensors", tensors)
        manifest["graphs"][kind]["inputs"] = {"value": "x"}
        manifest["graphs"][kind]["outputs"] = {"result": "y"}
    refresh_split_package_identities(manifest, root)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _add_export_reports(root: Path) -> None:
    manifest_path = root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for kind in GRAPH_KINDS:
        path = root / kind / "export_report.json"
        path.write_text(json.dumps({"graph_kind": kind}), encoding="utf-8")
        payload = path.read_bytes()
        manifest["graphs"][kind]["export_report"] = {
            "path": f"{kind}/export_report.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class GraphManifestInputsTest(unittest.TestCase):
    def test_preserves_semantics_and_adds_new_tensor_identity(self):
        manifest = {
            "graphs": {
                "decode": {
                    "inputs": {"tokens": "input0", "stale": "removed"},
                },
            },
        }
        document = {
            "inputs": {
                "input0": {"shape": [1, 192], "dtype": "int32"},
                "state": {"shape": [1, 192], "dtype": "int32"},
            },
        }

        refresh_graph_manifest_inputs(manifest, "decode", document)

        self.assertEqual(
            manifest["graphs"]["decode"]["inputs"],
            {"tokens": "input0", "state": "state"},
        )

    def test_explicit_mapping_rejects_duplicate_tensor_owners(self):
        document = {
            "inputs": {
                "input0": {"shape": [1], "dtype": "float32"},
                "input1": {"shape": [1], "dtype": "float32"},
            },
        }

        with self.assertRaisesRegex(ValueError, "share graph tensor 'input0'"):
            graph_manifest_inputs(
                document, {"sample": "input0", "alias": "input0"}
            )

    def test_invalid_input_descriptors_fail_closed(self):
        cases = (
            (None, None),
            ({"inputs": None}, None),
            ({"inputs": []}, []),
            ({"inputs": {"input0": None}}, {"input0": None}),
            ({"inputs": {"": {}}}, {"": {}}),
        )
        for document, label in cases:
            with self.subTest(inputs=label):
                with self.assertRaises(ValueError):
                    graph_manifest_inputs(document)


class SplitPackageManifestTest(unittest.TestCase):
    def test_validation_accepts_present_or_absent_export_reports(self):
        for reports_present in (False, True):
            with (
                self.subTest(reports_present=reports_present),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory) / "package"
                _write_package(root)
                if reports_present:
                    _add_export_reports(root)

                validate_split_package(
                    root,
                    graph_validator=lambda document, weights: None,
                )

    def test_output_mapping_is_order_independent_and_one_to_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root)
            graph_path = root / "second" / "graph.json"
            document = json.loads(graph_path.read_text(encoding="utf-8"))
            document["outputs"] = ["score", "kept"]
            graph_path.write_text(json.dumps(document), encoding="utf-8")
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["graphs"]["second"]["outputs"] = {
                "retained": "kept",
                "prediction": "score",
            }
            refresh_split_package_identities(manifest, root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            validate_split_package(
                root,
                graph_validator=lambda graph, weights: None,
            )

            manifest["graphs"]["second"]["outputs"] = {
                "first": "score",
                "second": "score",
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match graph outputs"):
                validate_split_package(
                    root,
                    graph_validator=lambda graph, weights: None,
                )

    def test_validation_rejects_untrusted_export_reports(self):
        cases = (
            ("missing", "missing"),
            ("symlink", "regular non-symlink"),
            ("tampered", "digest mismatch"),
            ("wrong-size", "size mismatch"),
            ("uppercase-digest", "lowercase SHA-256"),
            ("traversal", "path must be first/export_report.json"),
        )
        for case, pattern in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "package"
                _write_package(root)
                _add_export_reports(root)
                manifest_path = root / "package_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                report_path = root / "first" / "export_report.json"
                report_entry = manifest["graphs"]["first"]["export_report"]

                if case == "missing":
                    report_path.unlink()
                elif case == "symlink":
                    payload = report_path.read_bytes()
                    outside = Path(directory) / "outside-export-report.json"
                    outside.write_bytes(payload)
                    report_path.unlink()
                    report_path.symlink_to(outside)
                elif case == "tampered":
                    payload = bytearray(report_path.read_bytes())
                    payload[-1] ^= 1
                    report_path.write_bytes(payload)
                elif case == "wrong-size":
                    report_entry["bytes"] += 1
                elif case == "uppercase-digest":
                    report_entry["sha256"] = report_entry["sha256"].upper()
                else:
                    report_entry["path"] = "../outside-export-report.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, pattern):
                    validate_split_package(
                        root,
                        graph_validator=lambda document, weights: None,
                    )

    def test_validation_rejects_implicit_output_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root)
            graph_path = root / "first" / "graph.json"
            document = json.loads(graph_path.read_text(encoding="utf-8"))
            document["nodes"][0]["outputs"]["out"].pop("dtype")
            graph_path.write_text(json.dumps(document), encoding="utf-8")
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refresh_split_package_identities(manifest, root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ExporterError) as caught:
                validate_split_package(root)
            self.assertEqual(caught.exception.diagnostic.code, "VXRTIR020")

    def test_validation_forwards_canonical_dynamic_quantized_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root)
            proof = object()

            with mock.patch(
                "tools.exporter.split_package_manifest."
                "prove_dynamic_quantized_runtime_domain",
                return_value=proof,
            ) as prove, mock.patch(
                "tools.exporter.split_package_manifest.import_runtime_package",
            ) as import_package:
                validate_split_package(root)

            self.assertEqual(prove.call_count, 2)
            self.assertEqual(import_package.call_count, 2)
            for call in import_package.call_args_list:
                self.assertIs(call.kwargs["bounded_domain_proof"], proof)

    def test_validation_accepts_real_dynamic_quantized_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_dynamic_quantized_package(root)

            validate_split_package(root)

    def test_validation_rejects_nonportable_dynamic_quantized_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_dynamic_quantized_package(root)
            graph_path = root / "first" / "graph.json"
            document = json.loads(graph_path.read_text(encoding="utf-8"))
            document["inputs"]["indices"] = {
                "shape": ["B", 2],
                "dtype": "int32",
            }
            document["nodes"] = [{
                "id": "gather",
                "opType": "GatherElements",
                "inputs": {"input": "x", "indices": "indices"},
                "outputs": {"out": {
                    "tensor": "y", "shape": ["B", 2], "dtype": "int8",
                }},
                "params": {"axis": 1},
            }]
            document["quantization"]["tensors"] = {
                name: descriptor
                for name, descriptor in document["quantization"]["tensors"].items()
                if name in {"x", "y"}
            }
            graph_path.write_text(json.dumps(document), encoding="utf-8")
            write_safetensors(
                root / "first" / "model.safetensors",
                {
                    "x.scale": np.asarray([0.125], dtype=np.float32),
                    "x.zero": np.asarray([0], dtype=np.int8),
                    "y.scale": np.asarray([0.25], dtype=np.float32),
                    "y.zero": np.asarray([0], dtype=np.int8),
                },
            )
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["graphs"]["first"]["inputs"] = {
                "value": "x",
                "indices": "indices",
            }
            refresh_split_package_identities(
                manifest, root, graph_kinds=("first",)
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ExporterError) as caught:
                validate_split_package(root, graph_kinds=("first",))

            self.assertEqual(caught.exception.diagnostic.code, "VXRTIR037")
            self.assertIn("native-cpu", caught.exception.diagnostic.message)

    def test_graphs_use_the_strict_runtime_document_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root)
            graph_path = root / "first" / "graph.json"
            document = json.loads(graph_path.read_text(encoding="utf-8"))
            encoded = json.dumps(document)
            graph_path.write_text(
                encoded[:-1] + ',"format":"volvox-graph/v1"}',
                encoding="utf-8",
            )
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refresh_split_package_identities(manifest, root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ExporterError) as caught:
                validate_split_package(
                    root,
                    graph_validator=lambda document, weights: None,
                )
            self.assertEqual(caught.exception.diagnostic.code, "VXRTIR028")

    def test_custom_validation_receives_graph_scoped_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root)
            write_safetensors(
                root / "first" / "model.safetensors",
                {"first_only": np.asarray([1.0], dtype=np.float32)},
            )
            write_safetensors(
                root / "second" / "model.safetensors",
                {"second_only": np.asarray([2.0], dtype=np.float32)},
            )
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refresh_split_package_identities(manifest, root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            observed: dict[str, set[str]] = {}

            def validate(document, weights):
                kind = "first" if document["outputs"] == ["feature"] else "second"
                observed[kind] = set(weights)

            validate_split_package(root, graph_validator=validate)

            self.assertEqual(observed, {
                "first": {"first_only"},
                "second": {"second_only"},
            })

    def test_validation_rejects_unsafe_manifest_integer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root)
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"]["unsafe"] = 1 << 53
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON's safe range"):
                validate_split_package(root)

    def test_validation_inherits_strict_safetensors_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root)
            weights_path = root / "second" / "model.safetensors"
            weights_path.write_bytes(weights_path.read_bytes() + b"stowaway")
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refresh_split_package_identities(manifest, root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "full data payload"):
                validate_split_package(root)


if __name__ == "__main__":
    unittest.main()
