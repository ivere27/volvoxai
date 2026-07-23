from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from examples.tiny_receipt_vqa.tools.optimize_split_fp32 import optimize_package
from examples.tiny_receipt_vqa.tools.package_manifest import (
    CALIBRATION_PROVENANCE_FORMAT,
    calibration_provenance,
    graph_manifest_inputs,
    refresh_graph_manifest_inputs,
    refresh_split_package_identities,
    split_package_routing,
    validate_split_package,
    verify_calibration_source_package,
)
from examples.tiny_receipt_vqa.tools.quantize_split_package import quantize_package
from tools.exporter.errors import ExporterError
from tools.exporter.optimizer.safetensors_io import write_safetensors


FAMILY_NAMES = (
    "phone",
    "address",
    "store",
    "item_row",
    "item_math",
    "item_lookup",
    "math",
    "other",
)


def _write_package(root: Path, *, specialized_graphs: bool) -> None:
    graphs = {}
    for kind in ("encoder", "decoder"):
        directory = root / kind
        directory.mkdir(parents=True)
        if kind == "encoder":
            inputs = {
                "input0": {
                    "shape": [1, 1], "dtype": "float32", "source_name": "image",
                },
            }
            if not specialized_graphs:
                inputs["input2"] = {
                    "shape": [1], "dtype": "int32", "source_name": "family_ids",
                }
        else:
            inputs = {
                "input0": {
                    "shape": [1, 2],
                    "dtype": "float32",
                    "source_name": "decoder_input_ids",
                },
            }
            if specialized_graphs:
                inputs["v4_keep"] = {"shape": [1, 2], "dtype": "float32"}
            else:
                inputs["input3"] = {
                    "shape": [1], "dtype": "int32", "source_name": "family_ids",
                }
        nodes = []
        if kind == "encoder":
            nodes.append({
                "opType": "Identity",
                "inputs": {"input": "input0"},
                "outputs": {"out": "memory"},
                "outputs_shape": {"out": [1, 1]},
                "outputs_dtype": {"out": "float32"},
            })
        else:
            if not specialized_graphs:
                nodes.append({
                    "opType": "Identity",
                    "source_name": "/decoder/Equal_17",
                    "inputs": {"input": "input0"},
                    "outputs": {"out": "v4_keep"},
                    "outputs_shape": {"out": [1, 2]},
                    "outputs_dtype": {"out": "float32"},
                })
            nodes.append({
                "opType": "Identity",
                "inputs": {"input": "v4_keep"},
                "outputs": {"out": "logits"},
                "outputs_shape": {"out": [1, 2]},
                "outputs_dtype": {"out": "float32"},
            })
        document = {
            "format": "volvox-graph/v1",
            "inputs": inputs,
            "outputs": ["logits"] if kind == "decoder" else ["memory"],
            "nodes": nodes,
            "source": {},
        }
        (directory / "graph.json").write_text(json.dumps(document), encoding="utf-8")
        write_safetensors(directory / "model.safetensors", {})
        graphs[kind] = {
            "graph": {"path": f"{kind}/graph.json"},
            "weights": {"path": f"{kind}/model.safetensors"},
            "inputs": graph_manifest_inputs(document),
            "outputs": (
                {"logits": "logits"}
                if kind == "decoder"
                else {"memory": "memory"}
            ),
        }
    manifest = {
        "format": "volvoxai-test-split-v1",
        "routing": (
            {"mode": "specialized", "family_id": 0}
            if specialized_graphs
            else {
                "mode": "runtime",
                "family_inputs": {
                    "encoder": "input2",
                    "decoder": "input3",
                },
            }
        ),
        "families": {
            "ordered_names": list(FAMILY_NAMES),
            "name_to_id": {
                family: index for index, family in enumerate(FAMILY_NAMES)
            },
        },
        "graphs": graphs,
        "generation": {},
        "variant": {"compiled_graph_package_class": {}},
    }
    (root / "package_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _add_export_reports(root: Path) -> None:
    manifest_path = root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for kind in ("encoder", "decoder"):
        path = root / kind / "export_report.json"
        path.write_text(json.dumps({"graph_kind": kind}), encoding="utf-8")
        payload = path.read_bytes()
        manifest["graphs"][kind]["export_report"] = {
            "path": f"{kind}/export_report.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _calibration_document(source: Path | None = None) -> dict:
    digest = "a" * 64
    document = {
        "provenance": {
            "calibration_manifest": {
                "path": "/private/calibration-records.json",
                "sha256": digest,
            },
            "fixtures": {
                "files": {
                    "image.f32": {
                        "path": "fixtures/image.f32",
                        "sha256": digest,
                        "bytes": 64,
                    }
                }
            },
            "selected_records": {"count": 2, "ids": ["receipt-a", "receipt-b"]},
            "family_counts": {
                "required": {"phone": 1},
                "record": {"phone": 2},
            },
            "route_executions": {
                "mode": "graph-routing",
                "count": 2,
                "family_counts": {"phone": 2},
            },
            "settings": {
                "samples": 2,
                "prefixes_per_record": 3,
                "backend": "wasm",
                "batch": 512,
                "graphs": ["encoder", "decoder"],
                "routing_mode": "specialized",
            },
            "package": {
                "path": "/private/fp32-package",
                "graphs": {
                    kind: {
                        "graph": {
                            "path": f"/private/{kind}/graph.json",
                            "sha256": digest,
                        },
                        "weights": {
                            "path": f"/private/{kind}/model.safetensors",
                            "sha256": digest,
                        },
                    }
                    for kind in ("encoder", "decoder")
                },
            },
            "calibrator_source": {
                "path": "/private/calibrate.mjs",
                "sha256": digest,
            },
            "affine_exclusions": {
                "decoder": {
                    "semantic_domain": "additive-attention-mask/v1",
                    "count": 1,
                    "tensors": ["mask"],
                    "reasons": {"mask": "zero-negative-infinity-where-mask"},
                },
            },
            # Unknown fields are never copied by the whitelist.
            "unreviewed": {"path": "/private/leak", "scale": 0.125},
        },
        # Activation ranges drive PTQ but must never enter package metadata.
        "encoder": {"activation": {"min": -1.25, "max": 2.5}},
        "decoder": {"logits": {"min": -8.0, "max": 6.0}},
    }

    if source is not None:
        document["provenance"]["package"] = {
            "path": str(source.resolve()),
            "graphs": {
                kind: {
                    asset: {
                        "path": str((source / kind / filename).resolve()),
                        "sha256": hashlib.sha256(
                            (source / kind / filename).read_bytes()
                        ).hexdigest(),
                        "bytes": (source / kind / filename).stat().st_size,
                    }
                    for asset, filename in (
                        ("graph", "graph.json"),
                        ("weights", "model.safetensors"),
                    )
                }
                for kind in ("encoder", "decoder")
            },
        }
    return document


class CalibrationProvenanceTest(unittest.TestCase):
    def test_projects_identity_and_only_curated_non_affine_provenance(self):
        document = _calibration_document()
        payload = json.dumps(document, separators=(",", ":")).encode("utf-8")

        projected = calibration_provenance(document, payload)

        self.assertEqual(projected["format"], CALIBRATION_PROVENANCE_FORMAT)
        self.assertEqual(
            projected["artifact"],
            {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)},
        )
        self.assertEqual(
            projected["calibration_manifest"], {"sha256": "a" * 64}
        )
        self.assertEqual(
            projected["selected_records"],
            {"count": 2, "ids": ["receipt-a", "receipt-b"]},
        )
        self.assertEqual(projected["settings"]["prefixes_per_record"], 3)
        self.assertEqual(projected["selected_records"]["count"], 2)
        self.assertEqual(projected["route_executions"], {
            "mode": "graph-routing",
            "count": 2,
            "family_counts": {"phone": 2},
        })
        self.assertEqual(
            projected["package"]["graphs"]["encoder"]["graph"],
            {"sha256": "a" * 64},
        )
        self.assertEqual(projected["calibrator_source"], {"sha256": "a" * 64})

        serialized = json.dumps(projected)
        self.assertNotIn("/private/", serialized)
        self.assertNotIn("unreviewed", projected)

        def keys(value):
            if isinstance(value, dict):
                for name, child in value.items():
                    yield name
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        self.assertTrue({"min", "max", "scale", "zero_point"}.isdisjoint(keys(projected)))

    def test_file_identity_is_present_without_producer_provenance(self):
        payload = b"{}"
        self.assertEqual(
            calibration_provenance({}, payload),
            {
                "format": CALIBRATION_PROVENANCE_FORMAT,
                "artifact": {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": 2,
                },
            },
        )

    def test_malformed_whitelisted_provenance_fails_closed(self):
        base = _calibration_document()
        cases = []

        invalid = copy.deepcopy(base)
        invalid["provenance"] = []
        cases.append(invalid)
        invalid = copy.deepcopy(base)
        invalid["provenance"]["calibration_manifest"]["sha256"] = "not-a-hash"
        cases.append(invalid)
        invalid = copy.deepcopy(base)
        invalid["provenance"]["selected_records"]["count"] = 3
        cases.append(invalid)
        invalid = copy.deepcopy(base)
        invalid["provenance"]["family_counts"]["record"]["phone"] = True
        cases.append(invalid)
        invalid = copy.deepcopy(base)
        invalid["provenance"]["route_executions"]["count"] = 3
        cases.append(invalid)
        invalid = copy.deepcopy(base)
        invalid["provenance"]["settings"]["samples"] = 1
        cases.append(invalid)
        invalid = copy.deepcopy(base)
        invalid["provenance"]["settings"]["routing_mode"] = "legacy"
        cases.append(invalid)
        invalid = copy.deepcopy(base)
        del invalid["provenance"]["package"]["graphs"]["encoder"]["weights"]
        cases.append(invalid)
        invalid = copy.deepcopy(base)
        invalid["provenance"]["affine_exclusions"]["decoder"]["reasons"] = {}
        cases.append(invalid)

        for index, document in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                calibration_provenance(document, b"{}")

    def test_verifies_exact_graph_and_weight_bytes_of_source_package(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            _write_package(source, specialized_graphs=True)
            document = _calibration_document(source)
            payload = json.dumps(document).encode("utf-8")
            provenance = calibration_provenance(document, payload)

            verify_calibration_source_package(provenance, source)

            graph_path = source / "encoder" / "graph.json"
            graph_path.write_bytes(graph_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                ValueError, "identity mismatch for encoder graph"
            ):
                verify_calibration_source_package(provenance, source)


class GraphManifestInputsTest(unittest.TestCase):
    def test_replaces_stale_entries_and_uses_tensor_name_without_source_name(self):
        manifest = {
            "graphs": {
                "decoder": {
                    "inputs": {"family_ids": "input3", "stale": "removed"},
                },
            },
        }
        document = {
            "inputs": {
                "input0": {"source_name": "decoder_input_ids"},
                "v4_keep": {"shape": [1, 192], "dtype": "int32"},
            },
        }

        refresh_graph_manifest_inputs(manifest, "decoder", document)

        self.assertEqual(
            manifest["graphs"]["decoder"]["inputs"],
            {"decoder_input_ids": "input0", "v4_keep": "v4_keep"},
        )

    def test_duplicate_semantics_fail_without_mutating_manifest(self):
        manifest = {"graphs": {"encoder": {"inputs": {"old": "input9"}}}}
        before = json.loads(json.dumps(manifest))
        document = {
            "inputs": {
                "input0": {"source_name": "image"},
                "input1": {"source_name": "image"},
            },
        }

        with self.assertRaisesRegex(ValueError, "share semantic name 'image'"):
            refresh_graph_manifest_inputs(manifest, "encoder", document)

        self.assertEqual(manifest, before)

    def test_invalid_input_descriptors_fail_closed(self):
        cases = (
            (None, None),
            ({"inputs": None}, None),
            ({"inputs": []}, []),
            ({"inputs": {"input0": None}}, {"input0": None}),
            (
                {"inputs": {"input0": {"source_name": None}}},
                {"input0": {"source_name": None}},
            ),
            (
                {"inputs": {"input0": {"source_name": ""}}},
                {"input0": {"source_name": ""}},
            ),
            ({"inputs": {"": {}}}, {"": {}}),
        )
        for document, label in cases:
            with self.subTest(inputs=label):
                with self.assertRaises(ValueError):
                    graph_manifest_inputs(document)


class SplitPackageManifestRefreshTest(unittest.TestCase):
    def test_routing_contract_is_explicit_and_matches_graph_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            specialized = root / "specialized"
            _write_package(runtime, specialized_graphs=False)
            _write_package(specialized, specialized_graphs=True)

            runtime_manifest = json.loads(
                (runtime / "package_manifest.json").read_text(encoding="utf-8")
            )
            specialized_manifest = json.loads(
                (specialized / "package_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                split_package_routing(runtime_manifest, required=True),
                {
                    "mode": "runtime",
                    "family_inputs": {
                        "encoder": "input2",
                        "decoder": "input3",
                    },
                },
            )
            self.assertEqual(
                split_package_routing(specialized_manifest, required=True),
                {"mode": "specialized", "family_id": 0},
            )

    def test_required_routing_rejects_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root, specialized_graphs=False)
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["routing"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires explicit routing"):
                validate_split_package(root, require_routing=True)

            # Generic tooling keeps a deliberate read-compatibility mode for
            # old packages; session-ready FP32/PTQ publication opts into the
            # stricter contract above.
            refresh_split_package_identities(manifest, root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            validate_split_package(
                root,
                graph_validator=lambda document, weights: None,
            )

    def test_routing_rejects_mode_and_physical_input_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root, specialized_graphs=False)
            manifest = json.loads(
                (root / "package_manifest.json").read_text(encoding="utf-8")
            )

            wrong_input = copy.deepcopy(manifest)
            wrong_input["routing"]["family_inputs"]["encoder"] = "family_ids"
            with self.assertRaisesRegex(ValueError, "does not match graph inputs"):
                split_package_routing(wrong_input, required=True)

            wrong_mode = copy.deepcopy(manifest)
            wrong_mode["routing"] = {"mode": "specialized", "family_id": 0}
            with self.assertRaisesRegex(ValueError, "forbids encoder family_ids"):
                split_package_routing(wrong_mode, required=True)

    def test_package_validation_rejects_implicit_output_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root, specialized_graphs=False)
            graph_path = root / "encoder" / "graph.json"
            document = json.loads(graph_path.read_text(encoding="utf-8"))
            document["nodes"][0].pop("outputs_dtype")
            graph_path.write_text(json.dumps(document), encoding="utf-8")
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refresh_split_package_identities(manifest, root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ExporterError) as caught:
                validate_split_package(root)
            self.assertEqual(caught.exception.diagnostic.code, "VXRTIR011")

    def test_package_graphs_use_the_strict_runtime_document_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root, specialized_graphs=False)
            graph_path = root / "encoder" / "graph.json"
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

    def test_custom_validation_receives_documents_and_graph_scoped_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root, specialized_graphs=False)
            write_safetensors(
                root / "encoder" / "model.safetensors",
                {"encoder_only": np.asarray([1.0], dtype=np.float32)},
            )
            write_safetensors(
                root / "decoder" / "model.safetensors",
                {"decoder_only": np.asarray([2.0], dtype=np.float32)},
            )
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refresh_split_package_identities(manifest, root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            observed: dict[str, set[str]] = {}

            def validate(document, weights):
                self.assertIsInstance(document, dict)
                kind = "encoder" if document["outputs"] == ["memory"] else "decoder"
                observed[kind] = set(weights)

            validate_split_package(root, graph_validator=validate)

            self.assertEqual(observed, {
                "encoder": {"encoder_only"},
                "decoder": {"decoder_only"},
            })

    def test_package_validation_rejects_unsafe_manifest_integer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root, specialized_graphs=False)
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["generation"]["unsafe"] = 1 << 53
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON's safe range"):
                validate_split_package(root)

    def test_package_validation_inherits_strict_safetensors_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            _write_package(root, specialized_graphs=False)
            weights_path = root / "decoder" / "model.safetensors"
            weights_path.write_bytes(weights_path.read_bytes() + b"stowaway")
            manifest_path = root / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refresh_split_package_identities(manifest, root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "full data payload"):
                validate_split_package(root)

    def test_fp32_optimizer_publishes_frozen_and_hoisted_input_abi(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            out = Path(directory) / "out"
            _write_package(source, specialized_graphs=False)

            optimize_package(
                source,
                out,
                attention_fusion=False,
                freeze={"family_ids": np.asarray([3], dtype=np.int32)},
                # The removed producer and remaining Identity are both F32;
                # hoisting must preserve that execution dtype.
                hoist={"v4_keep": "float32"},
            )

            manifest = json.loads((out / "package_manifest.json").read_text())
            documents = {
                kind: json.loads((out / kind / "graph.json").read_text())
                for kind in ("encoder", "decoder")
            }
            self.assertEqual(
                manifest["graphs"]["encoder"]["inputs"], {"image": "input0"}
            )
            self.assertEqual(
                manifest["graphs"]["decoder"]["inputs"],
                {"decoder_input_ids": "input0", "v4_keep": "v4_keep"},
            )
            self.assertEqual(
                manifest["routing"],
                {"mode": "specialized", "family_id": 3},
            )
            for kind, document in documents.items():
                self.assertEqual(
                    manifest["graphs"][kind]["inputs"],
                    graph_manifest_inputs(document),
                )
            self.assertNotIn(
                "source_name", documents["decoder"]["inputs"]["v4_keep"]
            )

    def test_fp32_optimizer_publishes_runtime_routing_without_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            out = Path(directory) / "out"
            _write_package(source, specialized_graphs=False)

            optimize_package(
                source,
                out,
                attention_fusion=False,
                hoist={"v4_keep": "float32"},
            )

            manifest = json.loads((out / "package_manifest.json").read_text())
            self.assertEqual(
                manifest["routing"],
                {
                    "mode": "runtime",
                    "family_inputs": {
                        "encoder": "input2",
                        "decoder": "input3",
                    },
                },
            )
            self.assertEqual(
                manifest["graphs"]["encoder"]["inputs"],
                {"image": "input0", "family_ids": "input2"},
            )
            self.assertEqual(
                manifest["graphs"]["decoder"]["inputs"],
                {
                    "decoder_input_ids": "input0",
                    "family_ids": "input3",
                    "v4_keep": "v4_keep",
                },
            )

    def test_runtime_ptq_publishes_partial_route_coverage_as_qualification_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            out = Path(directory) / "out"
            calibration = Path(directory) / "calibration.json"
            _write_package(source, specialized_graphs=False)
            source_manifest_before = (source / "package_manifest.json").read_bytes()
            document = _calibration_document(source)
            document["provenance"]["settings"]["routing_mode"] = "explicit"
            # Tensor calibration is complete for this fixture, while route
            # coverage remains deliberately partial application evidence.
            document["provenance"]["route_executions"] = {
                "mode": "graph-routing",
                "count": 2,
                "family_counts": {"phone": 2},
            }
            calibration.write_text(json.dumps(document), encoding="utf-8")

            def specialize_output(graph_document, weights, **_):
                graph_document = json.loads(json.dumps(graph_document))
                graph_document["nodes"].append({
                    "opType": "ArgMax",
                    "inputs": {"input": "logits"},
                    "outputs": {"out": "token_ids"},
                    "outputs_shape": {"out": [1]},
                    "outputs_dtype": {"out": "int32"},
                    "params": {"axis": -1, "keepdims": False},
                })
                graph_document["outputs"] = ["token_ids"]
                return graph_document, weights, SimpleNamespace(total_changes=1)

            with (
                mock.patch(
                    "examples.tiny_receipt_vqa.tools.quantize_split_package.quantize_graph",
                    return_value={},
                ) as quantize_graph_mock,
                mock.patch(
                    "examples.tiny_receipt_vqa.tools.quantize_split_package.optimize_runtime_package",
                    side_effect=specialize_output,
                ),
                mock.patch(
                    "examples.tiny_receipt_vqa.tools.quantize_split_package._refresh_package_class",
                    return_value="hybrid",
                ),
            ):
                quantize_package(source, out, calibration)

            self.assertEqual(
                (source / "package_manifest.json").read_bytes(),
                source_manifest_before,
            )
            manifest = json.loads((out / "package_manifest.json").read_text())
            self.assertEqual(manifest["routing"], {
                "mode": "runtime",
                "family_inputs": {"encoder": "input2", "decoder": "input3"},
            })
            self.assertEqual(manifest["families"]["ordered_names"], list(FAMILY_NAMES))
            qualification = manifest["calibration_routing_qualification"]
            self.assertFalse(qualification["coverage_complete"])
            self.assertEqual(qualification["execution_count"], 2)
            self.assertEqual(qualification["catalog_family_counts"], {
                family: 2 if family == "phone" else 0 for family in FAMILY_NAMES
            })
            self.assertEqual(
                qualification["unobserved_required_families"],
                list(FAMILY_NAMES[1:]),
            )
            self.assertEqual(
                [call.kwargs["sample_count"] for call in quantize_graph_mock.call_args_list],
                [2, 6],
            )

    def test_runtime_ptq_accepts_complete_explicit_route_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            out = Path(directory) / "out"
            calibration = Path(directory) / "calibration.json"
            _write_package(source, specialized_graphs=False)
            document = _calibration_document(source)
            selected_ids = [f"receipt-{index}" for index in range(len(FAMILY_NAMES))]
            document["provenance"]["selected_records"] = {
                "count": len(selected_ids),
                "ids": selected_ids,
            }
            document["provenance"]["settings"].update({
                "samples": len(selected_ids),
                "routing_mode": "explicit",
            })
            complete_counts = {family: 1 for family in FAMILY_NAMES}
            document["provenance"]["family_counts"] = {
                "required": dict(complete_counts),
                "record": dict(complete_counts),
            }
            document["provenance"]["route_executions"] = {
                "mode": "all-public",
                "count": len(selected_ids) * len(FAMILY_NAMES),
                "family_counts": {
                    family: len(selected_ids) for family in FAMILY_NAMES
                },
            }
            calibration.write_text(json.dumps(document), encoding="utf-8")
            calibration_bytes = calibration.read_bytes()
            expected_provenance = calibration_provenance(
                json.loads(calibration_bytes), calibration_bytes
            )

            def specialize_output(graph_document, weights, **_):
                graph_document = json.loads(json.dumps(graph_document))
                graph_document["nodes"].append({
                    "opType": "ArgMax",
                    "inputs": {"input": "logits"},
                    "outputs": {"out": "token_ids"},
                    "outputs_shape": {"out": [1]},
                    "outputs_dtype": {"out": "int32"},
                    "params": {"axis": -1, "keepdims": False},
                })
                graph_document["outputs"] = ["token_ids"]
                return graph_document, weights, SimpleNamespace(total_changes=1)

            with (
                mock.patch(
                    "examples.tiny_receipt_vqa.tools.quantize_split_package.quantize_graph",
                    return_value={},
                ) as quantize_graph_mock,
                mock.patch(
                    "examples.tiny_receipt_vqa.tools.quantize_split_package.optimize_runtime_package",
                    side_effect=specialize_output,
                ),
                mock.patch(
                    "examples.tiny_receipt_vqa.tools.quantize_split_package._refresh_package_class",
                    return_value="hybrid",
                ),
            ):
                quantize_package(source, out, calibration)

            self.assertEqual(quantize_graph_mock.call_count, 2)
            self.assertEqual(
                [call.kwargs["sample_count"] for call in quantize_graph_mock.call_args_list],
                [
                    len(selected_ids) * len(FAMILY_NAMES),
                    len(selected_ids) * len(FAMILY_NAMES) * 3,
                ],
            )
            manifest = json.loads((out / "package_manifest.json").read_text())
            self.assertEqual(manifest["routing"]["mode"], "runtime")
            self.assertEqual(
                manifest["calibration_provenance"],
                expected_provenance,
            )
            self.assertEqual(
                manifest["calibration_provenance"]["settings"]["routing_mode"],
                "explicit",
            )
            self.assertEqual(
                set(manifest["calibration_provenance"]["route_executions"]["family_counts"]),
                set(FAMILY_NAMES),
            )

    def test_quantizer_keeps_current_inputs_and_decoder_output_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            out = Path(directory) / "out"
            calibration = Path(directory) / "calibration.json"
            _write_package(source, specialized_graphs=True)
            _add_export_reports(source)
            calibration.write_text(
                json.dumps(_calibration_document(source)), encoding="utf-8"
            )
            calibration_bytes = calibration.read_bytes()
            expected_provenance = calibration_provenance(
                json.loads(calibration_bytes), calibration_bytes
            )

            def specialize_output(document, weights, **_):
                document = json.loads(json.dumps(document))
                document["nodes"].append({
                    "opType": "ArgMax",
                    "inputs": {"input": "logits"},
                    "outputs": {"out": "token_ids"},
                    "outputs_shape": {"out": [1]},
                    "outputs_dtype": {"out": "int32"},
                    "params": {"axis": -1, "keepdims": False},
                })
                document["outputs"] = ["token_ids"]
                return document, weights, SimpleNamespace(total_changes=1)

            with (
                mock.patch(
                    "examples.tiny_receipt_vqa.tools.quantize_split_package.quantize_graph",
                    return_value={},
                ) as quantize_graph_mock,
                mock.patch(
                    "examples.tiny_receipt_vqa.tools.quantize_split_package.optimize_runtime_package",
                    side_effect=specialize_output,
                ),
                mock.patch(
                    "examples.tiny_receipt_vqa.tools.quantize_split_package._refresh_package_class",
                    return_value="hybrid",
                ),
            ):
                quantize_package(
                    source,
                    out,
                    calibration,
                    float_ops=frozenset({"LayerNorm"}),
                    graph_float_ops={
                        "decoder": frozenset({"CrossSDPA"}),
                    },
                    activation_scheme="asymmetric",
                )

            manifest = json.loads((out / "package_manifest.json").read_text())
            documents = {
                kind: json.loads((out / kind / "graph.json").read_text())
                for kind in ("encoder", "decoder")
            }
            self.assertEqual(
                manifest["graphs"]["encoder"]["inputs"], {"image": "input0"}
            )
            self.assertEqual(
                manifest["graphs"]["decoder"]["inputs"],
                {"decoder_input_ids": "input0", "v4_keep": "v4_keep"},
            )
            self.assertEqual(
                manifest["graphs"]["decoder"]["outputs"],
                {"token_ids": "token_ids"},
            )
            self.assertEqual(
                manifest["routing"],
                {"mode": "specialized", "family_id": 0},
            )
            self.assertEqual(
                manifest["calibration_provenance"], expected_provenance
            )
            self.assertEqual(
                manifest["variant"]["precision_policy"],
                {
                    "format": "volvox-mixed-precision-policy/v1",
                    "activation_dtype": "int8",
                    "activation_scheme": "asymmetric",
                    "float_ops": {
                        "encoder": ["LayerNorm"],
                        "decoder": ["CrossSDPA", "LayerNorm"],
                    },
                },
            )
            self.assertEqual(
                quantize_graph_mock.call_args_list[0].kwargs["float_ops"],
                frozenset({"LayerNorm"}),
            )
            self.assertEqual(
                quantize_graph_mock.call_args_list[1].kwargs["float_ops"],
                frozenset({"CrossSDPA", "LayerNorm"}),
            )
            self.assertEqual(
                quantize_graph_mock.call_args_list[0].kwargs["activation_scheme"],
                "asymmetric",
            )
            self.assertEqual(
                quantize_graph_mock.call_args_list[1].kwargs["activation_scheme"],
                "asymmetric",
            )
            for kind, document in documents.items():
                self.assertEqual(
                    manifest["graphs"][kind]["inputs"],
                    graph_manifest_inputs(document),
                )
                self.assertNotIn("calibration_provenance", document)
                report_path = out / kind / "export_report.json"
                report_bytes = report_path.read_bytes()
                report = json.loads(report_bytes)
                self.assertEqual(
                    report["calibration_provenance"], expected_provenance
                )
                self.assertEqual(
                    manifest["graphs"][kind]["export_report"]["sha256"],
                    hashlib.sha256(report_bytes).hexdigest(),
                )
                self.assertEqual(
                    manifest["graphs"][kind]["export_report"]["bytes"],
                    len(report_bytes),
                )

    def test_quantizer_rejects_legacy_manifest_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            out = Path(directory) / "out"
            calibration = Path(directory) / "calibration.json"
            _write_package(source, specialized_graphs=True)
            manifest_path = source / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["routing"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            calibration.write_text(
                json.dumps(_calibration_document(source)), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "requires explicit routing"):
                quantize_package(source, out, calibration)

            self.assertFalse(out.exists())

    def test_malformed_provenance_is_rejected_before_copying_package(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            out = Path(directory) / "out"
            calibration = Path(directory) / "calibration.json"
            _write_package(source, specialized_graphs=True)
            calibration.write_text(
                json.dumps({"provenance": {"selected_records": []}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "selected_records"):
                quantize_package(source, out, calibration)

            self.assertFalse(out.exists())

    def test_nonfinite_calibration_json_is_rejected_before_copying_package(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            out = Path(directory) / "out"
            calibration = Path(directory) / "calibration.json"
            _write_package(source, specialized_graphs=True)
            document = _calibration_document(source)
            document["encoder"] = {"activation": {"min": float("nan"), "max": 1.0}}
            calibration.write_text(
                json.dumps(document, allow_nan=True), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "UTF-8 JSON object"):
                quantize_package(source, out, calibration)

            self.assertFalse(out.exists())

    def test_source_identity_mismatch_is_rejected_before_copying_package(self):
        for kind, filename, label in (
            ("encoder", "graph.json", "encoder graph"),
            ("decoder", "model.safetensors", "decoder weights"),
        ):
            with self.subTest(asset=label), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source"
                out = Path(directory) / "out"
                calibration = Path(directory) / "calibration.json"
                _write_package(source, specialized_graphs=True)
                calibration.write_text(
                    json.dumps(_calibration_document(source)), encoding="utf-8"
                )
                artifact = source / kind / filename
                artifact.write_bytes(artifact.read_bytes() + b"stale")

                with self.assertRaisesRegex(
                    ValueError, f"identity mismatch for {label}"
                ):
                    quantize_package(source, out, calibration)

                self.assertFalse(out.exists())

    def test_missing_source_identity_is_rejected_before_copying_package(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            out = Path(directory) / "out"
            calibration = Path(directory) / "calibration.json"
            _write_package(source, specialized_graphs=True)
            document = _calibration_document(source)
            del document["provenance"]["package"]
            calibration.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source package must be an object"):
                quantize_package(source, out, calibration)

            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
