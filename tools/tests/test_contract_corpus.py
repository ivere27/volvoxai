from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.exporter.errors import ExporterError
from tools.exporter.generated.kernel_registry import PROFILE_MEMBERS
from tools.exporter.optimizer.safetensors_io import (
    read_safetensors,
    write_safetensors,
)
from tools.exporter.portable_domain import (
    PortableDomainProofError,
    prove_portable_graph_domain,
)
from tools.exporter.runtime_ir import import_runtime_package, load_runtime_document


CORPUS_PATH = REPOSITORY_ROOT / "tests" / "contracts" / "negative_contracts.json"
PORTABLE_MEMBERS = tuple(PROFILE_MEMBERS["portable"])


def _identity_document() -> dict[str, Any]:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {},
        "inputs": {"x": {"dtype": "float32", "shape": [2, 4]}},
        "outputs": ["y"],
        "nodes": [
            {
                "id": "identity_0",
                "opType": "Identity",
                "inputs": {"input": "x"},
                "outputs": {
                    "out": {
                        "tensor": "y",
                        "dtype": "float32",
                        "shape": [2, 4],
                    }
                },
                "params": {},
            }
        ],
    }


def _exact_shape_mismatch_document() -> dict[str, Any]:
    return {
        "format": "volvox-graph/v1",
        "dimensions": {"B": {"min": 1, "max": 4}},
        "inputs": {
            "a": {"dtype": "float32", "shape": ["B", 4]},
            # Match the native refusal fixture: 3 cannot broadcast to 4.
            "b": {"dtype": "float32", "shape": ["B", 3]},
        },
        "outputs": ["y"],
        "nodes": [
            {
                "id": "add_0",
                "opType": "Add",
                "inputs": {"a": "a", "b": "b"},
                "outputs": {
                    "out": {
                        "tensor": "y",
                        "dtype": "float32",
                        "shape": ["B", 4],
                    }
                },
                "params": {},
            }
        ],
    }


def _mutated_document(mutation: str) -> dict[str, Any]:
    if mutation == "exact-shape-domain-mismatch":
        return _exact_shape_mismatch_document()
    document = copy.deepcopy(_identity_document())
    if mutation == "wrong-format":
        document["format"] = "volvox-graph/v0"
    elif mutation == "invalid-dimension-bounds":
        document["dimensions"] = {"B": {"min": 4, "max": 2}}
        document["inputs"]["x"]["shape"] = ["B", 4]
        document["nodes"][0]["outputs"]["out"]["shape"] = ["B", 4]
    elif mutation == "output-assertion-mismatch":
        document["nodes"][0]["outputs"]["out"]["shape"] = [2, 5]
    elif mutation == "empty-graph-outputs":
        document["outputs"] = []
    elif mutation == "unsupported-portable-operator":
        # Sin has the same one-input/one-output contract as Identity, but is
        # outside the generated portable backend intersection.
        # The fixture remains valid if membership and port checks are reordered.
        document["nodes"][0]["opType"] = "Sin"
    else:
        raise AssertionError(f"unknown corpus mutation {mutation!r}")
    return document


def _load_corpus() -> dict[str, Any]:
    document = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise AssertionError("negative contract corpus schema is unsupported")
    expected = {
        "schema_version",
        "runtime_document_cases",
        "runtime_package_cases",
        "portable_domain_cases",
        "safetensors_cases",
    }
    if set(document) != expected:
        raise AssertionError("negative contract corpus root fields are malformed")
    return document


def _assert_reuse_enabled(case: object) -> dict[str, Any]:
    if not isinstance(case, dict) or case.get("reuse_probe") is not True:
        raise AssertionError("every negative contract must include its reuse probe")
    return case


class NegativeContractCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = _load_corpus()

    def _assert_valid_import_reuse(self) -> None:
        graph = import_runtime_package(_identity_document(), {})
        self.assertEqual(graph.outputs, ["y"])

    def _assert_valid_domain_reuse(self) -> None:
        proof = prove_portable_graph_domain(
            _identity_document(), {}, PORTABLE_MEMBERS
        )
        self.assertEqual(tuple(node.operator for node in proof.nodes), ("Identity",))

    def test_strict_runtime_document_ingress_and_reuse(self) -> None:
        valid_source = json.dumps(_identity_document(), separators=(",", ":"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            for raw_case in self.corpus["runtime_document_cases"]:
                case = _assert_reuse_enabled(raw_case)
                with self.subTest(case=case["id"]):
                    path.write_text(case["source"], encoding="utf-8")
                    with self.assertRaises(ExporterError) as caught:
                        load_runtime_document(path)
                    self.assertEqual(caught.exception.diagnostic.code, case["error_code"])
                    self.assertIn(case["message_contains"], str(caught.exception))

                    path.write_text(valid_source, encoding="utf-8")
                    recovered = load_runtime_document(path)
                    graph = import_runtime_package(recovered, {})
                    self.assertEqual(graph.outputs, ["y"])

    def test_runtime_package_rejection_and_reuse(self) -> None:
        for raw_case in self.corpus["runtime_package_cases"]:
            case = _assert_reuse_enabled(raw_case)
            with self.subTest(case=case["id"]):
                with self.assertRaises(ExporterError) as caught:
                    import_runtime_package(_mutated_document(case["mutation"]), {})
                self.assertEqual(caught.exception.diagnostic.code, case["error_code"])
                self.assertIn(case["message_contains"], str(caught.exception))
                self._assert_valid_import_reuse()

    def test_portable_domain_rejection_and_reuse(self) -> None:
        for raw_case in self.corpus["portable_domain_cases"]:
            case = _assert_reuse_enabled(raw_case)
            with self.subTest(case=case["id"]):
                with self.assertRaises(PortableDomainProofError) as caught:
                    prove_portable_graph_domain(
                        _mutated_document(case["mutation"]),
                        {},
                        PORTABLE_MEMBERS,
                    )
                self.assertEqual(caught.exception.code, case["error_code"])
                self.assertIn(case["message_contains"], str(caught.exception))
                self._assert_valid_domain_reuse()

    def test_safetensors_ingress_rejection_and_reuse(self) -> None:
        valid = np.asarray([1.25], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            for raw_case in self.corpus["safetensors_cases"]:
                case = _assert_reuse_enabled(raw_case)
                with self.subTest(case=case["id"]):
                    if "raw_hex" in case:
                        payload = bytes.fromhex(case["raw_hex"])
                    else:
                        header = case["header_json"].encode("utf-8")
                        payload = (
                            struct.pack("<Q", len(header))
                            + header
                            + bytes.fromhex(case["body_hex"])
                        )
                    path.write_bytes(payload)
                    with self.assertRaises(ValueError) as caught:
                        read_safetensors(path)
                    self.assertIn(case["message_contains"], str(caught.exception))

                    write_safetensors(path, {"ok": valid})
                    recovered = read_safetensors(path)
                    self.assertEqual(set(recovered), {"ok"})
                    np.testing.assert_array_equal(recovered["ok"], valid)


if __name__ == "__main__":
    unittest.main()
