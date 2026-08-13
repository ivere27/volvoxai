from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools.exporter.generated.kernel_registry import PROFILE_MEMBERS
from tools.exporter.optimizer.safetensors_io import write_safetensors
from tools.exporter.portable_domain import PortableDomainProofError
from tools.exporter.validate_runtime_package import validate_runtime_package


def _identity_document(*, dynamic: bool) -> dict[str, object]:
    extent: int | str = "B" if dynamic else 2
    return {
        "format": "volvox-graph/v1",
        "dimensions": (
            {"B": {"min": 1, "max": 8, "multiple_of": 1}}
            if dynamic
            else {}
        ),
        "inputs": {"x": {"dtype": "float32", "shape": [extent, 4]}},
        "nodes": [{
            "id": "identity",
            "opType": "Identity",
            "inputs": {"input": "x"},
            "outputs": {"out": {
                "tensor": "y",
                "dtype": "float32",
                "shape": [extent, 4],
            }},
            "params": {},
        }],
        "outputs": ["y"],
    }


class RuntimePackageQualificationTests(unittest.TestCase):
    def _write_package(
        self,
        directory: str,
        document: dict[str, object],
        tensors: dict[str, np.ndarray] | None = None,
    ) -> tuple[Path, Path]:
        graph_path = Path(directory) / "graph.json"
        weights_path = Path(directory) / "model.safetensors"
        graph_path.write_text(json.dumps(document), encoding="utf-8")
        write_safetensors(weights_path, tensors or {})
        return graph_path, weights_path

    def test_qualification_proves_and_forwards_the_exact_portable_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_path, weights_path = self._write_package(
                directory,
                _identity_document(dynamic=True),
            )
            with mock.patch(
                "tools.exporter.validate_runtime_package."
                "prove_portable_graph_domain",
                wraps=__import__(
                    "tools.exporter.portable_domain",
                    fromlist=["prove_portable_graph_domain"],
                ).prove_portable_graph_domain,
            ) as prove, mock.patch(
                "tools.exporter.validate_runtime_package.import_runtime_package",
                wraps=__import__(
                    "tools.exporter.runtime_ir",
                    fromlist=["import_runtime_package"],
                ).import_runtime_package,
            ) as import_package:
                validate_runtime_package(graph_path, (weights_path,))

            self.assertEqual(prove.call_count, 1)
            self.assertEqual(
                prove.call_args.args[2],
                tuple(PROFILE_MEMBERS["portable"]),
            )
            proof = import_package.call_args.kwargs["bounded_domain_proof"]
            self.assertEqual(
                proof.backend_members,
                ("cpu-js", "wasm", "webgpu", "native-cpu"),
            )

    def test_qualification_applies_the_same_proof_to_constant_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_path, weights_path = self._write_package(
                directory,
                _identity_document(dynamic=False),
            )
            validate_runtime_package(graph_path, (weights_path,))

    def test_qualification_proves_f16_storage_as_runtime_f32(self):
        document = _identity_document(dynamic=False)
        document["inputs"] = {
            "x": {"dtype": "float32", "shape": [2, 2]},
        }
        document["nodes"] = [{
            "id": "linear",
            "opType": "Linear",
            "inputs": {"input": "x", "weight": "weight"},
            "outputs": {"out": {
                "tensor": "y",
                "dtype": "float32",
                "shape": [2, 3],
            }},
            "params": {"weight_layout": "din_dout"},
        }]
        with tempfile.TemporaryDirectory() as directory:
            graph_path, weights_path = self._write_package(
                directory,
                document,
                {"weight": np.zeros((2, 3), dtype=np.float16)},
            )
            validate_runtime_package(graph_path, (weights_path,))

    def test_qualification_rejects_an_operator_missing_from_one_member(self):
        document = _identity_document(dynamic=True)
        document["inputs"] = {
            "x": {"dtype": "float32", "shape": ["B", 4]},
            "indices": {"dtype": "int32", "shape": ["B", 4]},
        }
        document["nodes"] = [{
            "id": "gather",
            "opType": "GatherElements",
            "inputs": {"input": "x", "indices": "indices"},
            "outputs": {"out": {
                "tensor": "y",
                "dtype": "float32",
                "shape": ["B", 4],
            }},
            "params": {"axis": 1},
        }]
        with tempfile.TemporaryDirectory() as directory:
            graph_path, weights_path = self._write_package(directory, document)
            with self.assertRaises(PortableDomainProofError) as caught:
                validate_runtime_package(graph_path, (weights_path,))

        self.assertEqual(caught.exception.code, "VXDOMAIN_BACKEND")
        self.assertIn("native-cpu", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
