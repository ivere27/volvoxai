from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import MappingProxyType
import unittest

from tools.exporter.operator_shape_contracts import (
    ATTENTION_OPERATOR_SHAPE_CONTRACTS,
    OperatorShapeContractError,
    get_operator_shape_contract,
    infer_concrete_operator_shapes,
)
from tools.exporter.shape_system import ShapeEnvironment


_VECTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "operator_shape_contract_attention_vectors.json"
)


def _load_vectors() -> dict:
    return json.loads(_VECTOR_PATH.read_text(encoding="utf-8"))


def _outputs_json(outputs: object) -> dict:
    return {
        name: {"shape": list(descriptor.shape), "dtype": descriptor.dtype}
        for name, descriptor in outputs.items()
    }


class AttentionSharedVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vectors = _load_vectors()

    def test_schema_manifest_and_semantic_variant_coverage_are_closed(self):
        vectors = self.vectors
        self.assertEqual(
            set(vectors),
            {"format", "families", "success_cases", "failure_cases"},
        )
        self.assertEqual(vectors["format"], "volvox-operator-shape-vectors/v1")
        manifest: set[str] = set()
        family_operators: dict[str, set[str]] = {}
        for family in vectors["families"]:
            self.assertEqual(set(family), {"id", "operators"})
            self.assertNotIn(family["id"], family_operators)
            operators = set(family["operators"])
            self.assertEqual(len(operators), len(family["operators"]))
            self.assertTrue(manifest.isdisjoint(operators))
            family_operators[family["id"]] = operators
            manifest.update(operators)
        self.assertEqual(manifest, set(ATTENTION_OPERATOR_SHAPE_CONTRACTS))

        ids: set[str] = set()
        legal = {operator: 0 for operator in manifest}
        illegal = {operator: 0 for operator in manifest}
        for cases, coverage in (
            (vectors["success_cases"], legal),
            (vectors["failure_cases"], illegal),
        ):
            for vector in cases:
                self.assertNotIn(vector["id"], ids)
                ids.add(vector["id"])
                self.assertTrue(
                    set(vector["operators"]).issubset(
                        family_operators[vector["family"]]
                    )
                )
                for operator in vector["operators"]:
                    coverage[operator] += 1
        for operator in manifest:
            self.assertGreaterEqual(legal[operator], 2, operator)
            self.assertGreaterEqual(illegal[operator], 1, operator)
        for variant in ("mask-k", "mask-bk", "mask-qk", "mask-bqk"):
            self.assertIn(f"sdpa-{variant}", ids)
            self.assertIn(f"cross-sdpa-{variant}", ids)

    def test_legal_vectors_match_concrete_and_bounded_python_contracts(self):
        for original in self.vectors["success_cases"]:
            for operator in original["operators"]:
                vector = deepcopy(original)
                before = deepcopy(vector)
                contract = get_operator_shape_contract(operator)
                self.assertEqual(
                    contract.shape_function_id,
                    vector["expected_shape_function_id"],
                    f"{vector['id']}:{operator}",
                )
                outputs = infer_concrete_operator_shapes(operator, vector["request"])
                self.assertIsInstance(outputs, MappingProxyType)
                self.assertEqual(
                    _outputs_json(outputs),
                    vector["expected"],
                    f"{vector['id']}:{operator}",
                )
                self.assertTrue(
                    all(descriptor.quantization is None for descriptor in outputs.values())
                )
                logical_request = deepcopy(vector["request"])
                logical_request["environment"] = ShapeEnvironment(())
                proof = contract.prove_domain(logical_request)
                self.assertTrue(
                    proof.supported,
                    f"{vector['id']}:{operator}: {getattr(proof, 'reason', '')}",
                )
                self.assertEqual(_outputs_json(proof.outputs), vector["expected"])
                self.assertEqual(vector, before)

    def test_illegal_vectors_have_exact_concrete_codes_and_paths(self):
        for original in self.vectors["failure_cases"]:
            for operator in original["operators"]:
                vector = deepcopy(original)
                before = deepcopy(vector)
                with self.assertRaises(OperatorShapeContractError) as caught:
                    infer_concrete_operator_shapes(operator, vector["request"])
                self.assertEqual(caught.exception.code, vector["expected_error"]["code"])
                self.assertEqual(caught.exception.path, vector["expected_error"]["path"])
                logical_request = deepcopy(vector["request"])
                logical_request["environment"] = ShapeEnvironment(())
                proof = get_operator_shape_contract(operator).prove_domain(
                    logical_request
                )
                self.assertFalse(proof.supported, f"{vector['id']}:{operator}")
                self.assertEqual(vector, before)


class AttentionSymbolicDomainTests(unittest.TestCase):
    def test_dynamic_batch_sequence_query_key_and_even_rope_width_are_proved(self):
        environment = ShapeEnvironment(
            (
                {"name": "B", "min": 1, "max": 4},
                {"name": "Q", "min": 1, "max": 8},
                {"name": "K", "min": 1, "max": 12},
                {"name": "S", "min": 1, "max": 16},
                {"name": "D", "min": 4, "max": 12, "multiple_of": 2},
            )
        )
        tensor = lambda shape, dtype="float32": {"shape": shape, "dtype": dtype}
        cases = {
            "SDPA": (
                {
                    "inputs": {
                        "qkv": tensor(["B", "S", 24]),
                        "mask": tensor(["B", "S", "S"], "int32"),
                    },
                    "params": {"heads": 2, "causal": True},
                },
                ("B", "S", 8),
            ),
            "CrossSDPA": (
                {
                    "inputs": {
                        "q": tensor(["B", "Q", 8]),
                        "k": tensor(["B", "K", 8]),
                        "v": tensor(["B", "K", 8]),
                        "mask": tensor(["Q", "K"], "int32"),
                    },
                    "params": {"heads": 4, "causal": False},
                },
                ("B", "Q", 8),
            ),
            "RoPE": (
                {
                    "inputs": {
                        "input": tensor(["B", "S", "D"]),
                        "position_ids": tensor(["B", "S"], "int32"),
                    },
                    "params": {},
                },
                ("B", "S", "D"),
            ),
        }
        for operator, (request, expected_shape) in cases.items():
            proof = get_operator_shape_contract(operator).prove_domain(
                {"environment": environment, **request}
            )
            self.assertTrue(
                proof.supported,
                f"{operator}: {getattr(proof, 'reason', '')}",
            )
            self.assertEqual(proof.outputs["out"].shape, expected_shape)

    def test_unprovable_attention_widths_are_rejected_independently(self):
        environment = ShapeEnvironment(
            (
                {"name": "S", "min": 1, "max": 16},
                {"name": "Packed", "min": 24, "max": 48, "multiple_of": 24},
                {"name": "Feature", "min": 8, "max": 16, "multiple_of": 8},
                {"name": "OddPossible", "min": 4, "max": 9},
            )
        )
        tensor = lambda shape: {"shape": shape, "dtype": "float32"}
        cases = {
            "SDPA": {
                "inputs": {"qkv": tensor(["S", "Packed"])},
                "params": {"heads": 2, "causal": True},
            },
            "CrossSDPA": {
                "inputs": {
                    "q": tensor(["S", "Feature"]),
                    "k": tensor(["S", "Feature"]),
                    "v": tensor(["S", "Feature"]),
                },
                "params": {"heads": 2, "causal": False},
            },
            "RoPE": {
                "inputs": {"input": tensor(["S", "OddPossible"])},
                "params": {},
            },
        }
        for operator, request in cases.items():
            proof = get_operator_shape_contract(operator).prove_domain(
                {"environment": environment, **request}
            )
            self.assertFalse(proof.supported, operator)
            self.assertEqual(proof.code, "UNPROVABLE_DYNAMIC_FEATURE")


if __name__ == "__main__":
    unittest.main()
