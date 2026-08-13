from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import MappingProxyType
import unittest

from tools.exporter.generated.kernel_registry import OPERATOR_SHAPE_CONTRACTS
from tools.exporter.operator_shape_contracts import (
    OperatorShapeContractError,
    PerAxisQuantization,
    PerTensorQuantization,
    SPATIAL_OPERATOR_SHAPE_CONTRACTS,
    infer_concrete_operator_shapes,
    prove_operator_shape_domain,
)
from tools.exporter.shape_system import ShapeEnvironment


_VECTORS_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "operator_shape_contract_spatial_vectors.json"
)
_OPERATORS = frozenset(
    (
        "BatchMatMul",
        "Conv1D",
        "Conv2D",
        "ConvTranspose2D",
        "MaxPool2D",
        "AveragePool2D",
        "GlobalAveragePool",
        "Resize",
        "ResizeNearest2D",
        "UpsampleNearest2D",
    )
)


def _load_vectors() -> dict:
    return json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))


def _quantization_to_json(quantization: object) -> dict:
    if isinstance(quantization, PerTensorQuantization):
        return {
            "scheme": quantization.scheme,
            "scale": quantization.scale,
            "zero_point": quantization.zero_point,
        }
    if isinstance(quantization, PerAxisQuantization):
        return {
            "scheme": quantization.scheme,
            "axis": quantization.axis,
            "scales": list(quantization.scales),
            "zero_points": list(quantization.zero_points),
        }
    raise AssertionError(f"unexpected quantization {quantization!r}")


def _outputs_to_json(outputs: object) -> dict:
    converted = {}
    for name, descriptor in outputs.items():
        value = {"shape": list(descriptor.shape), "dtype": descriptor.dtype}
        if descriptor.quantization is not None:
            value["quantization"] = _quantization_to_json(
                descriptor.quantization
            )
        converted[name] = value
    return converted


def _tensor(shape: list[object], dtype: str = "float32") -> dict:
    return {"shape": shape, "dtype": dtype}


class SpatialShapeContractVectorTests(unittest.TestCase):
    def test_corpus_is_strict_and_covers_every_operator(self):
        vectors = _load_vectors()
        self.assertEqual(
            set(vectors),
            {"format", "success_cases", "failure_cases"},
        )
        self.assertEqual(
            vectors["format"],
            "volvox-operator-shape-spatial-vectors/v1",
        )
        self.assertEqual(set(SPATIAL_OPERATOR_SHAPE_CONTRACTS), _OPERATORS)

        ids: set[str] = set()
        successes = {operator: 0 for operator in _OPERATORS}
        failures = {operator: 0 for operator in _OPERATORS}
        for kind, cases, counts in (
            ("success", vectors["success_cases"], successes),
            ("failure", vectors["failure_cases"], failures),
        ):
            for case in cases:
                expected_fields = (
                    {"id", "operator", "request", "expected"}
                    if kind == "success"
                    else {"id", "operator", "request", "expected_error"}
                )
                self.assertEqual(set(case), expected_fields, case.get("id"))
                self.assertNotIn(case["id"], ids)
                ids.add(case["id"])
                self.assertIn(case["operator"], _OPERATORS)
                counts[case["operator"]] += 1
                serialized = json.dumps(case["request"])
                self.assertNotIn('"buffer"', serialized)
                self.assertNotIn('"data"', serialized)
                self.assertNotIn('"values"', serialized)
        for operator in _OPERATORS:
            self.assertGreaterEqual(successes[operator], 2, operator)
            self.assertGreaterEqual(failures[operator], 1, operator)

    def test_success_vectors_match_python_and_generated_registry(self):
        for original in _load_vectors()["success_cases"]:
            with self.subTest(case=original["id"]):
                case = deepcopy(original)
                before = deepcopy(case)
                operator = case["operator"]
                generated = OPERATOR_SHAPE_CONTRACTS[operator]
                contract = SPATIAL_OPERATOR_SHAPE_CONTRACTS[operator]
                self.assertEqual(generated["classification"], "canonical")
                self.assertEqual(
                    contract.shape_function_id,
                    generated["shape_function_id"],
                )
                outputs = infer_concrete_operator_shapes(
                    operator,
                    case["request"],
                )
                self.assertEqual(_outputs_to_json(outputs), case["expected"])
                self.assertEqual(case, before)
                self.assertIsInstance(outputs, MappingProxyType)

    def test_failure_vectors_have_exact_code_and_path(self):
        for original in _load_vectors()["failure_cases"]:
            with self.subTest(case=original["id"]):
                case = deepcopy(original)
                before = deepcopy(case)
                with self.assertRaises(OperatorShapeContractError) as caught:
                    infer_concrete_operator_shapes(
                        case["operator"],
                        case["request"],
                    )
                self.assertEqual(
                    caught.exception.code,
                    case["expected_error"]["code"],
                )
                self.assertEqual(
                    caught.exception.path,
                    case["expected_error"]["path"],
                )
                self.assertEqual(case, before)


class SpatialShapeContractDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = ShapeEnvironment(
            (
                {"name": "B", "min": 1, "max": 8},
                {"name": "H", "min": 1, "max": 32},
                {"name": "K", "min": 1, "max": 16},
                {"name": "S", "min": 1, "max": 16},
                {"name": "T", "min": 1, "max": 20},
                {"name": "W", "min": 1, "max": 32},
            )
        )

    def _prove(
        self,
        operator: str,
        inputs: dict,
        params: dict | None = None,
        declared_outputs: dict | None = None,
    ):
        request = {
            "environment": self.environment,
            "inputs": inputs,
        }
        if params is not None:
            request["params"] = params
        if declared_outputs is not None:
            request["declaredOutputs"] = declared_outputs
        return prove_operator_shape_domain(operator, request)

    def test_exact_symbolic_formulas_are_proved(self):
        cases = (
            (
                "BatchMatMul",
                {"a": _tensor(["B", 2, "K"]), "b": _tensor([1, "K", 3])},
                {},
                None,
                ("B", 2, 3),
            ),
            (
                "Conv1D",
                {"input": _tensor(["B", "S", 4]), "weight": _tensor([3, 4, 6])},
                {"padding": 1},
                None,
                ("B", "S", 6),
            ),
            (
                "Conv2D",
                {
                    "input": _tensor(["B", "H", "W", 4]),
                    "weight": _tensor([3, 3, 4, 6]),
                },
                {"padding": [1, 1], "weight_layout": "HWIO"},
                None,
                ("B", "H", "W", 6),
            ),
            (
                "ConvTranspose2D",
                {
                    "input": _tensor(["B", "H", "W", 4]),
                    "weight": _tensor([3, 3, 4, 6]),
                },
                {"kernel": [3, 3], "padding": [1, 1]},
                None,
                ("B", "H", "W", 6),
            ),
            (
                "MaxPool2D",
                {"input": _tensor(["B", "H", "W", 4])},
                {"kernel": [3, 3], "padding": [1, 1]},
                None,
                ("B", "H", "W", 4),
            ),
            (
                "AveragePool2D",
                {"input": _tensor(["B", "H", "W", 4])},
                {"kernel": [3, 3], "padding": [1, 1]},
                None,
                ("B", "H", "W", 4),
            ),
            (
                "GlobalAveragePool",
                {"input": _tensor(["B", "H", "W", 4])},
                {},
                None,
                ("B", 1, 1, 4),
            ),
            (
                "UpsampleNearest2D",
                {"input": _tensor(["B", 3, 5, 4])},
                {},
                None,
                ("B", 6, 10, 4),
            ),
        )
        for operator, inputs, params, declared, expected in cases:
            with self.subTest(operator=operator):
                proof = self._prove(operator, inputs, params, declared)
                self.assertTrue(
                    proof.supported,
                    getattr(proof, "reason", "domain proof rejected"),
                )
                self.assertEqual(proof.outputs["out"].shape, expected)

        for operator in ("Resize", "ResizeNearest2D"):
            with self.subTest(operator=operator):
                proof = self._prove(
                    operator,
                    {"input": _tensor(["B", "H", "W", 4])},
                    {"mode": "nearest"} if operator == "Resize" else {},
                    {"out": _tensor(["B", "S", "T", 4])},
                )
                self.assertTrue(
                    proof.supported,
                    getattr(proof, "reason", "domain proof rejected"),
                )
                self.assertEqual(
                    proof.outputs["out"].shape,
                    ("B", "S", "T", 4),
                )

    def test_unrepresentable_or_invalid_domains_fail_closed(self):
        cases = (
            (
                "BatchMatMul",
                {"a": _tensor(["B", 2, "K"]), "b": _tensor([1, "S", 3])},
                {},
                None,
                "UNPROVABLE_DYNAMIC_CONTRACTION",
            ),
            (
                "Conv1D",
                {"input": _tensor(["B", "S", 4]), "weight": _tensor([3, 4, 6])},
                {"stride": 2, "padding": 1},
                None,
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "Conv2D",
                {
                    "input": _tensor(["B", "H", "W", 4]),
                    "weight": _tensor([3, 3, 4, 6]),
                },
                {"stride": [2, 2], "padding": [1, 1]},
                None,
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "ConvTranspose2D",
                {
                    "input": _tensor(["B", "H", "W", 4]),
                    "weight": _tensor([3, 3, 4, 6]),
                },
                {"kernel": [3, 3], "stride": [2, 2], "padding": [1, 1]},
                None,
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "MaxPool2D",
                {"input": _tensor(["B", "H", "W", 4])},
                {"kernel": [2, 2], "stride": [2, 2]},
                None,
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "AveragePool2D",
                {"input": _tensor(["B", "H", "W", 4])},
                {"kernel": [2, 2], "stride": [2, 2]},
                None,
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "GlobalAveragePool",
                {"input": _tensor(["B", "H", 4])},
                {},
                None,
                "INVALID_RANK",
            ),
            (
                "Resize",
                {"input": _tensor(["B", "H", "W", 4])},
                {"mode": "nearest"},
                {"out": _tensor(["H", "S", "T", 4])},
                "SHAPE_MISMATCH",
            ),
            (
                "ResizeNearest2D",
                {"input": _tensor(["B", "H", "W", 4])},
                {"mode": "linear"},
                {"out": _tensor(["B", "S", "T", 4])},
                "INVALID_PARAMS",
            ),
            (
                "UpsampleNearest2D",
                {"input": _tensor(["B", "H", "W", 4])},
                {},
                None,
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
        )
        for operator, inputs, params, declared, expected_code in cases:
            with self.subTest(operator=operator):
                proof = self._prove(operator, inputs, params, declared)
                self.assertFalse(proof.supported)
                self.assertEqual(proof.code, expected_code, proof.reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
