from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import MappingProxyType
import unittest

from tools.exporter.operator_shape_contracts import (
    OperatorShapeContractError,
    OperatorTensorDescriptor,
    PerAxisQuantization,
    PerTensorQuantization,
    WAVE_A_OPERATOR_SHAPE_CONTRACTS,
    get_operator_shape_contract,
    infer_concrete_operator_shapes,
)
from tools.exporter.shape_system import ShapeEnvironment


_VECTORS_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "operator_shape_contract_vectors.json"
)
_ROOT_FIELDS = {
    "format",
    "families",
    "success_cases",
    "failure_cases",
}
_SUCCESS_FIELDS = {
    "id",
    "family",
    "operators",
    "request",
    "expected_shape_function_id",
    "expected",
}
_FAILURE_FIELDS = {
    "id",
    "family",
    "operators",
    "request",
    "expected_shape_function_id",
    "expected_error",
}
_FORBIDDEN_VALUE_FIELDS = {"data", "value", "values", "buffer"}


def _load_vectors() -> dict:
    return json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))


def _assert_exact_fields(
    testcase: unittest.TestCase,
    value: object,
    expected: set[str],
    path: str,
) -> None:
    testcase.assertIsInstance(value, dict, path)
    testcase.assertEqual(set(value), expected, path)


def _assert_no_values(
    testcase: unittest.TestCase,
    value: object,
    path: str,
) -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            testcase.assertNotIn(name, _FORBIDDEN_VALUE_FIELDS, path)
            _assert_no_values(testcase, child, f"{path}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_values(testcase, child, f"{path}[{index}]")


def _assert_descriptor_schema(
    testcase: unittest.TestCase,
    descriptor: object,
    path: str,
) -> None:
    testcase.assertIsInstance(descriptor, dict, path)
    allowed = {"shape", "dtype", "quantization"}
    testcase.assertTrue(set(descriptor).issubset(allowed), path)
    testcase.assertIn("shape", descriptor, path)
    testcase.assertIn("dtype", descriptor, path)
    testcase.assertIsInstance(descriptor["shape"], list, path)
    for dimension in descriptor["shape"]:
        testcase.assertIsInstance(dimension, int, path)
        testcase.assertNotIsInstance(dimension, bool, path)
        testcase.assertGreater(dimension, 0, path)
    testcase.assertIn(
        descriptor["dtype"],
        ("float32", "int32", "int8", "uint8"),
        path,
    )
    if "quantization" not in descriptor:
        return
    quantization = descriptor["quantization"]
    testcase.assertIsInstance(quantization, dict, path)
    if quantization.get("scheme") == "per_tensor":
        testcase.assertEqual(
            set(quantization),
            {"scheme", "scale", "zero_point"},
            path,
        )
    elif quantization.get("scheme") == "per_axis":
        testcase.assertEqual(
            set(quantization),
            {"scheme", "axis", "scales", "zero_points"},
            path,
        )
        testcase.assertIsInstance(quantization["scales"], list, path)
        testcase.assertIsInstance(quantization["zero_points"], list, path)
    else:
        testcase.fail(f"{path}: unknown quantization scheme")


def _assert_request_schema(
    testcase: unittest.TestCase,
    request: object,
    path: str,
) -> None:
    testcase.assertIsInstance(request, dict, path)
    testcase.assertTrue(
        set(request).issubset({"inputs", "params", "declaredOutputs"}),
        path,
    )
    testcase.assertIn("inputs", request, path)
    testcase.assertIsInstance(request["inputs"], dict, path)
    testcase.assertGreater(len(request["inputs"]), 0, path)
    for name, descriptor in request["inputs"].items():
        testcase.assertIsInstance(name, str, path)
        testcase.assertTrue(name, path)
        _assert_descriptor_schema(
            testcase,
            descriptor,
            f"{path}.inputs.{name}",
        )
    if "params" in request:
        testcase.assertIsInstance(request["params"], dict, path)
    if "declaredOutputs" in request:
        testcase.assertEqual(set(request["declaredOutputs"]), {"out"}, path)
        _assert_descriptor_schema(
            testcase,
            request["declaredOutputs"]["out"],
            f"{path}.declaredOutputs.out",
        )
    _assert_no_values(testcase, request, path)


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
        raw = {"shape": list(descriptor.shape), "dtype": descriptor.dtype}
        if descriptor.quantization is not None:
            raw["quantization"] = _quantization_to_json(
                descriptor.quantization
            )
        converted[name] = raw
    return converted


def _logical_tensor(
    shape: list[object],
    dtype: str = "float32",
    quantization: dict | None = None,
) -> dict:
    descriptor = {"shape": shape, "dtype": dtype}
    if quantization is not None:
        descriptor["quantization"] = quantization
    return descriptor


class OperatorShapeCorpusSchemaTests(unittest.TestCase):
    def test_corpus_schema_coverage_and_duplicate_ids_are_strict(self):
        vectors = _load_vectors()
        _assert_exact_fields(self, vectors, _ROOT_FIELDS, "corpus")
        self.assertEqual(
            vectors["format"],
            "volvox-operator-shape-vectors/v1",
        )
        family_ids: set[str] = set()
        family_operators: dict[str, set[str]] = {}
        all_manifest_operators: set[str] = set()
        for index, family in enumerate(vectors["families"]):
            path = f"families[{index}]"
            _assert_exact_fields(self, family, {"id", "operators"}, path)
            self.assertIsInstance(family["id"], str, path)
            self.assertTrue(family["id"], path)
            self.assertNotIn(family["id"], family_ids, path)
            family_ids.add(family["id"])
            self.assertIsInstance(family["operators"], list, path)
            self.assertGreater(len(family["operators"]), 0, path)
            operators = set(family["operators"])
            self.assertEqual(len(operators), len(family["operators"]), path)
            self.assertTrue(operators.isdisjoint(all_manifest_operators), path)
            family_operators[family["id"]] = operators
            all_manifest_operators.update(operators)

        self.assertEqual(
            all_manifest_operators,
            set(WAVE_A_OPERATOR_SHAPE_CONTRACTS),
        )

        all_case_ids: set[str] = set()
        success_by_family = {family_id: 0 for family_id in family_ids}
        failure_by_family = {family_id: 0 for family_id in family_ids}
        successful_operators: set[str] = set()
        failed_operators: set[str] = set()
        for kind, cases, fields, counts, covered in (
            (
                "success",
                vectors["success_cases"],
                _SUCCESS_FIELDS,
                success_by_family,
                successful_operators,
            ),
            (
                "failure",
                vectors["failure_cases"],
                _FAILURE_FIELDS,
                failure_by_family,
                failed_operators,
            ),
        ):
            self.assertIsInstance(cases, list)
            for index, case in enumerate(cases):
                path = f"{kind}_cases[{index}]"
                _assert_exact_fields(self, case, fields, path)
                self.assertIsInstance(case["id"], str, path)
                self.assertTrue(case["id"], path)
                self.assertNotIn(case["id"], all_case_ids, path)
                all_case_ids.add(case["id"])
                self.assertIn(case["family"], family_ids, path)
                self.assertIsInstance(case["operators"], list, path)
                self.assertGreater(len(case["operators"]), 0, path)
                operators = set(case["operators"])
                self.assertEqual(len(operators), len(case["operators"]), path)
                self.assertTrue(
                    operators.issubset(family_operators[case["family"]]),
                    path,
                )
                counts[case["family"]] += 1
                covered.update(operators)
                _assert_request_schema(self, case["request"], path)
                self.assertIsInstance(
                    case["expected_shape_function_id"],
                    str,
                    path,
                )
                if kind == "success":
                    self.assertEqual(set(case["expected"]), {"out"}, path)
                    _assert_descriptor_schema(
                        self,
                        case["expected"]["out"],
                        f"{path}.expected.out",
                    )
                else:
                    _assert_exact_fields(
                        self,
                        case["expected_error"],
                        {"code", "path"},
                        f"{path}.expected_error",
                    )

        self.assertTrue(all(count >= 2 for count in success_by_family.values()))
        self.assertTrue(all(count >= 1 for count in failure_by_family.values()))
        self.assertEqual(successful_operators, all_manifest_operators)
        self.assertEqual(failed_operators, all_manifest_operators)


class OperatorShapeSharedVectorTests(unittest.TestCase):
    def test_shared_success_vectors_and_generated_ids(self):
        vectors = _load_vectors()
        for original in vectors["success_cases"]:
            for operator in original["operators"]:
                with self.subTest(case=original["id"], operator=operator):
                    case = deepcopy(original)
                    before = deepcopy(case)
                    contract = get_operator_shape_contract(operator)
                    self.assertEqual(
                        contract.shape_function_id,
                        case["expected_shape_function_id"],
                    )
                    outputs = infer_concrete_operator_shapes(
                        operator,
                        case["request"],
                    )
                    self.assertEqual(
                        _outputs_to_json(outputs),
                        case["expected"],
                    )
                    self.assertEqual(case, before)
                    self.assertIsInstance(outputs, MappingProxyType)
                    self.assertIsInstance(outputs["out"], OperatorTensorDescriptor)
                    self.assertIsInstance(outputs["out"].shape, tuple)
                    with self.assertRaises(TypeError):
                        outputs["other"] = outputs["out"]
                    with self.assertRaises(FrozenInstanceError):
                        outputs["out"].dtype = "int8"
                    quantization = outputs["out"].quantization
                    if isinstance(quantization, PerAxisQuantization):
                        self.assertIsInstance(quantization.scales, tuple)
                        self.assertIsInstance(quantization.zero_points, tuple)
                    logical_request = deepcopy(case["request"])
                    logical_request["environment"] = ShapeEnvironment(())
                    proof = contract.prove_domain(logical_request)
                    self.assertTrue(
                        proof.supported,
                        getattr(proof, "reason", "domain proof rejected"),
                    )
                    self.assertEqual(
                        _outputs_to_json(proof.outputs),
                        case["expected"],
                    )

    def test_shared_failure_vectors_have_exact_code_and_path(self):
        vectors = _load_vectors()
        for original in vectors["failure_cases"]:
            for operator in original["operators"]:
                with self.subTest(case=original["id"], operator=operator):
                    case = deepcopy(original)
                    before = deepcopy(case)
                    contract = get_operator_shape_contract(operator)
                    self.assertEqual(
                        contract.shape_function_id,
                        case["expected_shape_function_id"],
                    )
                    with self.assertRaises(OperatorShapeContractError) as caught:
                        infer_concrete_operator_shapes(
                            operator,
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
                    logical_request = deepcopy(case["request"])
                    logical_request["environment"] = ShapeEnvironment(())
                    proof = contract.prove_domain(logical_request)
                    self.assertFalse(proof.supported)
                    self.assertEqual(case, before)

    def test_unknown_operator_is_fail_closed(self):
        with self.assertRaises(OperatorShapeContractError) as caught:
            infer_concrete_operator_shapes(
                "Unknown",
                {"inputs": {"input": {"shape": [1], "dtype": "float32"}}},
            )
        self.assertEqual(caught.exception.code, "UNKNOWN_OPERATOR")
        self.assertEqual(caught.exception.path, "operator")


class OperatorShapeSymbolicDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = ShapeEnvironment(
            (
                {"name": "B", "min": 1, "max": 8},
                {"name": "C", "min": 4, "max": 12},
                {"name": "F", "min": 2, "max": 8},
                {"name": "H", "min": 1, "max": 32},
                {"name": "S", "min": 1, "max": 16},
                {"name": "T", "min": 1, "max": 20},
                {"name": "W", "min": 1, "max": 32},
            )
        )

    def _prove(self, operator: str, request: dict):
        return get_operator_shape_contract(operator).prove_domain(
            {"environment": self.environment, **request}
        )

    def test_every_wave_a_operator_accepts_a_genuinely_symbolic_domain(self):
        dynamic = _logical_tensor(["B", "S", 4])
        requests: dict[str, tuple[dict, tuple[object, ...], str]] = {
            "Identity": ({"inputs": {"input": dynamic}}, ("B", "S", 4), "float32"),
            "PReLU": (
                {"inputs": {"input": dynamic, "slope": _logical_tensor([4])}},
                ("B", "S", 4),
                "float32",
            ),
            "Cast": (
                {
                    "inputs": {"input": _logical_tensor(["B", "S", 4], "int32")},
                    "params": {"to": "float32"},
                },
                ("B", "S", 4),
                "float32",
            ),
            "LayerNorm": (
                {
                    "inputs": {
                        "input": dynamic,
                        "weight": _logical_tensor([4]),
                        "bias": _logical_tensor([4]),
                    }
                },
                ("B", "S", 4),
                "float32",
            ),
            "RMSNorm": (
                {"inputs": {"input": dynamic, "weight": _logical_tensor([4])}},
                ("B", "S", 4),
                "float32",
            ),
            "GroupNorm": (
                {
                    "inputs": {
                        "input": _logical_tensor(["B", "H", "W", 8]),
                        "weight": _logical_tensor([8]),
                        "bias": _logical_tensor([8]),
                    },
                    "params": {"num_groups": 4},
                },
                ("B", "H", "W", 8),
                "float32",
            ),
            "Embedding": (
                {
                    "inputs": {
                        "input": _logical_tensor(["B", "S"], "int32"),
                        "weight": _logical_tensor([32, 4]),
                    }
                },
                ("B", "S", 4),
                "float32",
            ),
        }
        for operator in (
            "ReLU",
            "LeakyReLU",
            "GELU",
            "SiLU",
            "Sigmoid",
            "HardSwish",
            "HardSigmoid",
            "Tanh",
            "Sin",
            "Cos",
            "Softmax",
            "LogSoftmax",
        ):
            requests[operator] = (
                {"inputs": {"input": dynamic}},
                ("B", "S", 4),
                "float32",
            )
        requests["Clip"] = (
            {
                "inputs": {"input": _logical_tensor(["B", "S"], "int32")},
                "params": {"min": 0, "max": 31},
            },
            ("B", "S"),
            "int32",
        )

        quantization = {
            "scheme": "per_tensor",
            "scale": 0.25,
            "zero_point": 0,
        }
        requests["QuantizeLinear"] = (
            {
                "inputs": {
                    "input": dynamic,
                    "scale": _logical_tensor([1]),
                    "zero_point": _logical_tensor([1], "int8"),
                },
                "declaredOutputs": {
                    "out": _logical_tensor(
                        ["B", "S", 4], "int8", quantization
                    )
                },
            },
            ("B", "S", 4),
            "int8",
        )
        requests["DequantizeLinear"] = (
            {
                "inputs": {
                    "input": _logical_tensor(
                        ["B", "S", 4], "int8", quantization
                    ),
                    "scale": _logical_tensor([1]),
                    "zero_point": _logical_tensor([1], "int8"),
                }
            },
            ("B", "S", 4),
            "float32",
        )
        for operator, weight_shape in (
            ("Linear", [6, 4]),
            ("Gemm", [4, 6]),
            ("MatMul", [4, 6]),
        ):
            requests[operator] = (
                {
                    "inputs": {
                        "input": dynamic,
                        "weight": _logical_tensor(weight_shape),
                        "bias": _logical_tensor([6]),
                    }
                },
                ("B", "S", 6),
                "float32",
            )
        for operator in ("Add", "Mul"):
            requests[operator] = (
                {"inputs": {"a": dynamic, "b": dynamic}},
                ("B", "S", 4),
                "float32",
            )

        self.assertEqual(set(requests), set(WAVE_A_OPERATOR_SHAPE_CONTRACTS))
        for operator, (request, expected_shape, expected_dtype) in requests.items():
            with self.subTest(operator=operator):
                proof = self._prove(operator, request)
                self.assertTrue(proof.supported, getattr(proof, "reason", ""))
                self.assertEqual(proof.outputs["out"].shape, expected_shape)
                self.assertEqual(proof.outputs["out"].dtype, expected_dtype)

    def test_unprovable_wave_a_dynamic_relations_fail_closed(self):
        per_axis = {
            "scheme": "per_axis",
            "axis": 1,
            "scales": [0.25] * 16,
            "zero_points": [0] * 16,
        }
        cases = (
            (
                "PReLU",
                {
                    "inputs": {
                        "input": _logical_tensor(["B", "S", "F"]),
                        "slope": _logical_tensor([4]),
                    }
                },
                "UNPROVABLE_DYNAMIC_FEATURE",
            ),
            (
                "QuantizeLinear",
                {
                    "inputs": {
                        "input": _logical_tensor(["B", "S", 4]),
                        "scale": _logical_tensor([16]),
                    },
                    "declaredOutputs": {
                        "out": _logical_tensor(
                            ["B", "S", 4], "int8", per_axis
                        )
                    },
                },
                "UNPROVABLE_DYNAMIC_PER_AXIS_EXTENT",
            ),
            (
                "LayerNorm",
                {
                    "inputs": {
                        "input": _logical_tensor(["B", "S", "F"]),
                        "weight": _logical_tensor([4]),
                    }
                },
                "UNPROVABLE_DYNAMIC_FEATURE",
            ),
            (
                "GroupNorm",
                {
                    "inputs": {
                        "input": _logical_tensor(["B", "H", "W", "C"]),
                        "weight": _logical_tensor([8]),
                        "bias": _logical_tensor([8]),
                    },
                    "params": {"num_groups": 4},
                },
                "UNPROVABLE_DYNAMIC_CHANNEL",
            ),
            (
                "Linear",
                {
                    "inputs": {
                        "input": _logical_tensor(["B", "S", "F"]),
                        "weight": _logical_tensor([6, 4]),
                    }
                },
                "UNPROVABLE_DYNAMIC_CONTRACTION",
            ),
            (
                "Add",
                {
                    "inputs": {
                        "a": _logical_tensor(["B", "S", 4]),
                        "b": _logical_tensor(["B", "T", 4]),
                    }
                },
                "SHAPE_MISMATCH",
            ),
        )
        for operator, request, expected_code in cases:
            with self.subTest(operator=operator):
                proof = self._prove(operator, request)
                self.assertFalse(proof.supported)
                self.assertEqual(proof.code, expected_code, proof.reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
