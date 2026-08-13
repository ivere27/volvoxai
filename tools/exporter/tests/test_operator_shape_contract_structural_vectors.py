from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import MappingProxyType
import unittest

from tools.exporter.operator_shape_contracts import (
    STRUCTURAL_OPERATOR_SHAPE_CONTRACTS,
    OperatorShapeContractError,
    PerAxisQuantization,
    PerTensorQuantization,
    get_operator_shape_contract,
    infer_concrete_operator_shapes,
)
from tools.exporter.portable_domain import (
    PortableDomainProofError,
    prove_portable_graph_domain,
)
from tools.exporter.shape_system import ShapeEnvironment


_VECTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "operator_shape_contract_structural_vectors.json"
)
_CONCAT_AFFINE_VECTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "operator_shape_contract_concat_affine_vectors.json"
)


def _load_vectors() -> dict:
    return json.loads(_VECTOR_PATH.read_text(encoding="utf-8"))


def _quantization_json(value: object) -> dict:
    if isinstance(value, PerTensorQuantization):
        return {
            "scheme": value.scheme,
            "scale": value.scale,
            "zero_point": value.zero_point,
        }
    if isinstance(value, PerAxisQuantization):
        return {
            "scheme": value.scheme,
            "axis": value.axis,
            "scales": list(value.scales),
            "zero_points": list(value.zero_points),
        }
    raise AssertionError(f"unexpected quantization {value!r}")


def _outputs_json(outputs: object) -> dict:
    converted: dict[str, dict] = {}
    for name, descriptor in outputs.items():
        raw = {"shape": list(descriptor.shape), "dtype": descriptor.dtype}
        if descriptor.quantization is not None:
            raw["quantization"] = _quantization_json(descriptor.quantization)
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


class NonSpatialStructuralSharedVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vectors = _load_vectors()

    def test_schema_and_per_operator_coverage_are_closed(self):
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
        self.assertEqual(
            manifest,
            set(STRUCTURAL_OPERATOR_SHAPE_CONTRACTS),
        )

        case_ids: set[str] = set()
        legal = {operator: 0 for operator in manifest}
        illegal = {operator: 0 for operator in manifest}
        for cases, coverage in (
            (vectors["success_cases"], legal),
            (vectors["failure_cases"], illegal),
        ):
            for vector in cases:
                self.assertNotIn(vector["id"], case_ids)
                case_ids.add(vector["id"])
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
                outputs = infer_concrete_operator_shapes(
                    operator, vector["request"]
                )
                self.assertIsInstance(outputs, MappingProxyType)
                self.assertEqual(
                    _outputs_json(outputs),
                    vector["expected"],
                    f"{vector['id']}:{operator}",
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

    def test_literal_scalar_broadcast_preserves_singleton_symbol(self):
        environment = ShapeEnvironment((
            {"name": "B", "min": 1, "max": 1},
            {"name": "Q", "min": 1, "max": 192},
        ))
        proof = get_operator_shape_contract("Equal").prove_domain({
            "environment": environment,
            "inputs": {
                "a": {"shape": ["B", "Q"], "dtype": "int32"},
                "b": {"shape": [], "dtype": "int32"},
            },
        })

        self.assertTrue(proof.supported, getattr(proof, "reason", ""))
        self.assertEqual(
            _outputs_json(proof.outputs),
            {"out": {"shape": ["B", "Q"], "dtype": "int32"}},
        )

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


class ConcatAffineSharedVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vectors = json.loads(
            _CONCAT_AFFINE_VECTOR_PATH.read_text(encoding="utf-8")
        )

    def test_typescript_python_affine_concat_corpus_is_closed(self):
        self.assertEqual(set(self.vectors), {"format", "cases"})
        self.assertEqual(
            self.vectors["format"],
            "volvox-concat-affine-domain-vectors/v1",
        )
        identifiers = [vector["id"] for vector in self.vectors["cases"]]
        self.assertTrue(identifiers)
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_one_dynamic_concat_term_has_exact_python_domain_parity(self):
        contract = get_operator_shape_contract("Concat")
        for original in self.vectors["cases"]:
            vector = deepcopy(original)
            before = deepcopy(vector)
            environment = ShapeEnvironment(tuple(
                {"name": name, **constraint}
                for name, constraint in vector["environment"].items()
            ))
            proof = contract.prove_domain({
                **vector["request"],
                "environment": environment,
            })
            if "expected" in vector:
                self.assertTrue(
                    proof.supported,
                    f"{vector['id']}: {getattr(proof, 'reason', '')}",
                )
                self.assertEqual(_outputs_json(proof.outputs), vector["expected"])
                self.assertEqual(list(proof.facts), vector["facts"])
                self.assertEqual(
                    [
                        {
                            "target": relation.target,
                            "source": relation.source,
                            "offset": relation.offset,
                        }
                        for relation in proof.affine_relations
                    ],
                    vector["relations"],
                )
            else:
                self.assertFalse(proof.supported, vector["id"])
                self.assertEqual(
                    proof.code, vector["expected_error"]["code"], vector["id"]
                )
                self.assertTrue(
                    proof.reason.startswith(
                        f"{vector['expected_error']['path']}:"
                    ),
                    f"{vector['id']}: {proof.reason}",
                )
            self.assertEqual(vector, before)

    @staticmethod
    def _repeated_affine_graph() -> dict:
        return {
            "format": "volvox-graph/v1",
            "dimensions": {
                "B": {"min": 1, "max": 1},
                "Q": {"min": 1, "max": 192},
                "M": {"min": 211, "max": 402},
            },
            "inputs": {
                "image": {"dtype": "float32", "shape": ["B", 210, 320]},
                "question": {"dtype": "float32", "shape": ["B", "Q", 320]},
                "image_mask": {"dtype": "int32", "shape": ["B", 210]},
                "question_mask": {"dtype": "int32", "shape": ["B", "Q"]},
            },
            "nodes": [
                {
                    "id": "memory_concat",
                    "opType": "Concat",
                    "inputs": {"input0": "image", "input1": "question"},
                    "outputs": {"out": {
                        "tensor": "memory",
                        "dtype": "float32",
                        "shape": ["B", "M", 320],
                    }},
                    "params": {"axis": 1},
                },
                {
                    "id": "mask_concat",
                    "opType": "Concat",
                    "inputs": {
                        "input0": "image_mask",
                        "input1": "question_mask",
                    },
                    "outputs": {"out": {
                        "tensor": "memory_mask",
                        "dtype": "int32",
                        "shape": ["B", "M"],
                    }},
                    "params": {"axis": 1},
                },
            ],
            "outputs": ["memory", "memory_mask"],
        }

    def test_portable_graph_allows_exact_affine_reuse(self):
        proof = prove_portable_graph_domain(
            self._repeated_affine_graph(), {}, ("cpu-js",)
        )
        self.assertEqual(
            [node.facts for node in proof.nodes],
            [
                ("M=Q+210 exactly over the complete bounded Concat domain",),
                ("M=Q+210 exactly over the complete bounded Concat domain",),
            ],
        )

    def test_portable_graph_rejects_public_or_conflicting_affine_target(self):
        public_target = self._repeated_affine_graph()
        public_target["inputs"]["prebound_memory"] = {
            "dtype": "float32",
            "shape": ["M"],
        }
        with self.assertRaises(PortableDomainProofError) as public_error:
            prove_portable_graph_domain(public_target, {}, ("cpu-js",))
        self.assertEqual(public_error.exception.code, "VXCAP_DOMAIN_PROOF")
        self.assertIn("public input or non-affine output", public_error.exception.detail)

        conflicting = self._repeated_affine_graph()
        conflicting["dimensions"]["R"] = {"min": 1, "max": 192}
        conflicting["inputs"]["question_mask"]["shape"] = ["B", "R"]
        with self.assertRaises(PortableDomainProofError) as conflict_error:
            prove_portable_graph_domain(conflicting, {}, ("cpu-js",))
        self.assertEqual(conflict_error.exception.code, "VXCAP_DOMAIN_PROOF")
        self.assertIn("Q+210, not R+210", conflict_error.exception.detail)


class NonSpatialStructuralSymbolicDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = ShapeEnvironment(
            (
                {"name": "B", "min": 1, "max": 8},
                {"name": "S", "min": 1, "max": 16},
                {"name": "T", "min": 1, "max": 20},
            )
        )

    def _prove(self, operator: str, request: dict):
        logical_request = {"environment": self.environment, **request}
        return get_operator_shape_contract(operator).prove_domain(logical_request)

    def test_every_non_spatial_structural_operator_accepts_a_symbolic_domain(self):
        float_dynamic = _logical_tensor(["B", "S", 4])
        int_dynamic = _logical_tensor(["B", "S", 4], "int32")
        broadcast_float = {
            "inputs": {
                "a": float_dynamic,
                "b": _logical_tensor([1, 1, 4]),
            }
        }
        broadcast_int = {
            "inputs": {
                "a": int_dynamic,
                "b": _logical_tensor([1, 1, 4], "int32"),
            }
        }
        requests: dict[str, tuple[dict, dict[str, tuple[tuple[object, ...], str]]]] = {}
        for operator in ("Sub", "Div"):
            requests[operator] = (
                broadcast_float,
                {"out": (("B", "S", 4), "float32")},
            )
        for operator in ("Equal", "GreaterOrEqual"):
            requests[operator] = (
                broadcast_int,
                {"out": (("B", "S", 4), "int32")},
            )
        requests.update(
            {
                "Where": (
                    {
                        "inputs": {
                            "condition": _logical_tensor(["B", 1, 1], "int32"),
                            "a": float_dynamic,
                            "b": float_dynamic,
                        }
                    },
                    {"out": (("B", "S", 4), "float32")},
                ),
                "ReduceSum": (
                    {
                        "inputs": {"input": float_dynamic},
                        "params": {"axis": -1, "keepdims": False},
                    },
                    {"out": (("B", "S"), "float32")},
                ),
                "ReduceMean": (
                    {
                        "inputs": {"input": float_dynamic},
                        "params": {"axis": -1, "keepdims": False},
                    },
                    {"out": (("B", "S"), "float32")},
                ),
                "ArgMax": (
                    {
                        "inputs": {"input": float_dynamic},
                        "params": {"axis": 1, "keepdims": False},
                    },
                    {"out": (("B", 4), "int32")},
                ),
                "Transpose": (
                    {
                        "inputs": {"input": float_dynamic},
                        "params": {"perm": [1, 0, 2]},
                    },
                    {"out": (("S", "B", 4), "float32")},
                ),
                "Flatten": (
                    {
                        "inputs": {"input": _logical_tensor(["B", 1, "S"])},
                        "params": {"axis": 1},
                    },
                    {"out": (("B", "S"), "float32")},
                ),
                "Squeeze": (
                    {
                        "inputs": {"input": _logical_tensor(["B", 1, "S"])},
                        "params": {"axes": [1]},
                    },
                    {"out": (("B", "S"), "float32")},
                ),
                "Unsqueeze": (
                    {
                        "inputs": {"input": _logical_tensor(["B", "S"])},
                        "params": {"axes": [1]},
                    },
                    {"out": (("B", 1, "S"), "float32")},
                ),
                "Reshape": (
                    {
                        "inputs": {"input": float_dynamic},
                        "params": {"shape": ["B", "S", 2, 2]},
                        "declaredOutputs": {
                            "out": _logical_tensor(["B", "S", 2, 2])
                        },
                    },
                    {"out": (("B", "S", 2, 2), "float32")},
                ),
                "Expand": (
                    {
                        "inputs": {"input": _logical_tensor(["B", 1, 4])},
                        "params": {"shape": ["B", "S", 4]},
                        "declaredOutputs": {
                            "out": _logical_tensor(["B", "S", 4])
                        },
                    },
                    {"out": (("B", "S", 4), "float32")},
                ),
                "Concat": (
                    {
                        "inputs": {
                            "input0": _logical_tensor(["B", 2, 4]),
                            "input1": _logical_tensor(["B", 3, 4]),
                        },
                        "params": {"axis": 1},
                    },
                    {"out": (("B", 5, 4), "float32")},
                ),
                "Split": (
                    {
                        "inputs": {"input": _logical_tensor(["B", 6, 4])},
                        "params": {"axis": 1, "split": [2, 4]},
                    },
                    {
                        "out0": (("B", 2, 4), "float32"),
                        "out1": (("B", 4, 4), "float32"),
                    },
                ),
                "Slice": (
                    {
                        "inputs": {"input": float_dynamic},
                        "params": {
                            "starts": [0],
                            "ends": [16],
                            "axes": [1],
                        },
                    },
                    {"out": (("B", "S", 4), "float32")},
                ),
                "Pad": (
                    {
                        "inputs": {"input": float_dynamic},
                        "params": {"pads": [0, 0, 1, 0, 0, 1]},
                    },
                    {"out": (("B", "S", 6), "float32")},
                ),
                "Gather": (
                    {
                        "inputs": {
                            "input": _logical_tensor(["B", 16, 4]),
                            "indices": _logical_tensor(["S"], "int32"),
                        },
                        "params": {"axis": 1},
                    },
                    {"out": (("B", "S", 4), "float32")},
                ),
                "GatherElements": (
                    {
                        "inputs": {
                            "input": float_dynamic,
                            "indices": _logical_tensor(["B", "S", 2], "int32"),
                        },
                        "params": {"axis": 2},
                    },
                    {"out": (("B", "S", 2), "float32")},
                ),
            }
        )

        self.assertEqual(
            set(requests), set(STRUCTURAL_OPERATOR_SHAPE_CONTRACTS)
        )
        for operator, (request, expected_outputs) in requests.items():
            with self.subTest(operator=operator):
                proof = self._prove(operator, request)
                self.assertTrue(proof.supported, getattr(proof, "reason", ""))
                self.assertEqual(set(proof.outputs), set(expected_outputs))
                for name, (shape, dtype) in expected_outputs.items():
                    self.assertEqual(proof.outputs[name].shape, shape)
                    self.assertEqual(proof.outputs[name].dtype, dtype)

    def test_unrepresentable_structural_dynamic_relations_fail_closed(self):
        cases = (
            (
                "Sub",
                {
                    "inputs": {
                        "a": _logical_tensor(["B", "S", 4]),
                        "b": _logical_tensor(["B", "T", 4]),
                    }
                },
                "UNPROVABLE_DYNAMIC_BROADCAST",
            ),
            (
                "Equal",
                {
                    "inputs": {
                        "a": _logical_tensor(["B", "S", 4], "int32"),
                        "b": _logical_tensor(["B", "T", 4], "int32"),
                    }
                },
                "UNPROVABLE_DYNAMIC_BROADCAST",
            ),
            (
                "Where",
                {
                    "inputs": {
                        "condition": _logical_tensor(["B", "T", 4], "int32"),
                        "a": _logical_tensor(["B", "S", 4]),
                        "b": _logical_tensor(["B", "S", 4]),
                    }
                },
                "UNPROVABLE_DYNAMIC_BROADCAST",
            ),
            (
                "Flatten",
                {
                    "inputs": {"input": _logical_tensor(["B", "S", 4])},
                    "params": {"axis": 1},
                },
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "Squeeze",
                {
                    "inputs": {"input": _logical_tensor(["B", "S", 4])},
                    "params": {"axes": [1]},
                },
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "Reshape",
                {
                    "inputs": {"input": _logical_tensor(["B", "S", 4])},
                    "params": {"shape": ["B", "T", 2, 2]},
                },
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "Expand",
                {
                    "inputs": {"input": _logical_tensor(["B", "S", 4])},
                    "params": {"shape": ["B", "T", 4]},
                },
                "UNPROVABLE_DYNAMIC_BROADCAST",
            ),
            (
                "Concat",
                {
                    "inputs": {
                        "input0": _logical_tensor(["B", "S", 4]),
                        "input1": _logical_tensor(["B", "S", 4]),
                    },
                    "params": {"axis": 1},
                },
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "Split",
                {
                    "inputs": {"input": _logical_tensor(["B", "S", 4])},
                    "params": {"axis": 1, "num_outputs": 2},
                },
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "Slice",
                {
                    "inputs": {"input": _logical_tensor(["B", "S", 4])},
                    "params": {"starts": [1], "ends": [16], "axes": [1]},
                },
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "Pad",
                {
                    "inputs": {"input": _logical_tensor(["B", "S", 4])},
                    "params": {"pads": [0, 1, 0, 0, 0, 0]},
                },
                "UNPROVABLE_DYNAMIC_SHAPE_FORMULA",
            ),
            (
                "GatherElements",
                {
                    "inputs": {
                        "input": _logical_tensor(["B", "S", 4]),
                        "indices": _logical_tensor(["B", "T", 2], "int32"),
                    },
                    "params": {"axis": 2},
                },
                "INVALID_DOMAIN",
            ),
        )
        for operator, request, expected_code in cases:
            with self.subTest(operator=operator):
                proof = self._prove(operator, request)
                self.assertFalse(proof.supported)
                self.assertEqual(proof.code, expected_code, proof.reason)


if __name__ == "__main__":
    unittest.main()
