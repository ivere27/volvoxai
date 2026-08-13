from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from tools.exporter.shape_system import (
    CanonicalShapeInput,
    MAX_SAFE_INTEGER,
    MIN_SAFE_INTEGER,
    PublicInputShapeContract,
    ShapeContractError,
    ShapeEnvironment,
    ShapedRuntimeTensorView,
    bind_public_input_shapes,
    canonical_shape_signature,
    checked_shape_add,
    checked_shape_ceil_divide,
    checked_shape_element_count,
    checked_shape_floor_divide,
    checked_shape_multiply,
    checked_shape_subtract,
    checked_tensor_byte_length,
    create_tensor_shape_spec,
)


_VECTORS_PATH = Path(__file__).resolve().parents[3] / "tests" / "shape_system_vectors.json"


class _OpaqueData:
    """Mutable data stand-in used to prove binding retains, but never reads it."""

    def __init__(self, label: str):
        self.label = label
        self.contents = [label]


def _load_vectors() -> dict:
    return json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))


def _make_views(raw_views: dict) -> tuple[dict, dict, dict]:
    views = {}
    data_objects = {}
    caller_shapes = {}
    for name, raw in raw_views.items():
        data = _OpaqueData(name)
        shape = list(raw["shape"])
        data_objects[name] = data
        caller_shapes[name] = shape
        views[name] = ShapedRuntimeTensorView(
            data=data,
            shape=shape,
            dtype=raw["dtype"],
            byte_length=raw["byte_length"],
        )
    return views, data_objects, caller_shapes


class ShapeSystemVectorTests(unittest.TestCase):
    def test_shared_success_vectors(self):
        vectors = _load_vectors()
        self.assertEqual(vectors["format"], "volvox-shape-system-vectors/v1")
        self.assertEqual(vectors["max_safe_integer"], MAX_SAFE_INTEGER)

        for original in vectors["success_cases"]:
            with self.subTest(case=original["id"]):
                case = deepcopy(original)
                before = deepcopy(case)
                environment = ShapeEnvironment(case["dimensions"])
                contract = PublicInputShapeContract(environment, case["inputs"])
                views, data_objects, caller_shapes = _make_views(case["views"])
                data_before = {
                    name: list(data.contents) for name, data in data_objects.items()
                }

                binding = bind_public_input_shapes(contract, views)
                expected = case["expected"]
                self.assertEqual(binding.signature, expected["signature"])
                self.assertEqual(binding.symbols, tuple(
                    (name, value) for name, value in expected["symbols"]
                ))
                self.assertEqual(
                    tuple(bound.name for bound in binding.inputs),
                    tuple(expected["input_order"]),
                )
                for bound in binding.inputs:
                    self.assertEqual(
                        bound.element_count,
                        expected["element_counts"][bound.name],
                    )
                    self.assertEqual(
                        bound.size_bytes,
                        expected["byte_lengths"][bound.name],
                    )
                    self.assertIs(bound.data, data_objects[bound.name])

                # Binding is read-only and leaves all caller-owned metadata and
                # contents unchanged.
                self.assertEqual(case, before)
                self.assertEqual(
                    {name: view.shape for name, view in views.items()},
                    caller_shapes,
                )
                self.assertEqual(
                    {
                        name: data.contents
                        for name, data in data_objects.items()
                    },
                    data_before,
                )
                self.assertIsInstance(binding.inputs, tuple)
                self.assertIsInstance(binding.symbols, tuple)
                self.assertTrue(all(
                    isinstance(bound.shape, tuple) for bound in binding.inputs
                ))

                # The contract and binding snapshot mutable source shape lists.
                if caller_shapes:
                    first_name = next(iter(caller_shapes))
                    bound = binding.input(first_name)
                    assert bound is not None
                    bound_shape = bound.shape
                    caller_shapes[first_name][0] = 1
                    self.assertEqual(bound.shape, bound_shape)

    def test_shared_failure_vectors_have_exact_code_and_path(self):
        vectors = _load_vectors()
        for case in vectors["failure_cases"]:
            with self.subTest(case=case["id"]):
                expected = case["expected_error"]
                with self.assertRaises(ShapeContractError) as caught:
                    if case["operation"] == "environment":
                        ShapeEnvironment(case["dimensions"])
                    elif case["operation"] == "bind":
                        environment = ShapeEnvironment(case["dimensions"])
                        contract = PublicInputShapeContract(
                            environment, case["inputs"]
                        )
                        views, _, _ = _make_views(case["views"])
                        bind_public_input_shapes(contract, views)
                    elif case["operation"] == "tensor_byte_length":
                        checked_tensor_byte_length(
                            case["shape"], case["dtype"], case["path"]
                        )
                    else:  # pragma: no cover - corpus schema assertion
                        self.fail(f"unknown vector operation {case['operation']}")
                self.assertEqual(caught.exception.code, expected["code"])
                self.assertEqual(caught.exception.path, expected["path"])

    def test_vector_symbol_length_boundary_is_exact(self):
        vectors = _load_vectors()
        long_case = next(
            case for case in vectors["failure_cases"]
            if case["id"] == "symbol-name-over-64-characters"
        )
        self.assertEqual(len(long_case["dimensions"][0]["name"]), 65)


class ShapeArithmeticTests(unittest.TestCase):
    def assert_shape_error(self, code: str, path: str, operation):
        with self.assertRaises(ShapeContractError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.path, path)
        self.assertTrue(str(caught.exception).startswith(f"{path}: "))

    def test_signed_safe_arithmetic_and_exact_element_bytes(self):
        self.assertEqual(
            checked_shape_add(MAX_SAFE_INTEGER - 1, 1), MAX_SAFE_INTEGER
        )
        self.assertEqual(
            checked_shape_subtract(MIN_SAFE_INTEGER + 1, 1), MIN_SAFE_INTEGER
        )
        self.assertEqual(checked_shape_multiply(-3, 7), -21)
        self.assertEqual(checked_shape_floor_divide(8, 3), 2)
        self.assertEqual(checked_shape_ceil_divide(8, 3), 3)
        self.assertEqual(checked_shape_floor_divide(-8, 3), -3)
        self.assertEqual(checked_shape_ceil_divide(-8, 3), -2)
        self.assertEqual(checked_shape_element_count((2, 3, 5)), 30)
        self.assertEqual(checked_shape_element_count(()), 1)
        self.assertEqual(checked_tensor_byte_length((2, 3, 5), "float32"), 120)

    def test_arithmetic_rejects_invalid_operands_and_overflow(self):
        for operation, path in (
            (lambda: checked_shape_add(MAX_SAFE_INTEGER, 1, "sum"), "sum"),
            (
                lambda: checked_shape_subtract(
                    MIN_SAFE_INTEGER, 1, "difference"
                ),
                "difference",
            ),
            (
                lambda: checked_shape_multiply(
                    MAX_SAFE_INTEGER, 2, "product"
                ),
                "product",
            ),
            (
                lambda: checked_shape_element_count(
                    (MAX_SAFE_INTEGER, 2), "huge"
                ),
                "huge element count",
            ),
        ):
            with self.subTest(path=path):
                self.assert_shape_error(
                    "ARITHMETIC_OVERFLOW", path, operation
                )

        self.assert_shape_error(
            "ARITHMETIC_INVALID",
            "shape floor division divisor",
            lambda: checked_shape_floor_divide(1, 0),
        )
        self.assert_shape_error(
            "ARITHMETIC_INVALID",
            "shape addition left operand",
            lambda: checked_shape_add(True, 1),
        )


class ShapeContractConstructionTests(unittest.TestCase):
    def assert_shape_error(self, code: str, path: str, operation):
        with self.assertRaises(ShapeContractError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.path, path)

    def test_constraints_and_shape_specs_are_immutable_snapshots(self):
        constraint = {"name": "B", "min": 1, "max": 4}
        source_shape = ["B", 2]
        environment = ShapeEnvironment([constraint])
        contract = PublicInputShapeContract(environment, [
            {"name": "x", "dtype": "int8", "shape": source_shape}
        ])

        constraint["min"] = 3
        source_shape[0] = 4
        self.assertEqual(environment.dimensions[0].min, 1)
        self.assertEqual(contract.inputs[0].shape, ("B", 2))
        with self.assertRaises(FrozenInstanceError):
            environment.dimensions[0].min = 2
        with self.assertRaises(FrozenInstanceError):
            contract.inputs[0].dtype = "uint8"

    def test_symbol_grammar_bounds_domain_and_unknown_references(self):
        valid_name = f"B{'x' * 63}"
        environment = ShapeEnvironment([
            {"name": valid_name, "min": 1, "max": MAX_SAFE_INTEGER},
            {"name": "S", "min": 2, "max": 4, "multiple_of": 2},
        ])
        self.assertEqual(len(valid_name), 64)
        self.assertEqual(
            create_tensor_shape_spec([valid_name, "S", 3], environment),
            (valid_name, "S", 3),
        )

        self.assert_shape_error(
            "UNKNOWN_SYMBOL",
            "tensor shape spec[0]",
            lambda: create_tensor_shape_spec(["Unknown"], environment),
        )
        self.assert_shape_error(
            "INVALID_CONSTRAINT",
            "dimension constraints[0].min",
            lambda: ShapeEnvironment([{"name": "B", "min": True, "max": 2}]),
        )
        self.assert_shape_error(
            "INVALID_CONSTRAINT",
            "dimension constraints[0].max",
            lambda: ShapeEnvironment([
                {"name": "B", "min": 1, "max": MAX_SAFE_INTEGER + 1}
            ]),
        )

    def test_canonical_signature_is_order_independent_and_byte_counted(self):
        first = canonical_shape_signature([
            CanonicalShapeInput("é", [3]),
            CanonicalShapeInput("z", [2]),
            CanonicalShapeInput("a", [1]),
        ])
        second = canonical_shape_signature([
            CanonicalShapeInput("a", [1]),
            CanonicalShapeInput("é", [3]),
            CanonicalShapeInput("z", [2]),
        ])
        self.assertEqual(first, second)
        self.assertEqual(first, "v1|1:a|1:1|1:z|1:2|2:é|1:3")
        self.assertNotEqual(
            canonical_shape_signature([CanonicalShapeInput("a", [1, 4])]),
            canonical_shape_signature([CanonicalShapeInput("a", [2, 2])]),
        )


class ShapeBindingValidationTests(unittest.TestCase):
    def setUp(self):
        environment = ShapeEnvironment([
            {"name": "B", "min": 1, "max": 4},
            {"name": "S", "min": 2, "max": 16, "multiple_of": 2},
        ])
        self.contract = PublicInputShapeContract(environment, [
            {"name": "tokens", "dtype": "int32", "shape": ["B", "S"]},
            {"name": "mask", "dtype": "int32", "shape": ["B", "S"]},
        ])

    @staticmethod
    def view(dtype="int32", shape=(2, 4), byte_length=32, data=None):
        return ShapedRuntimeTensorView(
            data=_OpaqueData("view") if data is None else data,
            shape=list(shape),
            dtype=dtype,
            byte_length=byte_length,
        )

    def assert_bind_error(self, values, code: str, path: str):
        shapes_before = {
            name: list(view.shape)
            for name, view in values.items()
            if isinstance(view, ShapedRuntimeTensorView)
        }
        contents_before = {
            name: list(view.data.contents)
            for name, view in values.items()
            if isinstance(view, ShapedRuntimeTensorView)
            and isinstance(view.data, _OpaqueData)
        }
        with self.assertRaises(ShapeContractError) as caught:
            bind_public_input_shapes(self.contract, values)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.path, path)
        self.assertEqual(
            shapes_before,
            {
                name: list(view.shape)
                for name, view in values.items()
                if isinstance(view, ShapedRuntimeTensorView)
            },
        )
        self.assertEqual(
            contents_before,
            {
                name: list(view.data.contents)
                for name, view in values.items()
                if isinstance(view, ShapedRuntimeTensorView)
                and isinstance(view.data, _OpaqueData)
            },
        )

    def test_exact_names_dtype_rank_and_byte_length(self):
        valid = self.view()
        self.assert_bind_error(
            {"tokens": valid},
            "INVALID_INPUT_SET",
            "execution inputs",
        )
        self.assert_bind_error(
            {"tokens": valid, "mask": self.view(), "extra": self.view()},
            "INVALID_INPUT_SET",
            "execution inputs",
        )
        self.assert_bind_error(
            {
                "tokens": self.view(dtype="float32"),
                "mask": self.view(),
            },
            "DTYPE_MISMATCH",
            "execution input 'tokens'.data",
        )
        self.assert_bind_error(
            {
                "tokens": self.view(shape=(8,), byte_length=32),
                "mask": self.view(),
            },
            "RANK_MISMATCH",
            "execution input 'tokens'.shape",
        )
        self.assert_bind_error(
            {
                "tokens": self.view(byte_length=28),
                "mask": self.view(),
            },
            "BYTE_LENGTH_MISMATCH",
            "execution input 'tokens'.data",
        )

    def test_failure_is_atomic_and_success_retains_opaque_data(self):
        tokens_data = _OpaqueData("tokens")
        mask_data = _OpaqueData("mask")
        tokens = self.view(data=tokens_data)
        conflicting = self.view(shape=(1, 8), data=mask_data)
        self.assert_bind_error(
            {"tokens": tokens, "mask": conflicting},
            "SYMBOL_CONFLICT",
            "execution input 'tokens'.shape[0]",
        )

        mask = self.view(data=mask_data)
        binding = bind_public_input_shapes(
            self.contract, {"tokens": tokens, "mask": mask}
        )
        self.assertIs(binding.input("tokens").data, tokens_data)
        self.assertIs(binding.input("mask").data, mask_data)


if __name__ == "__main__":
    unittest.main()
