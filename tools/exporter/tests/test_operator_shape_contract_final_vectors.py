from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import struct
import unittest

from tools.exporter.operator_shape_contracts import (
    OperatorShapeContractError,
    get_operator_shape_contract,
    infer_concrete_operator_shapes,
)
from tools.exporter.shape_system import ShapeEnvironment


_VECTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "operator_shape_contract_final_vectors.json"
)
_EXPECTED_OPERATORS = {
    "MoERouter", "MoELinear", "CrossAttention", "BatchNorm2D",
    "Interpolate1D", "Not", "Mask", "Broadcast", "Concat2",
    "RequantizeLinear", "SSMScan", "SelectiveScan",
    "SpatialSoftargmaxY", "MeanHeight", "ProfileX", "ProfileY", "Dropout",
}


def _outputs_json(outputs: object) -> dict:
    result: dict[str, dict] = {}
    for name, descriptor in outputs.items():
        item = {"shape": list(descriptor.shape), "dtype": descriptor.dtype}
        quantization = descriptor.quantization
        if quantization is not None:
            if quantization.scheme == "per_tensor":
                item["quantization"] = {
                    "scheme": "per_tensor",
                    "scale": quantization.scale,
                    "zero_point": quantization.zero_point,
                }
            else:
                item["quantization"] = {
                    "scheme": "per_axis",
                    "axis": quantization.axis,
                    "scales": list(quantization.scales),
                    "zero_points": list(quantization.zero_points),
                }
        result[name] = item
    return result


def _canonical_expected(outputs: dict) -> dict:
    result = deepcopy(outputs)
    for descriptor in result.values():
        quantization = descriptor.get("quantization")
        if quantization is None:
            continue
        if quantization["scheme"] == "per_tensor":
            quantization["scale"] = struct.unpack(
                "!f", struct.pack("!f", quantization["scale"])
            )[0]
        else:
            quantization["scales"] = [
                struct.unpack("!f", struct.pack("!f", scale))[0]
                for scale in quantization["scales"]
            ]
    return result


class FinalSharedVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vectors = json.loads(_VECTOR_PATH.read_text(encoding="utf-8"))

    def test_manifest_and_coverage_are_closed(self):
        manifest = {
            operator
            for family in self.vectors["families"]
            for operator in family["operators"]
        }
        self.assertEqual(manifest, _EXPECTED_OPERATORS)
        success = {operator: 0 for operator in manifest}
        failure = {operator: 0 for operator in manifest}
        for cases, coverage in (
            (self.vectors["success_cases"], success),
            (self.vectors["failure_cases"], failure),
        ):
            for vector in cases:
                for operator in vector["operators"]:
                    coverage[operator] += 1
        for operator in manifest:
            self.assertGreaterEqual(success[operator], 1)
            self.assertGreaterEqual(failure[operator], 1)

    def test_legal_vectors_converge_in_concrete_and_domain_contracts(self):
        for original in self.vectors["success_cases"]:
            for operator in original["operators"]:
                vector = deepcopy(original)
                before = deepcopy(vector)
                contract = get_operator_shape_contract(operator)
                self.assertEqual(
                    contract.shape_function_id,
                    vector["expected_shape_function_id"],
                )
                expected = _canonical_expected(vector["expected"])
                self.assertEqual(
                    _outputs_json(
                        infer_concrete_operator_shapes(operator, vector["request"])
                    ),
                    expected,
                    f"{vector['id']}:{operator}:concrete",
                )
                proof = contract.prove_domain(
                    {
                        **vector["request"],
                        "environment": ShapeEnvironment([]),
                    }
                )
                self.assertTrue(
                    proof.supported,
                    f"{vector['id']}:{operator}:{getattr(proof, 'reason', '')}",
                )
                self.assertEqual(_outputs_json(proof.outputs), expected)
                self.assertEqual(vector, before)

    def test_illegal_vectors_have_stable_codes_and_paths(self):
        for vector in self.vectors["failure_cases"]:
            for operator in vector["operators"]:
                with self.assertRaises(OperatorShapeContractError) as caught:
                    infer_concrete_operator_shapes(operator, vector["request"])
                self.assertEqual(caught.exception.code, vector["expected_error"]["code"])
                self.assertEqual(caught.exception.path, vector["expected_error"]["path"])
                proof = get_operator_shape_contract(operator).prove_domain(
                    {
                        **vector["request"],
                        "environment": ShapeEnvironment([]),
                    }
                )
                self.assertFalse(proof.supported)

    def test_dynamic_language_vision_and_scan_axes_are_preserved(self):
        environment = ShapeEnvironment(
            [
                {"name": "B", "min": 1, "max": 4},
                {"name": "S", "min": 1, "max": 32},
                {"name": "Q", "min": 1, "max": 16},
                {"name": "K", "min": 1, "max": 24},
                {"name": "H", "min": 2, "max": 32},
                {"name": "W", "min": 2, "max": 32},
            ]
        )

        def f32(shape):
            return {"shape": shape, "dtype": "float32"}

        def i32(shape):
            return {"shape": shape, "dtype": "int32"}

        pt = {"scheme": "per_tensor", "scale": 0.5, "zero_point": 0}

        def byte(shape, dtype="int8", quantization=None):
            return {
                "shape": shape,
                "dtype": dtype,
                "quantization": pt if quantization is None else quantization,
            }

        cases = [
            ("MoERouter", {
                "inputs": {"input": f32(["B", "S", 4]), "weight": f32([4, 3])},
                "params": {"top_k": 2},
            }, {"indices": ("B", "S", 2), "weights": ("B", "S", 2)}),
            ("MoELinear", {
                "inputs": {
                    "input": f32(["B", "S", 4]),
                    "expert_weight": f32([3, 4, 5]),
                    "route_indices": f32(["B", "S", 2]),
                    "route_weights": f32(["B", "S", 2]),
                }, "params": {},
            }, {"out": ("B", "S", 5)}),
            ("CrossAttention", {
                "inputs": {
                    "q": f32(["B", "Q", 4]), "kv": f32(["B", "K", 4]),
                    "weight": f32([12, 4]),
                }, "params": {"heads": 2},
            }, {"out": ("B", "Q", 4)}),
            ("BatchNorm2D", {
                "inputs": {
                    "input": f32(["B", "H", "W", 5]), "weight": f32([5]),
                    "bias": f32([5]), "running_mean": f32([5]),
                    "running_var": f32([5]),
                }, "params": {},
            }, {"out": ("B", "H", "W", 5)}),
            ("Interpolate1D", {
                "inputs": {"input": f32(["B", 3, "S"])}, "params": {"size": 7},
            }, {"out": ("B", 3, 7)}),
            ("Not", {"inputs": {"input": i32(["B", "S"])}, "params": {}},
             {"out": ("B", "S")}),
            ("Mask", {
                "inputs": {
                    "mask": i32(["B", "S"]), "a": f32(["B", "S"]),
                    "b": f32(["B", "S"]),
                }, "params": {},
            }, {"out": ("B", "S")}),
            ("Broadcast", {
                "inputs": {"input": f32(["B", 1])}, "params": {"shape": ["B", "S"]},
            }, {"out": ("B", "S")}),
            ("Concat2", {
                "inputs": {"a": f32(["B", 2]), "b": f32(["B", 3])},
                "params": {"axis": 1},
            }, {"out": ("B", 5)}),
            ("RequantizeLinear", {
                "inputs": {"input": byte(["B", "S"])}, "params": {},
                "declaredOutputs": {
                    "out": byte(
                        ["B", "S"], "uint8",
                        {"scheme": "per_tensor", "scale": 0.5, "zero_point": 128},
                    )
                },
            }, {"out": ("B", "S")}),
            ("SSMScan", {
                "inputs": {
                    "input": f32(["B", "S", 4]), "delta": f32(["B", "S", 4]),
                    "A": f32([4, 5]), "B": f32([5]), "C": f32(["S", 5]),
                }, "params": {},
            }, {"out": ("B", "S", 4), "state": ("B", 4, 5)}),
            ("SelectiveScan", {
                "inputs": {
                    "input": f32(["B", "S", 4]), "delta": f32(["B", "S", 4]),
                    "A": f32([4, 5]), "B": f32(["B", "S", 5]), "C": f32([5]),
                }, "params": {},
            }, {"out": ("B", "S", 4), "state": ("B", 4, 5)}),
            ("SpatialSoftargmaxY", {
                "inputs": {"input": f32(["B", "H", "W", 5])}, "params": {},
            }, {"out": ("B", 5, "W")}),
            ("MeanHeight", {
                "inputs": {"input": f32(["B", "H", "W", 5])}, "params": {},
            }, {"out": ("B", 5, "W")}),
            ("ProfileX", {
                "inputs": {"input": f32(["B", "H", "W", 5])}, "params": {},
            }, {"out": ("B", 10, "W")}),
            ("ProfileY", {
                "inputs": {"input": f32(["B", "H", "W", 5])}, "params": {},
            }, {"out": ("B", 10, "H")}),
            ("Dropout", {
                "inputs": {"input": f32(["B", "S", 4])}, "params": {"ratio": 0.25},
            }, {"out": ("B", "S", 4)}),
        ]
        for operator, request, expected_shapes in cases:
            proof = get_operator_shape_contract(operator).prove_domain(
                {**request, "environment": environment}
            )
            self.assertTrue(
                proof.supported,
                f"{operator}: {getattr(proof, 'reason', '')}",
            )
            self.assertEqual(
                {name: output.shape for name, output in proof.outputs.items()},
                expected_shapes,
                operator,
            )


if __name__ == "__main__":
    unittest.main()
