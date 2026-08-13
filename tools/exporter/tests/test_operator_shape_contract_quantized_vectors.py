from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from tools.exporter.operator_shape_contracts import (
    QUANTIZED_OPERATOR_SHAPE_CONTRACTS,
    OperatorShapeContractError,
    get_operator_shape_contract,
    infer_concrete_operator_shapes,
)
from tools.exporter.shape_system import ShapeEnvironment


_VECTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "operator_shape_contract_quantized_vectors.json"
)


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
            import struct
            quantization["scale"] = struct.unpack(
                "!f", struct.pack("!f", quantization["scale"])
            )[0]
        else:
            import struct
            quantization["scales"] = [
                struct.unpack("!f", struct.pack("!f", scale))[0]
                for scale in quantization["scales"]
            ]
    return result


class QuantizedSharedVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vectors = json.loads(_VECTOR_PATH.read_text(encoding="utf-8"))

    def test_manifest_and_per_operator_coverage_are_closed(self):
        manifest = {
            operator
            for family in self.vectors["families"]
            for operator in family["operators"]
        }
        self.assertEqual(manifest, set(QUANTIZED_OPERATOR_SHAPE_CONTRACTS))
        legal = {operator: 0 for operator in manifest}
        illegal = {operator: 0 for operator in manifest}
        for cases, coverage in (
            (self.vectors["success_cases"], legal),
            (self.vectors["failure_cases"], illegal),
        ):
            for vector in cases:
                for operator in vector["operators"]:
                    coverage[operator] += 1
        for operator in manifest:
            self.assertGreaterEqual(legal[operator], 2)
            self.assertGreaterEqual(illegal[operator], 1)

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

    def test_illegal_vectors_have_exact_concrete_codes_and_paths(self):
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

    def test_dynamic_bounded_axes_are_preserved(self):
        environment = ShapeEnvironment(
            [
                {"name": "B", "min": 1, "max": 4},
                {"name": "S", "min": 1, "max": 32},
                {"name": "Q", "min": 1, "max": 16},
                {"name": "K", "min": 1, "max": 24},
                {"name": "H", "min": 3, "max": 32},
                {"name": "W", "min": 3, "max": 32},
            ]
        )

        def pt(scale=0.05, zero_point=0):
            return {"scheme": "per_tensor", "scale": scale, "zero_point": zero_point}

        def pa(count):
            return {
                "scheme": "per_axis",
                "axis": 0,
                "scales": [0.01 * (index + 1) for index in range(count)],
                "zero_points": [0] * count,
            }

        def byte(shape, quantization=None):
            return {
                "shape": shape,
                "dtype": "int8",
                "quantization": pt() if quantization is None else quantization,
            }

        def output(shape):
            return {"out": byte(shape, pt(0.08))}

        def i32(shape):
            return {"shape": shape, "dtype": "int32"}

        def f32(shape):
            return {"shape": shape, "dtype": "float32"}

        cases = [
            *[
                (
                    operator,
                    {
                        "inputs": {
                            "input": byte(["B", "S", 4]),
                            "weight": byte([3, 4], pa(3)),
                            "bias": i32([3]),
                        },
                        "params": {},
                        "declaredOutputs": output(["B", "S", 3]),
                    },
                )
                for operator in ("QLinear", "QMatMul", "QGemm")
            ],
            (
                "QBatchMatMul",
                {
                    "inputs": {
                        "a": byte(["B", "Q", 4]),
                        "b": byte([1, 4, 3]),
                    },
                    "params": {},
                    "declaredOutputs": output(["B", "Q", 3]),
                },
            ),
            (
                "QConv2D",
                {
                    "inputs": {
                        "input": byte(["B", "H", "W", 2]),
                        "weight": byte([3, 3, 3, 2], pa(3)),
                    },
                    "params": {
                        "stride": [1, 1],
                        "padding": [1, 1],
                        "pads": [1, 1, 1, 1],
                        "data_layout": "NHWC",
                        "weight_layout": "OHWI",
                    },
                    "declaredOutputs": output(["B", "H", "W", 3]),
                },
            ),
            (
                "QAdd",
                {
                    "inputs": {
                        "a": byte(["B", "S", 4]),
                        "b": byte(["B", "S", 4]),
                    },
                    "params": {"relu": 0},
                    "declaredOutputs": output(["B", "S", 4]),
                },
            ),
            (
                "QEmbedding",
                {
                    "inputs": {
                        "input": i32(["B", "S"]),
                        "weight": byte([5, 4], pa(5)),
                    },
                    "params": {},
                    "declaredOutputs": output(["B", "S", 4]),
                },
            ),
            (
                "QGELU",
                {
                    "inputs": {"input": byte(["B", "S", 4])},
                    "params": {"approximate": "none"},
                    "declaredOutputs": output(["B", "S", 4]),
                },
            ),
            (
                "QSiLU",
                {
                    "inputs": {"input": byte(["B", "S", 4])},
                    "params": {},
                    "declaredOutputs": output(["B", "S", 4]),
                },
            ),
            (
                "QLayerNorm",
                {
                    "inputs": {
                        "input": byte(["B", "S", 4]),
                        "weight": f32([4]),
                        "bias": f32([4]),
                    },
                    "params": {"eps": 1e-5, "d_model": 4},
                    "declaredOutputs": output(["B", "S", 4]),
                },
            ),
            (
                "QGroupNorm",
                {
                    "inputs": {
                        "input": byte(["B", "H", "W", 4]),
                        "weight": f32([4]),
                        "bias": f32([4]),
                    },
                    "params": {
                        "num_groups": 2,
                        "eps": 1e-5,
                        "data_layout": "NHWC",
                    },
                    "declaredOutputs": output(["B", "H", "W", 4]),
                },
            ),
            (
                "QMaskedMean",
                {
                    "inputs": {
                        "input": byte(["B", "S", 4]),
                        "mask": i32(["B", "S"]),
                    },
                    "params": {},
                    "declaredOutputs": output(["B", 4]),
                },
            ),
            (
                "QSDPA",
                {
                    "inputs": {
                        "q": byte(["B", "Q", 8]),
                        "k": byte(["B", "K", 8]),
                        "v": byte(["B", "K", 8]),
                        "mask": i32(["B", "Q", "K"]),
                    },
                    "params": {"heads": 2, "causal": False, "scale": 0.5},
                    "declaredOutputs": output(["B", "Q", 8]),
                },
            ),
            (
                "QArgMax",
                {
                    "inputs": {"input": byte(["B", "S", 7])},
                    "params": {"axis": -1},
                },
            ),
        ]
        for operator, request in cases:
            proof = get_operator_shape_contract(operator).prove_domain(
                {"environment": environment, **request}
            )
            self.assertTrue(
                proof.supported,
                f"{operator}:{getattr(proof, 'reason', '')}",
            )


if __name__ == "__main__":
    unittest.main()
