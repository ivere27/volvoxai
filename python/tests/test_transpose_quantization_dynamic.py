"""Fused quantization transposes preserve every admitted concrete shape."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import volvoxai as vx


def package(root, channels, dtype, dequantize):
    nchw = ["B", channels, "H", "W"]
    nhwc = ["B", "H", "W", channels]
    byte_dtype = np.dtype(dtype).name
    zero = -3 if dtype == np.int8 else 117

    def node(name, op, inputs, output, shape, kind, params=None):
        return {"id": name, "opType": op, "inputs": inputs,
                "outputs": {"out": {"tensor": output, "shape": shape, "dtype": kind}},
                "params": params or {}}

    if dequantize:
        nodes = [node("dq", "DequantizeLinear", {"input": "x", "scale": "scale", "zero_point": "zero"},
                      "middle", nhwc, "float32"),
                 node("transpose", "Transpose", {"input": "middle"}, "y", nchw,
                      "float32", {"perm": [0, 3, 1, 2]})]
    else:
        nodes = [node("transpose", "Transpose", {"input": "x"}, "middle", nhwc,
                      "float32", {"perm": [0, 2, 3, 1]}),
                 node("q", "QuantizeLinear", {"input": "middle", "scale": "scale", "zero_point": "zero"},
                      "y", nhwc, byte_dtype)]
    graph = {"format": "volvox-graph/v1",
             "dimensions": {"B": {"min": 1, "max": 3}, "H": {"min": 1, "max": 3},
                            "W": {"min": 1, "max": 17}},
             "inputs": {"x": {"shape": nhwc if dequantize else nchw,
                               "dtype": byte_dtype if dequantize else "float32"}},
             "nodes": nodes, "outputs": ["y"],
             "execution": {"stateless": {}, "entrypoints": [
                 {"name": "evaluate", "effect": 1, "allowed_ranges": [1],
                  "inputs": ["x"], "outputs": ["y"]}]},
             "quantization": {"format": "volvox-affine-safetensors/v1", "tensors": {
                 "x" if dequantize else "y": {"scheme": "per_tensor",
                     "scale_tensor": "scale", "zero_point_tensor": "zero"}}}}
    (root / "graph.json").write_text(json.dumps(graph))
    save_file({"scale": np.array([.25], np.float32), "zero": np.array([zero], dtype)},
              str(root / "model.safetensors"))
    return zero


class DynamicTransposeQuantizationTest(unittest.TestCase):
    def test_fused_and_unfused_shape_changes_match_independent_oracle(self):
        # Grow and shrink the same context, crossing the SIMD tile boundaries
        # in both axes and retaining an earlier independently owned result.
        shapes = [(1, 1, 1), (2, 1, 3), (1, 3, 9), (3, 2, 17), (1, 1, 1)]
        for dequantize in (False, True):
            for dtype in (np.int8, np.uint8):
                for channels in (1, 3, 8, 9):
                    with self.subTest(dequantize=dequantize, dtype=dtype, channels=channels), \
                            tempfile.TemporaryDirectory(prefix="volvox-transpose-quant-") as directory:
                        root = Path(directory)
                        zero = package(root, channels, dtype, dequantize)
                        for fusion in ("0", "1"):
                            with patch.dict(os.environ, {"VOLVOX_DISABLE_OPERATOR_FUSION": fusion}), \
                                    vx.InferenceSession(root, backend="cpu", cpu_threads=1) as session:
                                retained = None
                                for iteration, (batch, height, width) in enumerate(shapes):
                                    count = batch * channels * height * width
                                    base = np.arange(count, dtype=np.float32) + iteration
                                    if dequantize:
                                        values = ((base % 251) + np.iinfo(dtype).min).astype(dtype)
                                        values = values.reshape(batch, height, width, channels)
                                        expected = ((values.astype(np.float32) - zero) * np.float32(.25)).transpose(0, 3, 1, 2)
                                    else:
                                        values = ((base % 173) - 80) * np.float32(.125)
                                        values = values.reshape(batch, channels, height, width)
                                        expected = np.clip(np.rint(values / np.float32(.25) + zero),
                                                           np.iinfo(dtype).min, np.iinfo(dtype).max)
                                        expected = expected.astype(dtype).transpose(0, 2, 3, 1)
                                    actual = session.run(values)["y"]
                                    self.assertEqual(actual.shape, expected.shape)
                                    self.assertEqual(actual.dtype, expected.dtype)
                                    np.testing.assert_array_equal(actual, expected)
                                    if retained is None:
                                        retained = actual, expected.copy()
                                np.testing.assert_array_equal(*retained)


if __name__ == "__main__":
    unittest.main()
