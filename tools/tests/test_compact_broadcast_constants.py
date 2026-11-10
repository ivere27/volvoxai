"""Large repeated ONNX constants stay compact without changing execution."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from tools.export_safetensors import export_onnx_model

ROOT = Path(__file__).resolve().parents[2]


class CompactBroadcastConstantTests(unittest.TestCase):
    def export(self, root, rows, columns, *, where=False):
        x = np.ones((rows, columns), dtype=np.float32)
        if where:
            constants = [numpy_helper.from_array((np.arange(columns) % 2 == 0)[None, :], "mask"),
                         numpy_helper.from_array(np.array([0], dtype=np.float32), "zero")]
            node = helper.make_node("Where", ["mask", "x", "zero"], ["y"])
        else:
            constants = [numpy_helper.from_array((np.arange(columns) % 2)[None, :].astype(np.float32), "bias")]
            node = helper.make_node("Add", ["x", "bias"], ["y"])
        model = helper.make_model(helper.make_graph([node], "compact_broadcast",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, x.shape)], constants),
            opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        expected = ort.InferenceSession(model.SerializeToString(), options,
            providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        onnx.save(model, root / "source.onnx")
        package = root / "package"
        package.mkdir()
        graph = export_onnx_model(str(root / "source.onnx"), str(package / "model.safetensors"))
        return package, graph, x, expected

    def test_add_keeps_broadcast_constants_compact_at_all_sizes(self):
        for rows, columns in ((64, 256), (2, 9000), (4, 5000)):
            with self.subTest(rows=rows, columns=columns), tempfile.TemporaryDirectory() as directory:
                package, graph, _, _ = self.export(Path(directory), rows, columns)
                self.assertFalse(any(node["opType"] == "Expand" for node in graph["nodes"]))
                size = (package / "model.safetensors").stat().st_size
                self.assertLess(size, columns * 4 + 4096)
                self.assertGreaterEqual(size, columns * 4)

    def test_compact_add_and_boolean_where_match_onnx_on_cpu_and_vulkan(self):
        binary = ROOT / "native/volvoxai"
        if not binary.exists():
            self.skipTest("native full binary required for required-backend numerical parity")
        for where in (False, True):
            with self.subTest(where=where), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package, graph, x, expected = self.export(root, 257 if where else 65, 257, where=where)
                self.assertEqual(any(node["opType"] == "Expand" for node in graph["nodes"]), where)
                self.assertLess((package / "model.safetensors").stat().st_size, 8192)
                input_file = root / "input.f32"
                x.tofile(input_file)
                for backend in ("cpu", "vulkan"):
                    with self.subTest(backend=backend):
                        output, report = root / f"{backend}.f32", root / f"{backend}.json"
                        command = [str(binary), "run", str(package), "--" + backend,
                            "--input", f"input0={input_file}", "--output", f"output0={output}",
                            "--report-json", str(report)]
                        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        evidence = json.loads(report.read_text())
                        for phase in ("compile", "execute"):
                            self.assertEqual(evidence[phase]["backend"], backend)
                            route = evidence[phase]["route"]
                            self.assertTrue(route["builtin"] and route["attested"])
                            self.assertEqual(route["activeNodes"], route["selectedNodes"])
                            self.assertEqual((route["fallbackNodes"], route["missingNodes"]), (0, 0))
                        actual = np.fromfile(output, dtype=np.float32).reshape(expected.shape)
                        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
